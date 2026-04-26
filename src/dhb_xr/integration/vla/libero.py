"""Libero dataset adapter (robomimic-style HDF5).

LIBERO stores end-effector data as:
- Position: obs/ee_pos (N, 3)
- Orientation: robot_states[:, 5:9] as quaternion (w, x, y, z)

Note: obs/ee_ori contains Euler angles, not quaternions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import h5py
    HAS_H5PY = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_H5PY = False


DEFAULT_POS_KEYS = (
    "robot0_eef_pos",
    "eef_pos",
    "ee_pos",
)
DEFAULT_QUAT_KEYS = (
    "robot0_eef_quat",
    "eef_quat",
    "ee_quat",
)
LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD = 0.015


def libero_obs_quat_wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    """Convert raw LIBERO env observation quaternions from wxyz to xyzw.

    The robomimic-style HDF5 files store EE quaternions as wxyz in
    ``robot_states[:, 5:9]``. On this machine, the direct LIBERO env
    observations under ``robot0_eef_quat`` match that same scalar-first
    convention. Most downstream libraries here expect scipy-style xyzw.
    """
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    return q[[1, 2, 3, 0]]


def libero_obs_quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert a raw LIBERO env observation quaternion (wxyz) to axis-angle."""
    quat_xyzw = libero_obs_quat_wxyz_to_xyzw(quat)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
    return Rotation.from_quat(quat_xyzw).as_rotvec().astype(np.float32)


def build_state_history_window(
    states: np.ndarray | Sequence[np.ndarray],
    *,
    history_window: int,
    state_dim: int = 8,
) -> np.ndarray:
    """Pad or trim a past-inclusive state history to a fixed ``(H, D)`` window.

    The training loader now feeds DHB with a short history window ending at the
    current step. Offline stage-1 and live rollout should mirror that contract
    instead of passing a single flattened state. Missing history is padded by
    repeating the oldest available state, matching the action-head behavior.
    """
    history_window = max(1, int(history_window))
    arr = np.asarray(states, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != state_dim:
            raise ValueError(
                f"Expected single state dim {state_dim}, got shape {tuple(arr.shape)}"
            )
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != state_dim:
        raise ValueError(
            f"Expected state history shape (H, {state_dim}), got {tuple(arr.shape)}"
        )
    if arr.shape[0] < history_window:
        pad = np.repeat(arr[:1], history_window - arr.shape[0], axis=0)
        arr = np.concatenate([pad, arr], axis=0)
    elif arr.shape[0] > history_window:
        arr = arr[-history_window:]
    return arr.astype(np.float32, copy=False)


def build_libero_state_vector(
    ee_pos: np.ndarray,
    ee_quat_wxyz: np.ndarray,
    grip_qpos: float,
) -> np.ndarray:
    """Build the canonical 8D LIBERO state vector used by StarVLA+DHB."""
    axis_angle = libero_obs_quat_to_axis_angle(ee_quat_wxyz)
    return np.concatenate(
        (
            np.asarray(ee_pos, dtype=np.float32),
            axis_angle.astype(np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([grip_qpos], dtype=np.float32),
        )
    ).astype(np.float32)


def build_libero_state_vector_from_obs(obs: dict) -> np.ndarray:
    return build_libero_state_vector(
        np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
        np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
        float(np.asarray(obs["robot0_gripper_qpos"][:1], dtype=np.float32)[0]),
    )


def get_state_history_window(cfg) -> int:
    framework = getattr(cfg, "framework", None)
    action_model = getattr(framework, "action_model", None) if framework is not None else None
    value = getattr(action_model, "state_history_window", 1) if action_model is not None else 1
    return max(1, int(value or 1))


def _extract_action_stats(payload: dict, stats_path: Path) -> dict:
    if not payload:
        raise ValueError(f"Dataset statistics file is empty: {stats_path}")

    if "action" in payload:
        action_stats = payload["action"]
    else:
        first_key = next(iter(payload.keys()))
        action_stats = payload[first_key]["action"]

    required = {"min", "max"}
    missing = required - set(action_stats.keys())
    if missing:
        raise KeyError(f"Dataset action stats missing required keys {sorted(missing)} in {stats_path}")
    return action_stats


def load_run_action_stats(output_dir: str | Path, *, checkpoint: str | None = None) -> dict:
    output_dir = Path(output_dir)
    candidate_paths = []
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        candidate_paths.append(checkpoint_path.parent.parent / "dataset_statistics.json")
    candidate_paths.append(output_dir / "dataset_statistics.json")

    seen = set()
    for stats_path in candidate_paths:
        if stats_path in seen:
            continue
        seen.add(stats_path)
        if stats_path.exists():
            payload = json.loads(stats_path.read_text())
            return _extract_action_stats(payload, stats_path)

    raise FileNotFoundError(
        "Missing dataset statistics. Looked in: " + ", ".join(str(path) for path in candidate_paths)
    )


def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: dict) -> np.ndarray:
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32).copy()
    mask = np.asarray(
        action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool)),
        dtype=bool,
    )
    action_high = np.asarray(action_norm_stats["max"], dtype=np.float32)
    action_low = np.asarray(action_norm_stats["min"], dtype=np.float32)
    normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
    normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0.0, 1.0)
    return np.where(
        mask,
        0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low,
        normalized_actions,
    )


def integrate_delta_action_chunk(
    start_pos: np.ndarray,
    start_quat_wxyz: np.ndarray,
    raw_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate delta-EF actions into absolute EE waypoints.

    Args:
        start_pos: Current EE position in world frame.
        start_quat_wxyz: Current EE quaternion in LIBERO `wxyz` convention.
        raw_actions: Action chunk shaped `(T, 7)` with
            `[dx, dy, dz, d_rx, d_ry, d_rz, next_open_gripper]`.

    Returns:
        positions: `(T, 3)` integrated EE positions.
        quaternions_wxyz: `(T, 4)` integrated EE quaternions in `wxyz`.
    """
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"Expected raw action chunk shape (T, 7), got {tuple(actions.shape)}")

    curr_pos = np.asarray(start_pos, dtype=np.float64).reshape(3).copy()
    curr_quat_xyzw = libero_obs_quat_wxyz_to_xyzw(np.asarray(start_quat_wxyz, dtype=np.float64))
    curr_quat_xyzw = curr_quat_xyzw / np.linalg.norm(curr_quat_xyzw)
    curr_rot = Rotation.from_quat(curr_quat_xyzw)

    positions = []
    quaternions_wxyz = []
    for action in actions:
        curr_pos = curr_pos + np.asarray(action[:3], dtype=np.float64)
        delta_rot = Rotation.from_rotvec(np.asarray(action[3:6], dtype=np.float64))
        curr_rot = curr_rot * delta_rot
        quat_xyzw = curr_rot.as_quat()
        positions.append(curr_pos.astype(np.float32))
        quaternions_wxyz.append(np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32))

    return np.asarray(positions, dtype=np.float32), np.asarray(quaternions_wxyz, dtype=np.float32)


def decode_contact_probs_to_gripper_cmds(
    contact_probs: np.ndarray,
    *,
    initial_gripper_cmd: float = -1.0,
    threshold: float = 0.5,
    contact_mode: str = "event",
    close_threshold: float | None = None,
    open_threshold: float | None = None,
    transition_debounce_steps: int = 1,
) -> np.ndarray:
    """Decode close/open event probabilities into persistent gripper commands.

    The contact head predicts event probabilities per step, not absolute open/close
    state. This decoder converts those event logits into the controller contract
    used throughout the repo: ``-1.0`` means open and ``1.0`` means closed.
    """
    probs = np.asarray(contact_probs, dtype=np.float32)
    is_closed = float(initial_gripper_cmd) > 0.0
    cmds = np.empty(len(probs), dtype=np.float32)

    if probs.ndim != 2 or probs.shape[-1] < 1:
        raise ValueError(f"Expected (T, 2+) contact probs, got shape {probs.shape}")

    if contact_mode == "gripper_state":
        if probs.shape[-1] >= 2:
            closed_probs = probs[:, 1]
        else:
            closed_probs = probs[:, 0]
        debounce_steps = max(1, int(transition_debounce_steps))
        if close_threshold is None and open_threshold is None:
            if debounce_steps == 1:
                return np.where(closed_probs >= threshold, 1.0, -1.0).astype(np.float32)
            close_thr = float(threshold)
            open_thr = float(threshold)
        else:
            close_thr = float(threshold if close_threshold is None else close_threshold)
            open_thr = float(threshold if open_threshold is None else open_threshold)
            if open_thr > close_thr:
                raise ValueError(
                    f"Expected open_threshold <= close_threshold for hysteresis, got "
                    f"open_threshold={open_thr} close_threshold={close_thr}"
                )

        pending_target: bool | None = None
        pending_count = 0

        for i, closed_prob in enumerate(closed_probs):
            target_closed: bool | None = None
            if is_closed:
                if float(closed_prob) <= open_thr:
                    target_closed = False
            else:
                if float(closed_prob) >= close_thr:
                    target_closed = True
            if target_closed is None or target_closed == is_closed:
                pending_target = None
                pending_count = 0
            else:
                if pending_target is None or pending_target != target_closed:
                    pending_target = target_closed
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= debounce_steps:
                    is_closed = target_closed
                    pending_target = None
                    pending_count = 0
            cmds[i] = 1.0 if is_closed else -1.0
        return cmds

    if probs.shape[-1] >= 3:
        event_ids = probs.argmax(axis=-1)
        for i, event_id in enumerate(event_ids):
            if int(event_id) == 1:
                is_closed = True
            elif int(event_id) == 2:
                is_closed = False
            cmds[i] = 1.0 if is_closed else -1.0
        return cmds

    close_probs = probs[:, 0]
    open_probs = probs[:, 1]
    for i, (close_p, open_p) in enumerate(zip(close_probs, open_probs)):
        if close_p >= threshold and close_p >= open_p:
            is_closed = True
        elif open_p >= threshold and open_p > close_p:
            is_closed = False
        cmds[i] = 1.0 if is_closed else -1.0
    return cmds


def decode_model_gripper_cmds(
    result: Dict,
    *,
    n_use: int,
    initial_gripper_cmd: float = -1.0,
    threshold: float = 0.5,
    close_threshold: float | None = None,
    open_threshold: float | None = None,
    transition_debounce_steps: int = 1,
    decode_source: str = "auto",
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Decode model output into controller gripper commands.

    Preference order:
    1. learned dense gripper-state probabilities
    2. learned close/open contact-phase probabilities from the DHB contact head
    3. learned close/open contact probabilities from the DHB contact head
    4. fallback binary open predictions from normalized action dim 6
    """
    valid_sources = {"auto", "gripper_state", "contact_phase", "contact", "normalized_action"}
    if decode_source not in valid_sources:
        raise ValueError(
            f"Unsupported model gripper decode source {decode_source!r}; "
            f"expected one of {sorted(valid_sources)}"
        )

    if decode_source in {"auto", "gripper_state"} and "pred_gripper_state_probs" in result and result["pred_gripper_state_probs"] is not None:
        gripper_state_probs = np.asarray(result["pred_gripper_state_probs"][0, :n_use], dtype=np.float32)
        cmds = decode_contact_probs_to_gripper_cmds(
            gripper_state_probs,
            initial_gripper_cmd=initial_gripper_cmd,
            threshold=threshold,
            contact_mode="gripper_state",
            close_threshold=close_threshold,
            open_threshold=open_threshold,
            transition_debounce_steps=transition_debounce_steps,
        )
        if gripper_state_probs.shape[-1] >= 2:
            closed_probs = gripper_state_probs[:, 1]
            open_probs = gripper_state_probs[:, 0]
        else:
            closed_probs = gripper_state_probs[:, 0]
            open_probs = 1.0 - closed_probs
        return cmds, {
            "source": "gripper_state_probs",
            "decode_source": decode_source,
            "state_probs": gripper_state_probs.tolist(),
            "close_probs": closed_probs.tolist(),
            "open_probs": open_probs.tolist(),
            "contact_phase_probs": (
                np.asarray(result["pred_contact_phase_probs"][0, :n_use], dtype=np.float32).tolist()
                if "pred_contact_phase_probs" in result and result["pred_contact_phase_probs"] is not None
                else None
            ),
            "threshold": float(threshold),
            "close_threshold": None if close_threshold is None else float(close_threshold),
            "open_threshold": None if open_threshold is None else float(open_threshold),
            "transition_debounce_steps": int(transition_debounce_steps),
        }

    if decode_source in {"auto", "contact_phase"} and "pred_contact_phase_probs" in result and result["pred_contact_phase_probs"] is not None:
        phase_probs = np.asarray(result["pred_contact_phase_probs"][0, :n_use], dtype=np.float32)
        cmds = decode_contact_probs_to_gripper_cmds(
            phase_probs,
            initial_gripper_cmd=initial_gripper_cmd,
            threshold=threshold,
            contact_mode="event",
            close_threshold=close_threshold,
            open_threshold=open_threshold,
            transition_debounce_steps=transition_debounce_steps,
        )
        return cmds, {
            "source": "contact_phase_probs",
            "decode_source": decode_source,
            "phase_probs": phase_probs.tolist(),
            "close_probs": phase_probs[:, 1].tolist() if phase_probs.shape[-1] >= 2 else [],
            "open_probs": phase_probs[:, 2].tolist() if phase_probs.shape[-1] >= 3 else [],
            "threshold": float(threshold),
            "transition_debounce_steps": int(transition_debounce_steps),
        }

    if decode_source in {"auto", "contact"} and "pred_contact_probs" in result and result["pred_contact_probs"] is not None:
        contact_probs = np.asarray(result["pred_contact_probs"][0, :n_use], dtype=np.float32)
        contact_mode = str(result.get("contact_mode", "event"))
        cmds = decode_contact_probs_to_gripper_cmds(
            contact_probs,
            initial_gripper_cmd=initial_gripper_cmd,
            threshold=threshold,
            contact_mode=contact_mode,
            close_threshold=close_threshold,
            open_threshold=open_threshold,
            transition_debounce_steps=transition_debounce_steps,
        )
        if contact_mode == "gripper_state":
            if contact_probs.shape[-1] >= 2:
                closed_probs = contact_probs[:, 1]
                open_probs = contact_probs[:, 0]
            else:
                closed_probs = contact_probs[:, 0]
                open_probs = 1.0 - closed_probs
            return cmds, {
                "source": "gripper_state_probs",
                "decode_source": decode_source,
                "state_probs": contact_probs.tolist(),
                "close_probs": closed_probs.tolist(),
                "open_probs": open_probs.tolist(),
                "threshold": float(threshold),
                "close_threshold": None if close_threshold is None else float(close_threshold),
                "open_threshold": None if open_threshold is None else float(open_threshold),
                "transition_debounce_steps": int(transition_debounce_steps),
            }
        return cmds, {
            "source": "contact_probs",
            "decode_source": decode_source,
            "event_probs": contact_probs.tolist(),
            "close_probs": (
                contact_probs[:, 1].tolist() if contact_probs.shape[-1] >= 3 else
                contact_probs[:, 0].tolist() if contact_probs.shape[-1] >= 1 else []
            ),
            "open_probs": (
                contact_probs[:, 2].tolist() if contact_probs.shape[-1] >= 3 else
                contact_probs[:, 1].tolist() if contact_probs.shape[-1] >= 2 else []
            ),
            "threshold": float(threshold),
            "transition_debounce_steps": int(transition_debounce_steps),
        }

    normalized_actions = np.asarray(result.get("normalized_actions", []), dtype=np.float32)
    if decode_source in {"auto", "normalized_action"} and normalized_actions.size:
        open_vals = normalized_actions[0, :n_use, 6]
        cmds = np.where(open_vals > 0.5, -1.0, 1.0).astype(np.float32)
        return cmds, {
            "source": "normalized_actions",
            "decode_source": decode_source,
            "open_vals": open_vals.tolist(),
        }

    if decode_source != "auto":
        raise ValueError(
            f"Requested model gripper decode source {decode_source!r}, but the model "
            "result did not contain the required probabilities."
        )
    return np.full(n_use, -1.0, dtype=np.float32), {"source": "default_open"}


@dataclass
class LiberoAdapter:
    """
    Minimal Libero adapter that yields (positions, quaternions, metadata).

    Libero datasets are typically robomimic-style HDF5 files.

    LIBERO stores quaternions in robot_states[:, 5:9] as (w, x, y, z),
    not in the obs group. This adapter handles both cases.
    """

    pos_keys: Tuple[str, ...] = DEFAULT_POS_KEYS
    quat_keys: Tuple[str, ...] = DEFAULT_QUAT_KEYS
    obs_group: str = "obs"
    # Robot states quaternion extraction (LIBERO-specific)
    robot_states_key: str = "robot_states"
    robot_states_quat_slice: Tuple[int, int] = field(default=(5, 9))
    robot_states_quat_format: str = "wxyz"  # LIBERO uses (w, x, y, z)

    def _find_key(self, group: "h5py.Group", candidates: Tuple[str, ...]) -> Optional[str]:
        for key in candidates:
            if key in group:
                return key
        return None

    def _extract_quaternion(
        self, demo: "h5py.Group", obs: "h5py.Group"
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Extract quaternions from demo, trying obs group first, then robot_states.

        Returns:
            (quaternions, source_key) where source_key describes where data came from.
        """
        # First try obs group
        quat_key = self._find_key(obs, self.quat_keys)
        if quat_key is not None:
            quats = np.asarray(obs[quat_key], dtype=np.float64)
            return quats, f"obs/{quat_key}"

        # Fallback: extract from robot_states (LIBERO-specific)
        if self.robot_states_key in demo:
            robot_states = np.asarray(demo[self.robot_states_key], dtype=np.float64)
            start, end = self.robot_states_quat_slice
            quats = robot_states[:, start:end]

            # Convert from wxyz to xyzw format (standard for scipy/transforms3d)
            if self.robot_states_quat_format == "wxyz":
                quats = quats[:, [1, 2, 3, 0]]  # w,x,y,z -> x,y,z,w

            return quats, f"{self.robot_states_key}[{start}:{end}]"

        return None, ""

    def load_dataset(self, dataset_path: str) -> Iterator[Dict]:
        if not HAS_H5PY:
            raise ImportError("h5py is required for Libero adapter (pip install h5py).")

        with h5py.File(dataset_path, "r") as h5:
            data_group = h5.get("data")
            if data_group is None:
                raise ValueError("Libero HDF5 missing /data group.")

            for demo_id in data_group.keys():
                demo = data_group[demo_id]
                obs = demo.get(self.obs_group)
                if obs is None:
                    continue

                # Get position
                pos_key = self._find_key(obs, self.pos_keys)
                if pos_key is None:
                    continue
                positions = np.asarray(obs[pos_key], dtype=np.float64)

                # Get quaternion (try obs first, then robot_states)
                quaternions, quat_source = self._extract_quaternion(demo, obs)
                if quaternions is None:
                    continue

                gripper = None
                gripper_qpos = None
                if self.robot_states_key in demo:
                    robot_states = np.asarray(demo[self.robot_states_key], dtype=np.float64)
                    gripper_qpos = robot_states[:, 0].astype(np.float32)
                    gripper = (gripper_qpos > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD).astype(np.float32)

                metadata = {
                    "demo_id": demo_id,
                    "pos_key": pos_key,
                    "quat_source": quat_source,
                    "source": "libero",
                    "num_frames": len(positions),
                }
                if "task" in demo.attrs:
                    metadata["task"] = demo.attrs["task"]
                if "language" in demo.attrs:
                    metadata["language_instruction"] = demo.attrs["language"]

                yield {
                    "positions": positions,
                    "quaternions": quaternions,
                    "gripper": gripper,
                    "gripper_qpos": gripper_qpos,
                    "metadata": metadata,
                }
