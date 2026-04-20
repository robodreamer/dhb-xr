from __future__ import annotations

"""Shared receding-horizon chunk utilities for DHB trajectory rollout.

This keeps the loop mechanics closer to the vendor StarVLA chunk pattern
(`predict -> post-process -> execute prefix -> replan`) while preserving the
custom DHB execution path:

    predict_trajectory() -> waypoint chunk -> embodik / sim

The main goal is to centralize chunk stabilization, overlap blending, and
learned gripper decoding so Stage 1 and live rollout do not reimplement the
same logic independently.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from dhb_xr.integration.vla.libero import decode_model_gripper_cmds


def clip_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if max_norm <= 0.0 or norm <= max_norm or norm <= 1e-9:
        return vec
    return vec * (max_norm / norm)


def stabilize_chunk_positions(
    chunk_pos: np.ndarray,
    *,
    anchor_pos: np.ndarray,
    max_first_offset_m: float,
    max_step_m: float,
    smooth_window: int,
) -> tuple[np.ndarray, dict[str, float]]:
    stabilized = np.asarray(chunk_pos, dtype=np.float32).copy()
    debug = {
        "raw_first_offset_mm": 0.0,
        "stabilized_first_offset_mm": 0.0,
        "raw_mean_step_mm": 0.0,
        "stabilized_mean_step_mm": 0.0,
    }
    if len(stabilized) == 0:
        return stabilized, debug

    anchor = np.asarray(anchor_pos, dtype=np.float32).reshape(3)
    debug["raw_first_offset_mm"] = float(np.linalg.norm(stabilized[0] - anchor) * 1000.0)
    if len(stabilized) > 1:
        debug["raw_mean_step_mm"] = float(
            np.linalg.norm(np.diff(stabilized, axis=0), axis=1).mean() * 1000.0
        )

    if max_first_offset_m > 0.0:
        stabilized[0] = anchor + clip_vector_norm(stabilized[0] - anchor, max_first_offset_m)

    if max_step_m > 0.0:
        for i in range(1, len(stabilized)):
            step = stabilized[i] - stabilized[i - 1]
            stabilized[i] = stabilized[i - 1] + clip_vector_norm(step, max_step_m)

    if smooth_window > 1 and len(stabilized) > 2:
        smoothed = stabilized.copy()
        radius = smooth_window // 2
        for i in range(len(stabilized)):
            lo = max(0, i - radius)
            hi = min(len(stabilized), i + radius + 1)
            smoothed[i] = stabilized[lo:hi].mean(axis=0)
        stabilized = smoothed
        stabilized[0] = anchor + clip_vector_norm(stabilized[0] - anchor, max_first_offset_m)
        if max_step_m > 0.0:
            for i in range(1, len(stabilized)):
                step = stabilized[i] - stabilized[i - 1]
                stabilized[i] = stabilized[i - 1] + clip_vector_norm(step, max_step_m)

    debug["stabilized_first_offset_mm"] = float(np.linalg.norm(stabilized[0] - anchor) * 1000.0)
    if len(stabilized) > 1:
        debug["stabilized_mean_step_mm"] = float(
            np.linalg.norm(np.diff(stabilized, axis=0), axis=1).mean() * 1000.0
        )
    return stabilized.astype(np.float32), debug


def blend_overlap_positions(
    current_chunk: np.ndarray,
    *,
    current_start: int,
    prev_chunk: np.ndarray | None,
    prev_start: int | None,
    blend_alpha: float,
    blend_mode: str = "constant",
    commit_prefix_steps: int = 0,
) -> tuple[np.ndarray, int, int]:
    if prev_chunk is None or prev_start is None:
        return current_chunk, 0, 0

    prev_end = prev_start + len(prev_chunk)
    overlap = min(max(prev_end - current_start, 0), len(current_chunk))
    if overlap <= 0:
        return current_chunk, 0, 0

    alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    commit = int(max(0, min(commit_prefix_steps, overlap)))
    if alpha <= 0.0 and commit <= 0:
        return current_chunk, 0, 0

    blended = np.asarray(current_chunk, dtype=np.float32).copy()
    prev_tail = np.asarray(prev_chunk[-overlap:], dtype=np.float32)
    if commit > 0:
        blended[:commit] = prev_tail[:commit]

    if overlap > commit and alpha > 0.0:
        if blend_mode == "ramp":
            alpha_lo = min(alpha, 0.2)
            alpha_sched = np.linspace(alpha_lo, alpha, overlap - commit, dtype=np.float32)
            for idx, alpha_i in enumerate(alpha_sched, start=commit):
                blended[idx] = (1.0 - alpha_i) * prev_tail[idx] + alpha_i * blended[idx]
        else:
            blended[commit:overlap] = (
                (1.0 - alpha) * prev_tail[commit:overlap]
                + alpha * blended[commit:overlap]
            )
    return blended, int(overlap), int(commit)


def compute_motion_metrics(positions: np.ndarray) -> dict[str, float | int | None]:
    pos = np.asarray(positions, dtype=np.float32)
    if len(pos) == 0:
        return {
            "num_points": 0,
            "path_length_m": 0.0,
            "net_displacement_m": 0.0,
            "path_efficiency": None,
            "mean_step_mm": 0.0,
            "std_step_mm": 0.0,
            "max_step_mm": 0.0,
            "hold_fraction": 0.0,
            "mean_accel_mm": 0.0,
            "max_accel_mm": 0.0,
            "mean_jerk_mm": 0.0,
            "max_jerk_mm": 0.0,
            "direction_flip_fraction": 0.0,
        }

    if len(pos) == 1:
        return {
            "num_points": 1,
            "path_length_m": 0.0,
            "net_displacement_m": 0.0,
            "path_efficiency": None,
            "mean_step_mm": 0.0,
            "std_step_mm": 0.0,
            "max_step_mm": 0.0,
            "hold_fraction": 1.0,
            "mean_accel_mm": 0.0,
            "max_accel_mm": 0.0,
            "mean_jerk_mm": 0.0,
            "max_jerk_mm": 0.0,
            "direction_flip_fraction": 0.0,
        }

    steps = np.diff(pos, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    path_length_m = float(step_norms.sum())
    net_displacement_m = float(np.linalg.norm(pos[-1] - pos[0]))
    hold_fraction = float(np.mean(step_norms < 1e-4))

    accel = np.diff(steps, axis=0)
    accel_norms = np.linalg.norm(accel, axis=1) if len(accel) else np.zeros((0,), dtype=np.float32)
    jerk = np.diff(accel, axis=0)
    jerk_norms = np.linalg.norm(jerk, axis=1) if len(jerk) else np.zeros((0,), dtype=np.float32)

    direction_flip_fraction = 0.0
    if len(steps) > 1:
        prev = steps[:-1]
        nxt = steps[1:]
        denom = np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1)
        valid = denom > 1e-8
        if np.any(valid):
            cos = np.sum(prev[valid] * nxt[valid], axis=1) / denom[valid]
            direction_flip_fraction = float(np.mean(cos < 0.0))

    return {
        "num_points": int(len(pos)),
        "path_length_m": path_length_m,
        "net_displacement_m": net_displacement_m,
        "path_efficiency": None if path_length_m <= 1e-8 else float(net_displacement_m / path_length_m),
        "mean_step_mm": float(step_norms.mean() * 1000.0),
        "std_step_mm": float(step_norms.std() * 1000.0),
        "max_step_mm": float(step_norms.max() * 1000.0),
        "hold_fraction": hold_fraction,
        "mean_accel_mm": 0.0 if len(accel_norms) == 0 else float(accel_norms.mean() * 1000.0),
        "max_accel_mm": 0.0 if len(accel_norms) == 0 else float(accel_norms.max() * 1000.0),
        "mean_jerk_mm": 0.0 if len(jerk_norms) == 0 else float(jerk_norms.mean() * 1000.0),
        "max_jerk_mm": 0.0 if len(jerk_norms) == 0 else float(jerk_norms.max() * 1000.0),
        "direction_flip_fraction": direction_flip_fraction,
    }


@dataclass
class ChunkRolloutConfig:
    max_first_offset_m: float = 0.0
    max_step_m: float = 0.0
    smooth_window: int = 0
    blend_overlap_alpha: float = 0.0
    blend_mode: str = "constant"
    commit_prefix_steps: int = 0
    gripper_source: str = "demo"


class TrajectoryChunkRunner:
    """Stateful helper for vendor-style chunk-boundary trajectory scheduling.

    The runner does not own environment stepping or IK. It only caches the
    previous predicted chunk so consecutive replans can be post-processed
    consistently across offline and live evaluation.
    """

    def __init__(self, config: ChunkRolloutConfig) -> None:
        self.config = config
        self._prev_chunk_pos: np.ndarray | None = None
        self._prev_chunk_start: int | None = None

    def reset(self) -> None:
        self._prev_chunk_pos = None
        self._prev_chunk_start = None

    def prepare_chunk(
        self,
        result: dict[str, Any],
        *,
        anchor_pos: np.ndarray,
        current_start: int,
        n_use: int,
        initial_gripper_cmd: float,
        demo_gripper_chunk: np.ndarray | None = None,
        config_override: ChunkRolloutConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config_override or self.config
        chunk_pos = np.asarray(result["positions"][0], dtype=np.float32)
        chunk_quat = np.asarray(result["quaternions"][0], dtype=np.float32)  # wxyz

        chunk_pos_stable, stability = stabilize_chunk_positions(
            chunk_pos,
            anchor_pos=anchor_pos,
            max_first_offset_m=max(0.0, cfg.max_first_offset_m),
            max_step_m=max(0.0, cfg.max_step_m),
            smooth_window=max(0, cfg.smooth_window),
        )
        chunk_pos_stable, overlap_steps, commit_steps = blend_overlap_positions(
            chunk_pos_stable,
            current_start=current_start,
            prev_chunk=self._prev_chunk_pos,
            prev_start=self._prev_chunk_start,
            blend_alpha=cfg.blend_overlap_alpha,
            blend_mode=cfg.blend_mode,
            commit_prefix_steps=cfg.commit_prefix_steps,
        )

        if cfg.gripper_source == "demo":
            if demo_gripper_chunk is None:
                raise ValueError("demo_gripper_chunk is required when gripper_source='demo'")
            gripper_cmds = np.asarray(demo_gripper_chunk[:n_use], dtype=np.float32)
            gripper_debug = {"source": "demo"}
        else:
            gripper_cmds, gripper_debug = decode_model_gripper_cmds(
                result,
                n_use=n_use,
                initial_gripper_cmd=float(initial_gripper_cmd),
            )

        self._prev_chunk_pos = chunk_pos_stable.copy()
        self._prev_chunk_start = int(current_start)

        return {
            "positions": chunk_pos_stable[:n_use],
            "positions_full": chunk_pos_stable,
            "quaternions": chunk_quat[:n_use],
            "gripper_cmds": gripper_cmds,
            "debug": {
                "gripper_source_used": gripper_debug.get("source"),
                "close_probs": gripper_debug.get("close_probs"),
                "open_probs": gripper_debug.get("open_probs"),
                "overlap_steps": int(overlap_steps),
                "commit_prefix_steps": int(commit_steps),
                "blend_mode": cfg.blend_mode,
                **stability,
            },
            "first_pred_pos": chunk_pos[0].tolist() if len(chunk_pos) else None,
            "first_stabilized_pos": chunk_pos_stable[0].tolist() if len(chunk_pos_stable) else None,
        }
