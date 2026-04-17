"""Libero dataset adapter (robomimic-style HDF5).

LIBERO stores end-effector data as:
- Position: obs/ee_pos (N, 3)
- Orientation: robot_states[:, 5:9] as quaternion (w, x, y, z)

Note: obs/ee_ori contains Euler angles, not quaternions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Tuple

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
