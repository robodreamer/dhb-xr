"""
Load UMI-XDOF trajectory data from numpy files (no xdof_sdk dependency).

Episodes are directories containing:
  - action-{arm}-hand_in_quest_world_frame.npy  (N, 4, 4) SE(3) poses
  - metadata.json
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from dhb_xr.core import geometry as geom


def se3_poses_to_positions_quaternions(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert (N, 4, 4) SE(3) pose matrices to positions (N, 3) and quaternions (N, 4) wxyz.

    Args:
        poses: Array of shape (N, 4, 4) homogeneous transformation matrices.

    Returns:
        positions: (N, 3) XYZ positions.
        quaternions: (N, 4) quaternions in wxyz (scalar-first) convention.
    """
    poses = np.asarray(poses)
    if poses.ndim == 2:
        poses = poses[np.newaxis, ...]
    positions = poses[:, :3, 3].copy()
    quaternions = np.array([geom.rot_to_quat(poses[i, :3, :3]) for i in range(len(poses))])
    return positions, quaternions


def load_xdof_episode(
    episode_path: Path,
    arm: str = "left",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load one UMI-XDOF episode and return (positions, quaternions) for the given arm.

    Args:
        episode_path: Path to episode directory (contains action-*-hand_in_quest_world_frame.npy).
        arm: "left" or "right".

    Returns:
        (positions, quaternions) with shapes (N, 3) and (N, 4) wxyz, or None if load fails.
    """
    episode_path = Path(episode_path)
    fname = f"action-{arm.lower()}-hand_in_quest_world_frame.npy"
    npy_path = episode_path / fname
    if not npy_path.exists():
        return None
    try:
        poses = np.load(npy_path)
    except Exception:
        return None
    if poses.ndim != 3 or poses.shape[1] != 4 or poses.shape[2] != 4:
        return None
    return se3_poses_to_positions_quaternions(poses)


def list_xdof_episodes(data_root: Path) -> List[Tuple[str, Path]]:
    """
    List all (task, episode_path) pairs under data_root.

    Expects structure: data_root / task_name / episode_xxx / metadata.json (or .npy files).

    Returns:
        List of (task_name, episode_dir_path).
    """
    data_root = Path(data_root)
    if not data_root.is_dir():
        return []
    out = []
    for task_dir in sorted(data_root.iterdir()):
        if not task_dir.is_dir():
            continue
        task_name = task_dir.name
        for ep_dir in sorted(task_dir.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode_"):
                continue
            # Consider valid if it has at least one hand pose file
            if (ep_dir / "action-left-hand_in_quest_world_frame.npy").exists() or (
                ep_dir / "action-right-hand_in_quest_world_frame.npy"
            ).exists():
                out.append((task_name, ep_dir))
    return out
