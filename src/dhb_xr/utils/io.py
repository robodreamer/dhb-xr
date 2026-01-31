"""I/O for trajectories (load/save)."""

import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def load_trajectory(path: str) -> Dict[str, np.ndarray]:
    """Load trajectory from .npz or .txt. Returns dict with 'positions' (N,3), 'quaternions' (N,4) wxyz."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        return {
            "positions": data["positions"],
            "quaternions": data["quaternions"],
        }
    if path.suffix in (".txt", ".csv"):
        arr = np.loadtxt(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Assume columns: x, y, z, qw, qx, qy, qz or timestamp, x, y, z, qx, qy, qz, qw
        if arr.shape[1] >= 7:
            positions = arr[:, :3]
            quat = arr[:, 3:7]
            if quat.shape[1] == 4:
                # If scalar-last (xyzw), convert to wxyz
                if np.all(np.abs(quat[:, 0]) < 1.1):
                    quat = np.column_stack([quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]])
                quaternions = quat
            else:
                quaternions = np.tile(np.array([1.0, 0, 0, 0]), (len(positions), 1))
        else:
            positions = arr[:, :3]
            quaternions = np.tile(np.array([1.0, 0, 0, 0]), (len(positions), 1))
        return {"positions": positions, "quaternions": quaternions}
    raise ValueError(f"Unsupported format: {path.suffix}")


def save_trajectory(
    path: str,
    positions: np.ndarray,
    quaternions: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> None:
    """Save trajectory to .npz or .txt."""
    path = Path(path)
    if path.suffix == ".npz":
        out = {"positions": np.asarray(positions), "quaternions": np.asarray(quaternions)}
        if timestamps is not None:
            out["timestamps"] = np.asarray(timestamps)
        np.savez(path, **out)
        return
    if path.suffix in (".txt", ".csv"):
        arr = np.hstack([positions, quaternions])
        if timestamps is not None:
            arr = np.column_stack([timestamps, arr])
        np.savetxt(path, arr)
        return
    raise ValueError(f"Unsupported format: {path.suffix}")
