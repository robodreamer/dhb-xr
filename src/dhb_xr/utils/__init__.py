"""Utilities: I/O, resampling, XDOF loader."""

from dhb_xr.utils.resampling import resample_trajectory

__all__ = [
    "resample_trajectory",
    "load_trajectory",
    "save_trajectory",
    "load_xdof_episode",
    "list_xdof_episodes",
    "se3_poses_to_positions_quaternions",
]

try:
    from dhb_xr.utils.io import load_trajectory, save_trajectory
except ImportError:
    load_trajectory = None
    save_trajectory = None

from dhb_xr.utils.xdof_loader import (
    load_xdof_episode,
    list_xdof_episodes,
    se3_poses_to_positions_quaternions,
)
