"""Utilities: I/O, resampling."""

from dhb_xr.utils.resampling import resample_trajectory

__all__ = ["resample_trajectory", "load_trajectory", "save_trajectory"]

try:
    from dhb_xr.utils.io import load_trajectory, save_trajectory
except ImportError:
    load_trajectory = None
    save_trajectory = None
