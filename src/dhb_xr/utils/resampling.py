"""Trajectory resampling and smoothing."""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import CubicSpline

from dhb_xr.core import geometry as geom


def resample_trajectory(
    positions: np.ndarray,
    quaternions: np.ndarray,
    num_points: int,
    smoothing: bool = False,
    window_length: int = 10,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample trajectory to num_points. Optionally apply Savitzky-Golay smoothing.
    positions (N, 3), quaternions (N, 4) wxyz.
    Returns (positions_resampled, quaternions_resampled) (num_points, 3), (num_points, 4).
    """
    from scipy.signal import savgol_filter

    n = positions.shape[0]
    time_orig = np.linspace(0, n - 1, n)
    time_new = np.linspace(0, n - 1, num_points)
    rvec = np.array([geom.quat_to_axis_angle(quaternions[i]) for i in range(n)])
    pos_resample = np.zeros((num_points, 3))
    rvec_resample = np.zeros((num_points, 3))
    for i in range(3):
        cs_p = CubicSpline(time_orig, positions[:, i])
        cs_r = CubicSpline(time_orig, rvec[:, i])
        pos_resample[:, i] = cs_p(time_new)
        rvec_resample[:, i] = cs_r(time_new)
        if smoothing:
            pos_resample[:, i] = savgol_filter(
                pos_resample[:, i], window_length=window_length, polyorder=polyorder
            )
            rvec_resample[:, i] = savgol_filter(
                rvec_resample[:, i], window_length=window_length, polyorder=polyorder
            )
    quat_resample = np.array([geom.axis_angle_to_quat(rvec_resample[i]) for i in range(num_points)])
    return pos_resample, quat_resample


def resample_and_smooth(
    pos_data: np.ndarray,
    quat_data: np.ndarray,
    length: int,
    smoothing: bool = False,
    window_length: int = 10,
    polyorder: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample to length points; return both original-length resampled and padded for encoder.
    Returns (pos_orig, quat_orig, rvec_orig, pos_padded, quat_padded, rvec_padded).
    """
    pos_orig, quat_orig = resample_trajectory(
        pos_data, quat_data, length, smoothing, window_length, polyorder
    )
    rvec_orig = np.array([geom.quat_to_axis_angle(quat_orig[i]) for i in range(length)])
    pos_padded = np.vstack([pos_orig, np.tile(pos_orig[-1], (3, 1))])
    quat_padded = np.vstack([quat_orig, np.tile(quat_orig[-1], (3, 1))])
    rvec_padded = np.vstack([rvec_orig, np.tile(rvec_orig[-1], (3, 1))])
    return pos_orig, quat_orig, rvec_orig, pos_padded, quat_padded, rvec_padded
