"""
DHB-QR decoder: reconstruct trajectory from quaternion-based invariants.
Per-step: magnitude + unit quaternion (5 values per component).
"""

import numpy as np
from typing import Dict, Any

from dhb_xr.core.types import EncodingMethod
from dhb_xr.core import geometry as geom


def decode_dhb_qr(
    linear_motion_invariants: np.ndarray,
    angular_motion_invariants: np.ndarray,
    initial_pose: Dict[str, np.ndarray],
    method: str = "pos",
    drop_padded: bool = True,
) -> Dict[str, Any]:
    """
    Reconstruct poses from DHB-QR invariants.

    initial_pose: {'position': (3,), 'quaternion': (4,) wxyz}
    drop_padded: if True, return positions/quaternions[2:].

    Returns dict with 'positions' (N,3), 'quaternions' (N,4) wxyz.
    """
    linear_inv = np.asarray(linear_motion_invariants, dtype=np.float64)
    angular_inv = np.asarray(angular_motion_invariants, dtype=np.float64)
    num_samples = linear_inv.shape[0]
    assert angular_inv.shape[0] == num_samples and linear_inv.shape[1] == 5

    linear_magnitude = linear_inv[:, 0]
    linear_quat = linear_inv[:, 1:5]  # (N, 4) wxyz
    angular_magnitude = angular_inv[:, 0]
    angular_quat = angular_inv[:, 1:5]

    # Normalize quaternions
    for i in range(num_samples):
        n = np.linalg.norm(linear_quat[i])
        if n > 1e-12:
            linear_quat[i] /= n
        n = np.linalg.norm(angular_quat[i])
        if n > 1e-12:
            angular_quat[i] /= n

    position_mode = method == EncodingMethod.POSITION
    init_pos = np.asarray(initial_pose["position"]).reshape(3)
    init_quat = np.asarray(initial_pose["quaternion"]).reshape(4)

    trajectory_position = np.zeros((num_samples, 3))
    trajectory_quaternions = np.zeros((num_samples, 4))
    trajectory_quaternions[0] = init_quat

    linear_frame = np.eye(4)
    linear_frame[:3, 3] = init_pos
    linear_frame[3, 3] = float(position_mode)
    angular_rot_temp = geom.quat_to_rot(init_quat)

    for i in range(num_samples):
        if position_mode:
            trajectory_position[i] = linear_frame[:3, 3]

        R_lin = geom.quat_to_rot(linear_quat[i])
        trans = R_lin @ np.array([linear_magnitude[i], 0.0, 0.0])
        T = np.eye(4)
        T[:3, :3] = R_lin
        T[:3, 3] = trans
        T[3, 3] = float(position_mode)
        linear_frame = linear_frame @ T

        if not position_mode:
            trajectory_position[i] = linear_frame[:3, 3]

        rvec = geom.quat_to_rot(angular_quat[i]) @ np.array(
            [angular_magnitude[i], 0.0, 0.0]
        )
        angular_rot_temp = angular_rot_temp @ geom.axis_angle_to_rot(rvec).T
        if i < num_samples - 1:
            trajectory_quaternions[i + 1] = geom.rot_to_quat(angular_rot_temp)

    if drop_padded:
        trajectory_position = trajectory_position[2:]
        trajectory_quaternions = trajectory_quaternions[2:]

    return {
        "positions": trajectory_position,
        "quaternions": trajectory_quaternions,
    }
