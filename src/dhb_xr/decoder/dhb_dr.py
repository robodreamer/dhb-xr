"""
DHB-DR decoder: reconstruct SE(3) trajectory from invariants.
Quaternion convention: wxyz (scalar-first).
"""

import numpy as np
from typing import Dict, Any

from dhb_xr.core.types import DHBMethod
from dhb_xr.core import geometry as geom


def decode_dhb_dr(
    linear_motion_invariants: np.ndarray,
    angular_motion_invariants: np.ndarray,
    initial_pose: Dict[str, np.ndarray],
    method: str = "pos",
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    drop_padded: bool = True,
) -> Dict[str, Any]:
    """
    Reconstruct poses from DHB invariants.

    initial_pose: {'position': (3,), 'quaternion': (4,) wxyz}
    method: 'pos' or 'vel'
    dhb_method: DHBMethod.ORIGINAL (3 inv) or DHBMethod.DOUBLE_REFLECTION (4 inv)
    drop_padded: if True, return positions/quaternions[2:] to match encoder padding convention.

    Returns dict with 'positions' (N,3), 'quaternions' (N,4) wxyz.
    """
    linear_inv = np.asarray(linear_motion_invariants, dtype=np.float64)
    angular_inv = np.asarray(angular_motion_invariants, dtype=np.float64)
    num_samples = linear_inv.shape[0]
    assert angular_inv.shape[0] == num_samples

    if dhb_method == DHBMethod.ORIGINAL:
        linear_magnitude = linear_inv[:, 0]
        linear_angle_y = linear_inv[:, 1]
        linear_angle_x = linear_inv[:, 2]
        angular_magnitude = angular_inv[:, 0]
        angular_angle_y = angular_inv[:, 1]
        angular_angle_x = angular_inv[:, 2]
    else:
        linear_magnitude = linear_inv[:, 0]
        linear_angles = linear_inv[:, 1:4]
        angular_magnitude = angular_inv[:, 0]
        angular_angles = angular_inv[:, 1:4]

    position_mode = method == "pos"
    init_pos = np.asarray(initial_pose["position"]).reshape(3)
    init_quat = np.asarray(initial_pose["quaternion"]).reshape(4)

    trajectory_position = np.zeros((num_samples, 3))
    trajectory_quaternions = np.zeros((num_samples, 4))
    trajectory_quaternions[0] = init_quat

    linear_frame = np.eye(4)
    linear_frame[:3, 3] = init_pos
    linear_frame[3, 3] = float(position_mode)

    angular_frame = np.eye(4)
    angular_rot_temp = geom.quat_to_rot(init_quat)

    for i in range(num_samples):
        if position_mode:
            trajectory_position[i] = linear_frame[:3, 3]

        if dhb_method == DHBMethod.ORIGINAL:
            R_lin = geom.y_rot(linear_angle_y[i]) @ geom.x_rot(linear_angle_x[i])
        else:
            R_lin = geom.euler_to_rot(linear_angles[i])

        trans = np.array([linear_magnitude[i], 0.0, 0.0])
        T = np.eye(4)
        T[:3, :3] = R_lin
        T[:3, 3] = trans
        T[3, 3] = float(position_mode)
        linear_frame = linear_frame @ T

        if not position_mode:
            trajectory_position[i] = linear_frame[:3, 3]

        rvec = angular_frame[:3, :3] @ np.array([angular_magnitude[i], 0.0, 0.0])

        if dhb_method == DHBMethod.ORIGINAL:
            R_ang = geom.y_rot(angular_angle_y[i]) @ geom.x_rot(angular_angle_x[i])
        else:
            R_ang = geom.euler_to_rot(angular_angles[i])

        angular_rot_temp = angular_rot_temp @ geom.axis_angle_to_rot(rvec).T
        if i < num_samples - 1:
            trajectory_quaternions[i + 1] = geom.rot_to_quat(angular_rot_temp)
        angular_frame[:3, :3] = angular_frame[:3, :3] @ R_ang

    if drop_padded:
        trajectory_position = trajectory_position[2:]
        trajectory_quaternions = trajectory_quaternions[2:]

    return {
        "positions": trajectory_position,
        "quaternions": trajectory_quaternions,
    }


def reconstruct_trajectory_single_step(
    linear_invariants: np.ndarray,
    angular_invariants: np.ndarray,
    current_linear_frame: np.ndarray,
    current_angular_frame: np.ndarray,
    method: str,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
) -> tuple:
    """Single-step reconstruction for use in optimization. Returns (new_linear_frame, new_angular_frame, new_position, new_rotation_vec)."""
    if dhb_method == DHBMethod.ORIGINAL:
        linear_magnitude = linear_invariants[0]
        linear_angle_y = linear_invariants[1]
        linear_angle_x = linear_invariants[2]
        angular_magnitude = angular_invariants[0]
        angular_angle_y = angular_invariants[1]
        angular_angle_x = angular_invariants[2]
    else:
        linear_magnitude = float(linear_invariants[0])
        linear_angles = linear_invariants[1:4]
        angular_magnitude = float(angular_invariants[0])
        angular_angles = angular_invariants[1:4]

    position_mode = method == "pos"
    if position_mode:
        new_position = current_linear_frame[:3, 3].copy()

    if dhb_method == DHBMethod.ORIGINAL:
        R_lin = geom.y_rot(linear_angle_y) @ geom.x_rot(linear_angle_x)
    else:
        R_lin = geom.euler_to_rot(linear_angles)

    T = np.eye(4)
    T[:3, :3] = R_lin
    T[:3, 3] = np.array([linear_magnitude, 0.0, 0.0])
    T[3, 3] = float(position_mode)
    new_linear_frame = current_linear_frame @ T

    if not position_mode:
        new_position = new_linear_frame[:3, 3].copy()

    new_rotation = current_angular_frame[:3, :3] @ np.array(
        [angular_magnitude, 0.0, 0.0]
    )

    if dhb_method == DHBMethod.ORIGINAL:
        R_ang = geom.y_rot(angular_angle_y) @ geom.x_rot(angular_angle_x)
    else:
        R_ang = geom.euler_to_rot(angular_angles)

    new_angular_frame = np.eye(4)
    new_angular_frame[:3, :3] = current_angular_frame[:3, :3] @ R_ang

    return new_linear_frame, new_angular_frame, new_position, new_rotation
