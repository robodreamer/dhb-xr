"""
DHB-DR decoder: reconstruct SE(3) trajectory from invariants.
Quaternion convention: wxyz (scalar-first).

Robustness features:
- Initial pose validation
- Divergence detection with optional warnings
- NaN/Inf handling for degenerate invariants
"""

import numpy as np
import warnings
from typing import Dict, Any, Optional
from dataclasses import dataclass

from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.core import geometry as geom

EPSILON = 1e-10


@dataclass
class DecodingDiagnostics:
    """Diagnostics from decoding."""
    initial_pose_valid: bool = True
    num_nan_positions: int = 0
    num_nan_quaternions: int = 0
    max_step_size: float = 0.0
    trajectory_length: float = 0.0


def _validate_initial_pose(initial_pose: Dict[str, np.ndarray]) -> bool:
    """Validate that initial pose is well-formed."""
    try:
        pos = np.asarray(initial_pose["position"]).reshape(3)
        quat = np.asarray(initial_pose["quaternion"]).reshape(4)

        # Check for NaN/Inf
        if not np.all(np.isfinite(pos)):
            return False
        if not np.all(np.isfinite(quat)):
            return False

        # Check quaternion is normalized
        quat_norm = np.linalg.norm(quat)
        if abs(quat_norm - 1.0) > 0.01:
            return False

        return True
    except (KeyError, ValueError):
        return False


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(quat)
    if norm > EPSILON:
        return quat / norm
    return np.array([1.0, 0.0, 0.0, 0.0])


def decode_dhb_dr(
    linear_motion_invariants: np.ndarray,
    angular_motion_invariants: np.ndarray,
    initial_pose: Dict[str, np.ndarray],
    method: EncodingMethod = EncodingMethod.POSITION,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    drop_padded: bool = True,
    robust_mode: bool = False,
    validate_initial_pose: bool = False,
    return_diagnostics: bool = False,
) -> Dict[str, Any]:
    """
    Reconstruct poses from DHB invariants.

    Parameters:
        linear_motion_invariants: (N, k) linear invariants
        angular_motion_invariants: (N, k) angular invariants
        initial_pose: {'position': (3,), 'quaternion': (4,) wxyz}
        method: 'pos' or 'vel'
        dhb_method: DHBMethod.ORIGINAL (3 inv) or DHBMethod.DOUBLE_REFLECTION (4 inv)
        drop_padded: if True, return positions/quaternions[2:] to match encoder padding
        robust_mode: Enable robustness features (NaN handling, normalization)
        validate_initial_pose: Check initial pose validity before decoding
        return_diagnostics: Include decoding diagnostics in output

    Returns:
        dict with 'positions' (N,3), 'quaternions' (N,4) wxyz,
        and optionally 'diagnostics' if return_diagnostics=True.
    """
    diagnostics = DecodingDiagnostics() if return_diagnostics else None

    # Validate initial pose if requested
    if validate_initial_pose or robust_mode:
        if not _validate_initial_pose(initial_pose):
            if diagnostics:
                diagnostics.initial_pose_valid = False
            if robust_mode:
                warnings.warn("Invalid initial pose detected. Using identity pose.")
                initial_pose = {
                    "position": np.zeros(3),
                    "quaternion": np.array([1.0, 0.0, 0.0, 0.0])
                }
            else:
                raise ValueError("Invalid initial pose: position or quaternion is malformed or contains NaN/Inf")
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

    position_mode = method == EncodingMethod.POSITION
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

    prev_pos = init_pos.copy()

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

        # Handle NaN in robust mode
        if robust_mode:
            if not np.all(np.isfinite(trajectory_position[i])):
                if diagnostics:
                    diagnostics.num_nan_positions += 1
                trajectory_position[i] = prev_pos.copy()
            else:
                # Track step size for diagnostics
                step_size = np.linalg.norm(trajectory_position[i] - prev_pos)
                if diagnostics:
                    diagnostics.max_step_size = max(diagnostics.max_step_size, step_size)
                    diagnostics.trajectory_length += step_size
                prev_pos = trajectory_position[i].copy()

        rvec = angular_frame[:3, :3] @ np.array([angular_magnitude[i], 0.0, 0.0])

        if dhb_method == DHBMethod.ORIGINAL:
            R_ang = geom.y_rot(angular_angle_y[i]) @ geom.x_rot(angular_angle_x[i])
        else:
            R_ang = geom.euler_to_rot(angular_angles[i])

        angular_rot_temp = angular_rot_temp @ geom.axis_angle_to_rot(rvec).T
        if i < num_samples - 1:
            quat = geom.rot_to_quat(angular_rot_temp)
            # Normalize quaternion in robust mode
            if robust_mode:
                quat = _normalize_quaternion(quat)
                if not np.all(np.isfinite(quat)):
                    if diagnostics:
                        diagnostics.num_nan_quaternions += 1
                    quat = trajectory_quaternions[i].copy()  # Use previous
            trajectory_quaternions[i + 1] = quat
        angular_frame[:3, :3] = angular_frame[:3, :3] @ R_ang

    if drop_padded:
        trajectory_position = trajectory_position[2:]
        trajectory_quaternions = trajectory_quaternions[2:]

    result = {
        "positions": trajectory_position,
        "quaternions": trajectory_quaternions,
    }

    if return_diagnostics and diagnostics is not None:
        result["diagnostics"] = diagnostics

    return result


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

    position_mode = method == EncodingMethod.POSITION
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
