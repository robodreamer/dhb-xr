"""
DHB-QR encoder: double-reflection frame transport with quaternion relative rotations.
Per-step invariant: [m, qw, qx, qy, qz] = 5 values; sign continuity enforced.
"""

import numpy as np
from typing import Dict, Optional, Any

from dhb_xr.encoder.dhb_dr import (
    encode_dhb_dr,
    _compute_frame_axis_x,
    _compute_frame_axis_y,
    _compute_frame_axis_z,
    _double_reflection_step,
    _compute_initial_frames,
    EPSILON,
)
from dhb_xr.core import geometry as geom


def encode_dhb_qr(
    positions: np.ndarray,
    quaternions: np.ndarray,
    init_pose: Optional[Dict[str, np.ndarray]] = None,
    method: str = "pos",
    use_default_initial_frames: bool = True,
    enforce_qw_nonnegative: bool = True,
) -> Dict[str, Any]:
    """
    Compute DHB-QR invariants: magnitude + unit quaternion (5 values per component).

    positions: (N, 3)
    quaternions: (N, 4) wxyz
    enforce_qw_nonnegative: if True, flip q to -q when qw < 0 for canonical form.
    Sign continuity: q_i chosen so q_i · q_{i-1} >= 0.

    Returns dict with linear_motion_invariants (N-2, 5), angular_motion_invariants (N-2, 5),
    linear_frame_initial, angular_frame_initial, initial_pose.
    """
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n = positions.shape[0]
    assert n > 2 and quaternions.shape[0] == n

    if use_default_initial_frames:
        positions = np.vstack((positions[0], positions[0], positions))
        quaternions = np.vstack((quaternions[0], quaternions[0], quaternions))
    positions = np.vstack((positions, positions[-1], positions[-1], positions[-1]))
    quaternions = np.vstack((quaternions, quaternions[-1], quaternions[-1], quaternions[-1]))

    num_samples = positions.shape[0]
    initial_pose = init_pose or {
        "position": positions[0].copy(),
        "quaternion": quaternions[0].copy(),
    }
    initial_pose["position"] = np.asarray(initial_pose["position"]).reshape(3)
    initial_pose["quaternion"] = np.asarray(initial_pose["quaternion"]).reshape(4)

    position_diff = np.diff(positions, axis=0)
    rotation_diff = np.zeros((num_samples - 1, 3))
    for i in range(1, num_samples):
        R_prev = geom.quat_to_rot(quaternions[i - 1]).T
        R_curr = geom.quat_to_rot(quaternions[i]).T
        R_rel = R_curr @ R_prev.T
        rotation_diff[i - 1] = geom.rot_to_axis_angle(R_rel)

    num_steps = position_diff.shape[0]
    frames = _compute_initial_frames(
        position_diff, rotation_diff, initial_pose, method, use_default_initial_frames
    )

    k = 5
    linear_inv = np.zeros((num_steps - 2, k))
    angular_inv = np.zeros((num_steps - 2, k))

    lx = frames["linear_frame_x"].copy()
    lx2 = frames["linear_frame_x2"].copy()
    ly = frames["linear_frame_y"].copy()
    ax = frames["angular_frame_x"].copy()
    ax2 = frames["angular_frame_x2"].copy()
    ay = frames["angular_frame_y"].copy()

    q_prev_lin = None
    q_prev_ang = None

    for i in range(num_steps - 2):
        lx3 = _compute_frame_axis_x(position_diff[i + 2], lx2)
        ax3 = _compute_frame_axis_x(rotation_diff[i + 2], ax2)

        ly2, linear_R_rel, _ = _double_reflection_step(
            lx, ly, lx2, position_diff[i], position_diff[i + 1]
        )
        ay2, angular_R_rel, _ = _double_reflection_step(
            ax, ay, ax2, rotation_diff[i], rotation_diff[i + 1]
        )

        linear_inv[i, 0] = np.dot(lx, position_diff[i])
        angular_inv[i, 0] = np.dot(ax, rotation_diff[i])

        q_lin = geom.rot_to_quat(linear_R_rel)
        q_ang = geom.rot_to_quat(angular_R_rel)

        if q_prev_lin is not None and np.dot(q_lin, q_prev_lin) < 0:
            q_lin = -q_lin
        if q_prev_ang is not None and np.dot(q_ang, q_prev_ang) < 0:
            q_ang = -q_ang
        if enforce_qw_nonnegative:
            if q_lin[0] < 0:
                q_lin = -q_lin
            if q_ang[0] < 0:
                q_ang = -q_ang
        q_prev_lin = q_lin.copy()
        q_prev_ang = q_ang.copy()

        linear_inv[i, 1:5] = q_lin
        angular_inv[i, 1:5] = q_ang

        lx, lx2, ly = lx2, lx3, ly2
        ax, ax2, ay = ax2, ax3, ay2

    return {
        "linear_motion_invariants": linear_inv,
        "angular_motion_invariants": angular_inv,
        "linear_frame_initial": frames["linear_frame_initial"],
        "angular_frame_initial": frames["angular_frame_initial"],
        "initial_pose": initial_pose,
    }
