"""
DHB-DR encoder: double-reflection (RMF) frame transport with Euler XYZ relative rotations.
Quaternion convention: wxyz (scalar-first).
"""

import numpy as np
from typing import Dict, Optional, Any

from dhb_xr.core.types import DHBMethod
from dhb_xr.core import geometry as geom

EPSILON = 1e-10


def _compute_frame_axis_x(vector_u: np.ndarray, default_x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vector_u)
    if n > EPSILON:
        return vector_u / n
    return default_x.copy()


def _compute_frame_axis_y(
    frame_x1: np.ndarray, frame_x2: np.ndarray, default_y: np.ndarray
) -> np.ndarray:
    y = np.cross(frame_x1, frame_x2)
    n = np.linalg.norm(y)
    if n > EPSILON:
        return y / n
    return default_y.copy()


def _compute_frame_axis_z(
    frame_x: np.ndarray, frame_y: np.ndarray, default_z: np.ndarray
) -> np.ndarray:
    z = np.cross(frame_x, frame_y)
    n = np.linalg.norm(z)
    if n > EPSILON:
        return z / n
    return default_z.copy()


def _householder(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector).reshape(-1, 1)
    n = np.linalg.norm(v)
    if n > EPSILON:
        u = v / n
        return np.eye(3) - 2 * (u @ u.T)
    return np.eye(3)


def _double_reflection_step(
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
    diff_old: np.ndarray,
    diff_new: np.ndarray,
    tol: float = EPSILON,
) -> tuple:
    v1 = diff_new - diff_old
    R1 = _householder(v1)
    x_ref = R1 @ x_old
    v2 = x_new - x_ref
    R2 = _householder(v2)
    if np.linalg.norm(v1) > tol and np.linalg.norm(v2) > tol:
        R = R2 @ R1
    else:
        R = np.eye(3)
    y_new = R @ y_old
    F_old = np.column_stack((x_old, y_old, np.cross(x_old, y_old)))
    F_new = np.column_stack((x_new, y_new, np.cross(x_new, y_new)))
    R_rel = F_old.T @ F_new
    return y_new, R_rel


def _compute_invariants_original(
    vector_u: np.ndarray,
    frame_x: np.ndarray,
    frame_x2: np.ndarray,
    frame_y: np.ndarray,
    frame_y2: np.ndarray,
) -> np.ndarray:
    m = np.dot(frame_x, vector_u)
    a1 = np.arctan2(
        np.dot(np.cross(frame_x, frame_x2), frame_y), np.dot(frame_x, frame_x2)
    )
    a2 = np.arctan2(
        np.dot(np.cross(frame_y, frame_y2), frame_x2), np.dot(frame_y, frame_y2)
    )
    return np.array([m, a1, a2])


def _compute_initial_frames(
    position_diff: np.ndarray,
    rotation_diff: np.ndarray,
    initial_pose: Dict[str, np.ndarray],
    method: str,
    use_default_initial_frames: bool,
) -> Dict[str, Any]:
    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    linear_frame_initial = np.eye(4)
    angular_frame_initial = np.eye(4)

    if use_default_initial_frames:
        lx, lx2, ly = x_axis, x_axis, y_axis
        ax, ax2, ay = x_axis, x_axis, y_axis
    else:
        lx = _compute_frame_axis_x(position_diff[0], x_axis)
        lx2 = _compute_frame_axis_x(position_diff[1], lx)
        if np.allclose(lx, x_axis) and np.allclose(lx, lx2):
            default_y = y_axis
        else:
            default_y = np.array([lx[1] - lx[2], lx[2] - lx[0], lx[0] - lx[1]])
            default_y = default_y / (np.linalg.norm(default_y) + EPSILON)
        ly = _compute_frame_axis_y(lx, lx2, default_y)
        lz = _compute_frame_axis_z(lx, ly, z_axis)
        linear_frame_initial[:3, :3] = np.vstack((lx, ly, lz)).T
        linear_frame_initial[:3, 3] = (
            initial_pose["position"] if method == "pos" else position_diff[0]
        )

        ax = _compute_frame_axis_x(rotation_diff[0], x_axis)
        ax2 = _compute_frame_axis_x(rotation_diff[1], ax)
        ay = _compute_frame_axis_y(ax, ax2, y_axis)
        az = _compute_frame_axis_z(ax, ay, z_axis)
        angular_frame_initial[:3, :3] = np.vstack((ax, ay, az)).T

    return {
        "linear_frame_initial": linear_frame_initial,
        "angular_frame_initial": angular_frame_initial,
        "linear_frame_x": lx if use_default_initial_frames else lx,
        "linear_frame_x2": lx2,
        "linear_frame_y": ly,
        "angular_frame_x": ax,
        "angular_frame_x2": ax2,
        "angular_frame_y": ay,
    }


def encode_dhb_dr(
    positions: np.ndarray,
    quaternions: np.ndarray,
    init_pose: Optional[Dict[str, np.ndarray]] = None,
    method: str = "pos",
    use_default_initial_frames: bool = True,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
) -> Dict[str, Any]:
    """
    Compute DHB invariants (original or double-reflection).

    positions: (N, 3)
    quaternions: (N, 4) wxyz
    init_pose: optional {'position': (3,), 'quaternion': (4,) wxyz}
    method: 'pos' or 'vel'
    use_default_initial_frames: if True, pad and use identity-like frames
    dhb_method: DHBMethod.ORIGINAL (3 inv) or DHBMethod.DOUBLE_REFLECTION (4 inv)

    Returns dict with linear_motion_invariants, angular_motion_invariants,
    linear_frame_initial, angular_frame_initial, initial_pose.
    """
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n = positions.shape[0]
    assert n > 2 and quaternions.shape[0] == n, "Need >2 samples and matching lengths"

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
    initial_pose["quaternion"] = np.asarray(initial_pose["quaternion"]).reshape(4)
    initial_pose["position"] = np.asarray(initial_pose["position"]).reshape(3)

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

    k = 4 if dhb_method == DHBMethod.DOUBLE_REFLECTION else 3
    linear_inv = np.zeros((num_steps - 2, k))
    angular_inv = np.zeros((num_steps - 2, k))

    lx = frames["linear_frame_x"].copy()
    lx2 = frames["linear_frame_x2"].copy()
    ly = frames["linear_frame_y"].copy()
    ax = frames["angular_frame_x"].copy()
    ax2 = frames["angular_frame_x2"].copy()
    ay = frames["angular_frame_y"].copy()

    for i in range(num_steps - 2):
        lx3 = _compute_frame_axis_x(position_diff[i + 2], lx2)
        ax3 = _compute_frame_axis_x(rotation_diff[i + 2], ax2)

        if dhb_method == DHBMethod.DOUBLE_REFLECTION:
            ly2, linear_R_rel = _double_reflection_step(
                lx, ly, lx2, position_diff[i], position_diff[i + 1]
            )
            ay2, angular_R_rel = _double_reflection_step(
                ax, ay, ax2, rotation_diff[i], rotation_diff[i + 1]
            )
            linear_inv[i, 0] = np.dot(lx, position_diff[i])
            angular_inv[i, 0] = np.dot(ax, rotation_diff[i])
            linear_inv[i, 1:4] = geom.rot_to_euler(linear_R_rel)
            angular_inv[i, 1:4] = geom.rot_to_euler(angular_R_rel)
        else:
            ly2 = _compute_frame_axis_y(lx2, lx3, ly)
            ay2 = _compute_frame_axis_y(ax2, ax3, ay)
            if np.dot(ly, ly2) < 0:
                ly2 = -ly2
            if np.dot(ay, ay2) < 0:
                ay2 = -ay2
            linear_inv[i] = _compute_invariants_original(
                position_diff[i], lx, lx2, ly, ly2
            )
            angular_inv[i] = _compute_invariants_original(
                rotation_diff[i], ax, ax2, ay, ay2
            )

        lx, lx2, ly = lx2, lx3, ly2
        ax, ax2, ay = ax2, ax3, ay2

    return {
        "linear_motion_invariants": linear_inv,
        "angular_motion_invariants": angular_inv,
        "linear_frame_initial": frames["linear_frame_initial"],
        "angular_frame_initial": frames["angular_frame_initial"],
        "initial_pose": initial_pose,
    }
