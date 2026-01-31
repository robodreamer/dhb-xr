"""
Boundary extrapolation for length preservation.
Prepends 2 poses and appends 1 pose so invariant length aligns with N-1 steps.
"""

import numpy as np
from typing import Dict, Any

from dhb_xr.core import geometry as geom


def extrapolate_boundary_poses(
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> tuple:
    """
    Extrapolate boundary poses for length-preserved encoding.

    Prepends g_{-2}, g_{-1} and appends g_N using first-order extrapolation:
    - Position: p_{-1} = 2*p_0 - p_1, p_{-2} = 3*p_0 - 2*p_1, p_N = 2*p_{N-1} - p_{N-2}
    - Rotation: Lie-algebra consistent; R_{-1} = R_0 * exp(-[Δr_0]_×), etc.

    positions: (N, 3)
    quaternions: (N, 4) wxyz

    Returns (positions_ext, quaternions_ext) with shape (N+3, 3) and (N+3, 4).
    """
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n = positions.shape[0]
    assert n >= 2 and quaternions.shape[0] == n

    # Rotation differentials for extrapolation
    rotation_diff = np.zeros((n - 1, 3))
    for i in range(1, n):
        R_prev = geom.quat_to_rot(quaternions[i - 1]).T
        R_curr = geom.quat_to_rot(quaternions[i]).T
        R_rel = R_curr @ R_prev.T
        rotation_diff[i - 1] = geom.rot_to_axis_angle(R_rel)

    # Prepend: p_{-1} = 2*p_0 - p_1, p_{-2} = 3*p_0 - 2*p_1
    dp0 = positions[1] - positions[0]
    p_minus1 = positions[0] - dp0  # 2*p_0 - p_1
    p_minus2 = p_minus1 - dp0  # 3*p_0 - 2*p_1

    # R_{-1} = R_0 * exp(-[Δr_0]_×), Δr_{-1} = Δr_0 for first-order
    dr0 = rotation_diff[0]
    R0 = geom.quat_to_rot(quaternions[0])
    R_minus1 = R0 @ geom.axis_angle_to_rot(-dr0).T
    q_minus1 = geom.rot_to_quat(R_minus1)
    dr_minus1 = dr0  # first-order
    R_minus2_mat = geom.quat_to_rot(q_minus1) @ geom.axis_angle_to_rot(-dr_minus1).T
    q_minus2 = geom.rot_to_quat(R_minus2_mat)

    # Append: p_N = 2*p_{N-1} - p_{N-2}
    p_last = 2 * positions[-1] - positions[-2]
    dr_last = rotation_diff[-1]
    R_last = geom.quat_to_rot(quaternions[-1]) @ geom.axis_angle_to_rot(dr_last).T
    q_last = geom.rot_to_quat(R_last)

    positions_ext = np.vstack(
        (p_minus2.reshape(1, 3), p_minus1.reshape(1, 3), positions, p_last.reshape(1, 3))
    )
    quaternions_ext = np.vstack(
        (
            q_minus2.reshape(1, 4),
            q_minus1.reshape(1, 4),
            quaternions,
            q_last.reshape(1, 4),
        )
    )
    return positions_ext, quaternions_ext


def apply_length_preserving_padding(
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> tuple:
    """
    Same as extrapolate_boundary_poses. Kept for API compatibility.
    """
    return extrapolate_boundary_poses(positions, quaternions)
