"""Tests for SE(3) invariance property of DHB invariants."""

import numpy as np
import pytest
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import DHBMethod
from dhb_xr.core import geometry as geom


def random_se3():
    """Random SE(3) transform: rotation R, translation t."""
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.uniform(0, np.pi)
    R = geom.axis_angle_to_rot(axis * angle)
    t = np.random.randn(3) * 0.5
    return R, t


def apply_se3_to_trajectory(positions, quaternions, R, t):
    """Apply global SE(3) to trajectory: p' = R@p + t, R' = R_global @ R_local so q' = rot_to_quat(R @ quat_to_rot(q))."""
    positions = np.asarray(positions)
    quaternions = np.asarray(quaternions)
    p_new = (R @ positions.T).T + t
    q_new = np.array([geom.rot_to_quat(R @ geom.quat_to_rot(q)) for q in quaternions])
    return p_new, q_new


def test_se3_invariance():
    """Invariant magnitudes (first column) should be SE(3) invariant; relative rotations may differ by representation."""
    n = 40
    np.random.seed(123)
    positions = np.cumsum(np.random.randn(n, 3) * 0.02, axis=0)
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    out1 = encode_dhb_dr(
        positions, quats,
        method="pos", use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    R, t = random_se3()
    p2, q2 = apply_se3_to_trajectory(positions, quats, R, t)
    out2 = encode_dhb_dr(
        p2, q2,
        method="pos", use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    # Magnitudes (first column of each component) are inner products and should match
    np.testing.assert_allclose(
        out1["linear_motion_invariants"][:, 0],
        out2["linear_motion_invariants"][:, 0],
        atol=1e-5,
        err_msg="Linear magnitude should be SE(3) invariant",
    )
    np.testing.assert_allclose(
        out1["angular_motion_invariants"][:, 0],
        out2["angular_motion_invariants"][:, 0],
        atol=1e-5,
        err_msg="Angular magnitude should be SE(3) invariant",
    )
