"""Tests for imitation learning losses."""

import numpy as np
import pytest
from dhb_xr.losses.invariant_loss import invariant_matching_loss
from dhb_xr.losses.geodesic_loss import so3_geodesic_loss_np, se3_geodesic_loss_np
from dhb_xr.losses.hybrid_loss import hybrid_invariant_pose_loss


def test_invariant_matching_loss():
    U = np.random.randn(10, 8) * 0.1
    V = U + np.random.randn(10, 8) * 0.01
    loss = invariant_matching_loss(U, V, method="dhb_dr")
    assert loss >= 0
    np.testing.assert_almost_equal(invariant_matching_loss(U, U, method="dhb_dr"), 0.0)


def test_so3_geodesic_loss():
    from dhb_xr.core import geometry as geom
    R = geom.axis_angle_to_rot(np.array([0.1, 0.2, 0.3]))
    loss = so3_geodesic_loss_np(R, R)
    assert loss >= 0
    np.testing.assert_almost_equal(loss, 0.0)


def test_se3_geodesic_loss():
    pos = np.array([0.0, 0.0, 0.0])
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    loss = se3_geodesic_loss_np(pos, quat, pos, quat, beta=1.0)
    np.testing.assert_almost_equal(loss, 0.0)
