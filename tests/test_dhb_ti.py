"""Tests for DHB-TI (time-invariant reparameterization)."""

import numpy as np
import pytest
from dhb_xr.encoder.dhb_ti import (
    compute_progress,
    resample_by_progress,
    encode_dhb_dr_ti,
    encode_dhb_qr_ti,
)
from dhb_xr.core.types import DHBMethod


def test_compute_progress_translation():
    n = 10
    positions = np.cumsum(np.ones((n, 3)) * 0.1, axis=0)
    quaternions = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
    progress = compute_progress(positions, quaternions, kind="translation")
    assert progress.shape == (n,)
    assert progress[0] == 0
    assert progress[-1] > 0
    np.testing.assert_allclose(np.diff(progress), np.sqrt(0.03), atol=1e-10)


def test_compute_progress_angular():
    n = 10
    positions = np.zeros((n, 3))
    from dhb_xr.core import geometry as geom
    quaternions = np.array([
        geom.axis_angle_to_quat(np.array([0.1 * i, 0, 0]))
        for i in range(n)
    ])
    progress = compute_progress(positions, quaternions, kind="angular")
    assert progress.shape == (n,)
    assert progress[0] == 0
    assert progress[-1] > 0


def test_compute_progress_hybrid():
    n = 8
    np.random.seed(1)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
    progress = compute_progress(positions, quaternions, kind="hybrid", alpha=0.5)
    assert progress.shape == (n,)
    assert progress[0] == 0
    assert progress[-1] >= 0


def test_resample_by_progress():
    n = 20
    np.random.seed(2)
    positions = np.cumsum(np.random.randn(n, 3) * 0.02, axis=0)
    quaternions = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
    M = 12
    pos_m, quat_m = resample_by_progress(
        positions, quaternions, M,
        progress_kind="hybrid", alpha=0.5,
    )
    assert pos_m.shape == (M, 3)
    assert quat_m.shape == (M, 4)
    np.testing.assert_allclose(pos_m[0], positions[0], atol=1e-8)
    np.testing.assert_allclose(pos_m[-1], positions[-1], atol=1e-8)
    np.testing.assert_allclose(quat_m[0], quaternions[0], atol=1e-8)
    np.testing.assert_allclose(quat_m[-1], quaternions[-1], atol=1e-8)


def test_encode_dhb_dr_ti():
    n = 25
    np.random.seed(3)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
    M = 20
    out = encode_dhb_dr_ti(
        positions, quaternions, M,
        progress_kind="hybrid", alpha=0.5,
        method="pos", use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    assert "linear_motion_invariants" in out
    assert "angular_motion_invariants" in out
    assert "initial_pose" in out
    assert out["linear_motion_invariants"].shape[1] == 4
    assert out["angular_motion_invariants"].shape[1] == 4


def test_resample_by_progress_with_given_progress():
    n = 15
    positions = np.cumsum(np.ones((n, 3)) * 0.1, axis=0)
    quaternions = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
    progress = compute_progress(positions, quaternions, kind="translation")
    pos_m, quat_m = resample_by_progress(
        positions, quaternions, 10, progress=progress,
    )
    assert pos_m.shape == (10, 3)
    assert quat_m.shape == (10, 4)


def test_resample_by_progress_raises_for_short_trajectory():
    with pytest.raises(ValueError, match="at least 2"):
        resample_by_progress(
            np.zeros((1, 3)), np.ones((1, 4)), 5,
        )


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("dhb_xr.encoder.dhb_qr") is None
    or getattr(__import__("dhb_xr.encoder", fromlist=["encode_dhb_qr"]), "encode_dhb_qr", None) is None,
    reason="encode_dhb_qr not available",
)
def test_encode_dhb_qr_ti():
    n = 20
    np.random.seed(4)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
    M = 15
    out = encode_dhb_qr_ti(
        positions, quaternions, M,
        progress_kind="hybrid", alpha=0.5,
        method="pos", use_default_initial_frames=True,
    )
    assert "linear_motion_invariants" in out
    assert out["linear_motion_invariants"].shape[1] == 5
