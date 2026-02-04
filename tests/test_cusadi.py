"""Tests for Cusadi implementation: batched_decode_dhb_dr and CusadiTrajectoryOptimizer."""

import numpy as np
import pytest

from dhb_xr.core.types import DHBMethod, EncodingMethod


def test_batched_decode_dhb_dr_single_batch():
    from dhb_xr.optimization.cusadi_solver import batched_decode_dhb_dr

    np.random.seed(42)
    T = 12
    lin = np.random.randn(1, T, 4).astype(np.float64)
    ang = np.random.randn(1, T, 4).astype(np.float64)
    inits = [{"position": np.zeros(3), "quaternion": np.array([1.0, 0, 0, 0])}]
    pos, quat = batched_decode_dhb_dr(
        lin, ang, inits, method=EncodingMethod.POSITION, dhb_method=DHBMethod.DOUBLE_REFLECTION, drop_padded=True
    )
    assert pos.shape[0] == 1
    assert quat.shape[0] == 1
    assert pos.shape[1] == T - 2
    assert quat.shape[1] == T - 2
    assert pos.shape[2] == 3
    assert quat.shape[2] == 4


def test_batched_decode_dhb_dr_multiple_batches():
    from dhb_xr.optimization.cusadi_solver import batched_decode_dhb_dr

    np.random.seed(43)
    B, T = 3, 10
    lin = np.random.randn(B, T, 4).astype(np.float64)
    ang = np.random.randn(B, T, 4).astype(np.float64)
    inits = [
        {"position": np.array([i * 0.1, 0, 0]), "quaternion": np.array([1.0, 0, 0, 0])}
        for i in range(B)
    ]
    pos, quat = batched_decode_dhb_dr(
        lin, ang, inits, method=EncodingMethod.POSITION, dhb_method=DHBMethod.DOUBLE_REFLECTION, drop_padded=True
    )
    assert pos.shape == (B, T - 2, 3)
    assert quat.shape == (B, T - 2, 4)


def test_batched_decode_matches_single_decode():
    from dhb_xr.optimization.cusadi_solver import batched_decode_dhb_dr
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr

    np.random.seed(44)
    T = 14
    lin = np.random.randn(T, 4).astype(np.float64)
    ang = np.random.randn(T, 4).astype(np.float64)
    init = {"position": np.array([0.1, 0, 0]), "quaternion": np.array([1.0, 0, 0, 0])}
    single = decode_dhb_dr(
        lin, ang, init, method=EncodingMethod.POSITION, dhb_method=DHBMethod.DOUBLE_REFLECTION, drop_padded=True
    )
    batched_pos, batched_quat = batched_decode_dhb_dr(
        lin[np.newaxis], ang[np.newaxis], [init],
        method=EncodingMethod.POSITION, dhb_method=DHBMethod.DOUBLE_REFLECTION, drop_padded=True
    )
    np.testing.assert_allclose(single["positions"], batched_pos[0], atol=1e-12)
    np.testing.assert_allclose(single["quaternions"], batched_quat[0], atol=1e-12)


def test_cusadi_trajectory_optimizer_forward():
    from dhb_xr.optimization.cusadi_solver import CusadiTrajectoryOptimizer

    np.random.seed(45)
    B, T = 2, 8
    lin = np.random.randn(B, T, 4).astype(np.float64)
    ang = np.random.randn(B, T, 4).astype(np.float64)
    inits = [
        {"position": np.zeros(3), "quaternion": np.array([1.0, 0, 0, 0])},
        {"position": np.ones(3) * 0.05, "quaternion": np.array([1.0, 0, 0, 0])},
    ]
    opt = CusadiTrajectoryOptimizer(batch_size=B, dhb_method=DHBMethod.DOUBLE_REFLECTION)
    pos, quat = opt.forward(lin, ang, inits, method=EncodingMethod.POSITION, drop_padded=True)
    assert pos.shape == (B, T - 2, 3)
    assert quat.shape == (B, T - 2, 4)


def test_cusadi_trajectory_optimizer_accepts_string_method():
    from dhb_xr.optimization.cusadi_solver import CusadiTrajectoryOptimizer

    opt = CusadiTrajectoryOptimizer(batch_size=1, dhb_method="double_reflection")
    np.random.seed(46)
    lin = np.random.randn(1, 6, 4).astype(np.float64)
    ang = np.random.randn(1, 6, 4).astype(np.float64)
    inits = [{"position": np.zeros(3), "quaternion": np.array([1.0, 0, 0, 0])}]
    pos, quat = opt.forward(lin, ang, inits)
    assert pos.shape[0] == 1 and quat.shape[0] == 1


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("casadi") is None,
    reason="casadi not installed (pip install dhb_xr[optimization])",
)
def test_export_casadi_decode_step():
    """Export CasADi decode step to a .casadi file (requires casadi)."""
    import tempfile
    import os
    from dhb_xr.optimization.export_casadi_decode import build_decode_step_casadi

    fn = build_decode_step_casadi()
    assert fn is not None
    with tempfile.NamedTemporaryFile(suffix=".casadi", delete=False) as f:
        out_path = f.name
    try:
        fn.save(out_path)
        assert os.path.isfile(out_path)
    finally:
        if os.path.isfile(out_path):
            os.unlink(out_path)
