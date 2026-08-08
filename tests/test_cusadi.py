"""Tests for Cusadi implementation: batched_decode_dhb_dr and CusadiTrajectoryOptimizer."""

from pathlib import Path

import numpy as np
import pytest

from dhb_xr.core.types import DHBMethod, EncodingMethod


def test_batched_decode_dhb_dr_single_batch():
    from dhb_xr.optimization.cusadi_solver import batched_decode_dhb_dr

    np.random.seed(42)
    num_samples = 12
    lin = np.random.randn(1, num_samples, 4).astype(np.float64)
    ang = np.random.randn(1, num_samples, 4).astype(np.float64)
    inits = [{"position": np.zeros(3), "quaternion": np.array([1.0, 0, 0, 0])}]
    pos, quat = batched_decode_dhb_dr(
        lin,
        ang,
        inits,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    assert pos.shape[0] == 1
    assert quat.shape[0] == 1
    assert pos.shape[1] == num_samples - 2
    assert quat.shape[1] == num_samples - 2
    assert pos.shape[2] == 3
    assert quat.shape[2] == 4


def test_batched_decode_dhb_dr_multiple_batches():
    from dhb_xr.optimization.cusadi_solver import batched_decode_dhb_dr

    np.random.seed(43)
    batch_size, num_samples = 3, 10
    lin = np.random.randn(batch_size, num_samples, 4).astype(np.float64)
    ang = np.random.randn(batch_size, num_samples, 4).astype(np.float64)
    inits = [
        {"position": np.array([i * 0.1, 0, 0]), "quaternion": np.array([1.0, 0, 0, 0])}
        for i in range(batch_size)
    ]
    pos, quat = batched_decode_dhb_dr(
        lin,
        ang,
        inits,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    assert pos.shape == (batch_size, num_samples - 2, 3)
    assert quat.shape == (batch_size, num_samples - 2, 4)


def test_batched_decode_matches_single_decode():
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr
    from dhb_xr.optimization.cusadi_solver import batched_decode_dhb_dr

    np.random.seed(44)
    num_samples = 14
    lin = np.random.randn(num_samples, 4).astype(np.float64)
    ang = np.random.randn(num_samples, 4).astype(np.float64)
    init = {"position": np.array([0.1, 0, 0]), "quaternion": np.array([1.0, 0, 0, 0])}
    single = decode_dhb_dr(
        lin,
        ang,
        init,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    batched_pos, batched_quat = batched_decode_dhb_dr(
        lin[np.newaxis],
        ang[np.newaxis],
        [init],
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    np.testing.assert_allclose(single["positions"], batched_pos[0], atol=1e-12)
    np.testing.assert_allclose(single["quaternions"], batched_quat[0], atol=1e-12)


def test_cusadi_trajectory_optimizer_forward():
    from dhb_xr.optimization.cusadi_solver import CusadiTrajectoryOptimizer

    np.random.seed(45)
    batch_size, num_samples = 2, 8
    lin = np.random.randn(batch_size, num_samples, 4).astype(np.float64)
    ang = np.random.randn(batch_size, num_samples, 4).astype(np.float64)
    inits = [
        {"position": np.zeros(3), "quaternion": np.array([1.0, 0, 0, 0])},
        {"position": np.ones(3) * 0.05, "quaternion": np.array([1.0, 0, 0, 0])},
    ]
    opt = CusadiTrajectoryOptimizer(batch_size=batch_size, dhb_method=DHBMethod.DOUBLE_REFLECTION)
    pos, quat = opt.forward(lin, ang, inits, method=EncodingMethod.POSITION, drop_padded=True)
    assert pos.shape == (batch_size, num_samples - 2, 3)
    assert quat.shape == (batch_size, num_samples - 2, 4)


def test_cusadi_trajectory_optimizer_accepts_string_method():
    from dhb_xr.optimization.cusadi_solver import CusadiTrajectoryOptimizer

    opt = CusadiTrajectoryOptimizer(batch_size=1, dhb_method="double_reflection")
    np.random.seed(46)
    lin = np.random.randn(1, 6, 4).astype(np.float64)
    ang = np.random.randn(1, 6, 4).astype(np.float64)
    inits = [{"position": np.zeros(3), "quaternion": np.array([1.0, 0, 0, 0])}]
    pos, quat = opt.forward(lin, ang, inits)
    assert pos.shape[0] == 1 and quat.shape[0] == 1


def test_cusadi_gpu_rejects_original_three_channel_invariants():
    from dhb_xr.optimization.cusadi_solver import CusadiDecodeSpec, CusadiTrajectoryOptimizer

    optimizer = CusadiTrajectoryOptimizer(
        batch_size=1,
        dhb_method=DHBMethod.ORIGINAL,
        decode_spec=CusadiDecodeSpec(mode="gpu_required", horizon=50, fallback="error"),
    )
    linear = np.zeros((1, 49, 3), dtype=np.float64)
    angular = np.zeros_like(linear)
    initial_poses = [
        {
            "position": np.zeros(3),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        }
    ]

    with pytest.raises(RuntimeError, match="unsupported_dhb_method"):
        optimizer.forward(linear, angular, initial_poses, drop_padded=False)


@pytest.mark.parametrize("samples", [50, 100, 150, 200])
def test_cusadi_optimizer_cpu_decode_large_fixed_horizons(samples):
    from dhb_xr.optimization.cusadi_solver import (
        CusadiDecodeSpec,
        CusadiTrajectoryOptimizer,
        batched_decode_dhb_dr,
    )

    rng = np.random.default_rng(samples)
    batch_size = 3
    controls = samples - 1
    linear = rng.normal(scale=0.01, size=(batch_size, controls, 4)).astype(np.float64)
    angular = rng.normal(scale=0.01, size=(batch_size, controls, 4)).astype(np.float64)
    initial_poses = [
        {
            "position": np.array([0.05 * i, -0.02 * i, 0.01 * i], dtype=np.float64),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
        for i in range(batch_size)
    ]

    optimizer = CusadiTrajectoryOptimizer(
        batch_size=batch_size,
        decode_spec=CusadiDecodeSpec(mode="cpu", horizon=samples),
    )

    positions, quaternions = optimizer.forward(
        linear,
        angular,
        initial_poses,
        drop_padded=False,
    )
    expected_positions, expected_quaternions = batched_decode_dhb_dr(
        linear,
        angular,
        initial_poses,
        drop_padded=False,
    )

    assert optimizer.last_decode_backend == "cpu"
    assert optimizer.last_decode_metadata["cusadi_decode"] == "cpu"
    assert optimizer.last_decode_metadata["cusadi_decode_requested_horizon"] == samples
    assert optimizer.last_decode_metadata["cusadi_decode_fallback_reason"] == "cpu_requested"
    np.testing.assert_allclose(positions, expected_positions, atol=1e-12)
    np.testing.assert_allclose(quaternions, expected_quaternions, atol=1e-12)
    assert np.isfinite(positions).all()
    assert np.isfinite(quaternions).all()


@pytest.mark.parametrize("samples", [50, 100, 150, 200])
def test_cusadi_optimizer_gpu_decode_large_fixed_horizons(samples):
    cusadi_solver = pytest.importorskip("dhb_xr.optimization.cusadi_solver")
    if not cusadi_solver.HAS_TORCH_CUDA or not cusadi_solver.HAS_CUSADI:
        pytest.skip("CusADi CUDA decode requires CUDA plus local CusADi")

    artifact = Path(
        cusadi_solver.DEFAULT_CUSADI_FUNCTION_DIR
    ) / cusadi_solver.cusadi_decode_filename(samples)
    if not artifact.exists():
        pytest.skip(f"compiled CusADi decode function not found: {artifact}")

    batch_size = 2
    controls = samples - 1
    linear = np.zeros((batch_size, controls, 4), dtype=np.float64)
    linear[:, :, 0] = 0.005
    angular = np.zeros((batch_size, controls, 4), dtype=np.float64)
    angular[:, :, 0] = np.linspace(0.0, 0.04, controls)
    angular[:, :, 1] = np.linspace(0.0, 0.02, controls)
    angular[:, :, 2] = np.linspace(0.01, -0.01, controls)
    initial_poses = [
        {
            "position": np.array([0.1 * i, 0.0, 0.0], dtype=np.float64),
            "quaternion": np.array(
                [np.cos(0.15 * (i + 1)), 0.0, 0.0, np.sin(0.15 * (i + 1))],
                dtype=np.float64,
            ),
        }
        for i in range(batch_size)
    ]
    optimizer = cusadi_solver.CusadiTrajectoryOptimizer(
        batch_size=batch_size,
        decode_spec=cusadi_solver.CusadiDecodeSpec(
            mode="gpu_required",
            horizon=samples,
            fallback="error",
        ),
    )
    frame_angles = np.linspace(-0.25, 0.25, batch_size)
    initial_linear_frames = np.stack(
        [
            np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            for angle in frame_angles
        ]
    )

    positions, quaternions = optimizer.forward(
        linear,
        angular,
        initial_poses,
        drop_padded=False,
        initial_linear_frames=initial_linear_frames,
    )

    assert optimizer.last_decode_backend == "gpu"
    assert optimizer.last_decode_metadata["cusadi_decode"] == "gpu"
    assert optimizer.last_decode_metadata["cusadi_decode_requested_horizon"] == samples
    assert optimizer.last_decode_metadata["cusadi_decode_artifact_horizon"] == samples
    assert optimizer.last_decode_metadata["cusadi_quaternion_decode"] == "gpu"
    assert positions.shape == (batch_size, samples, 3)
    assert quaternions.shape == (batch_size, samples, 4)
    assert np.isfinite(positions).all()
    assert np.isfinite(quaternions).all()

    terminal = np.zeros((batch_size, 1, 4), dtype=np.float64)
    expected_positions, expected_quaternions = cusadi_solver.batched_decode_dhb_dr(
        np.concatenate([linear, terminal], axis=1),
        np.concatenate([angular, terminal], axis=1),
        initial_poses,
        drop_padded=False,
        initial_linear_frames=initial_linear_frames,
    )
    np.testing.assert_allclose(positions, expected_positions, atol=1e-12)
    alignment = np.abs(np.sum(quaternions * expected_quaternions, axis=-1))
    np.testing.assert_allclose(alignment, np.ones_like(alignment), atol=1e-12)


def test_initial_linear_frames_are_independent_from_tool_quaternions():
    from dhb_xr.optimization.cusadi_solver import batched_decode_dhb_dr

    samples = 8
    linear = np.zeros((1, samples, 4), dtype=np.float64)
    linear[:, :, 0] = 0.01
    angular = np.zeros_like(linear)
    tool_quaternion = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    initial_poses = [
        {
            "position": np.zeros(3),
            "quaternion": tool_quaternion,
        }
    ]

    default_positions, default_quaternions = batched_decode_dhb_dr(
        linear,
        angular,
        initial_poses,
        drop_padded=False,
    )
    frame = np.array([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    rotated_positions, rotated_quaternions = batched_decode_dhb_dr(
        linear,
        angular,
        initial_poses,
        drop_padded=False,
        initial_linear_frames=frame,
    )

    np.testing.assert_allclose(default_positions[0, :, 1:], 0.0, atol=1e-12)
    np.testing.assert_allclose(rotated_positions[0, :, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(rotated_positions[0, :, 1], default_positions[0, :, 0], atol=1e-12)
    np.testing.assert_allclose(default_quaternions, rotated_quaternions, atol=1e-12)


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("casadi") is None,
    reason="casadi not installed (pip install dhb_xr[optimization])",
)
def test_export_casadi_decode_step():
    """Export CasADi decode step to a .casadi file (requires casadi)."""
    import os
    import tempfile

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


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("casadi") is None,
    reason="casadi not installed (pip install dhb_xr[optimization])",
)
def test_export_casadi_linear_decode_matches_fixed_horizon_contract():
    from dhb_xr.optimization.export_casadi_decode import build_linear_decode_casadi

    samples = 80
    fn = build_linear_decode_casadi(samples=samples)
    assert fn.name() == "fn_dhb_decode_linear_80"
    assert fn.nnz_in(0) == (samples - 1) * 4
    assert fn.size_out(0) == (samples, 3)

    controls = np.zeros((samples - 1, 4), dtype=np.float64)
    controls[:, 0] = 0.01
    positions = np.asarray(
        fn(controls.reshape(-1), np.zeros(3), np.eye(3).reshape(-1)),
        dtype=np.float64,
    )

    assert positions.shape == (samples, 3)
    np.testing.assert_allclose(positions[0], np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(positions[-1], [0.79, 0.0, 0.0], atol=1e-12)


@pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("casadi") is None,
    reason="casadi not installed (pip install dhb_xr[optimization])",
)
def test_export_casadi_decode_cli_writes_multiple_canonical_lengths(tmp_path, monkeypatch):
    from dhb_xr.optimization.export_casadi_decode import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "export_casadi_decode",
            "--length",
            "50",
            "100",
            "--out-dir",
            str(tmp_path),
        ],
    )

    main()

    assert (tmp_path / "fn_dhb_decode_linear_50.casadi").is_file()
    assert (tmp_path / "fn_dhb_decode_linear_100.casadi").is_file()
