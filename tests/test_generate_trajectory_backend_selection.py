"""Backend selection tests for the public trajectory-generation API."""

from pathlib import Path

import numpy as np
import pytest


def test_generate_trajectory_accepts_cusadi_backend_without_interpolation(monkeypatch):
    import dhb_xr.optimization as optimization

    calls = {}
    decode_specs = []

    def fake_generate_trajectory_cusadi_adapt(**kwargs):
        calls.update(kwargs)
        return {
            "positions": np.zeros((4, 3)),
            "quaternions": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1)),
            "linear_invariants": np.zeros((4, 4)),
            "angular_invariants": np.zeros((4, 4)),
            "solver": "cusadi",
            "optimizer_solver": "cusadi-adapt",
        }

    class FakeCusadiTrajectoryOptimizer:
        def __init__(self, batch_size, dhb_method, decode_spec=None):
            self.batch_size = batch_size
            self.dhb_method = dhb_method
            self.decode_spec = decode_spec
            decode_specs.append(decode_spec)
            self.last_decode_backend = "gpu"
            self.last_decode_metadata = {
                "cusadi_decode": "gpu",
                "cusadi_gpu": True,
                "cusadi_decode_requested_horizon": 100,
                "cusadi_decode_artifact_horizon": 100,
                "cusadi_decode_artifact_path": "/tmp/fn_dhb_decode_linear_100.casadi",
                "cusadi_decode_library_path": "/tmp/libfn_dhb_decode_linear_100.so",
                "cusadi_decode_fallback_reason": None,
            }

        def forward(self, linear_invariants, angular_invariants, initial_poses, drop_padded):
            assert linear_invariants.shape == (1, 4, 4)
            assert angular_invariants.shape == (1, 4, 4)
            assert len(initial_poses) == 1
            assert drop_padded is False
            return (
                np.zeros((1, 4, 3)),
                np.tile(np.array([[[1.0, 0.0, 0.0, 0.0]]]), (1, 4, 1)),
            )

    def fail_ipopt(**_):
        raise AssertionError("cusadi backend must not route through the IPOPT path")

    monkeypatch.setattr(
        optimization,
        "generate_trajectory_cusadi_adapt",
        fake_generate_trajectory_cusadi_adapt,
    )
    monkeypatch.setattr(optimization, "generate_trajectory_ipopt", fail_ipopt)
    monkeypatch.setattr(optimization, "CusadiTrajectoryOptimizer", FakeCusadiTrajectoryOptimizer)
    monkeypatch.setattr(optimization, "HAS_CUSADI", True)
    monkeypatch.setattr(optimization, "HAS_TORCH_CUDA", True)

    result = optimization.generate_trajectory(
        pos_data=np.zeros((4, 3)),
        quat_data=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1)),
        pose_target_init={
            "position": np.zeros(3),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        },
        pose_target_final={
            "position": np.ones(3),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        },
        traj_length=4,
        backend="cusadi",
        allow_fallback=False,
        cusadi_decode_horizon=100,
        cusadi_decode_artifact_dir="/tmp/cusadi-functions",
        cusadi_decode_library_dir="/tmp/dhb-xr-cusadi-libs",
    )

    assert result["solver"] == "cusadi"
    assert calls["traj_length"] == 4
    assert result["optimizer_solver"] == "cusadi-adapt"
    assert result["requested_backend"] == "cusadi"
    assert result["cusadi_decode"] == "gpu"
    assert result["cusadi_gpu"] is True
    assert result["cusadi_decode_requested_horizon"] == 100
    assert result["cusadi_decode_artifact_horizon"] == 100
    assert decode_specs[0].horizon == 100
    assert decode_specs[0].artifact_dir == "/tmp/cusadi-functions"
    assert decode_specs[0].library_dir == "/tmp/dhb-xr-cusadi-libs"


def test_generate_trajectory_reports_cusadi_cpu_decode_without_cuda(monkeypatch):
    import dhb_xr.optimization as optimization

    def fake_generate_trajectory_cusadi_adapt(**_):
        return {
            "positions": np.zeros((4, 3)),
            "quaternions": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1)),
            "linear_invariants": np.zeros((4, 4)),
            "angular_invariants": np.zeros((4, 4)),
            "solver": "cusadi",
            "optimizer_solver": "cusadi-adapt",
        }

    class FakeCusadiTrajectoryOptimizer:
        def __init__(self, batch_size, dhb_method, decode_spec=None):
            self.batch_size = batch_size
            self.dhb_method = dhb_method
            self.decode_spec = decode_spec
            self.last_decode_backend = "cpu"
            self.last_decode_metadata = {
                "cusadi_decode": "cpu",
                "cusadi_gpu": False,
                "cusadi_decode_requested_horizon": 4,
                "cusadi_decode_artifact_horizon": None,
                "cusadi_decode_artifact_path": None,
                "cusadi_decode_library_path": None,
                "cusadi_decode_fallback_reason": "cuda_unavailable",
            }

        def forward(self, linear_invariants, angular_invariants, initial_poses, drop_padded):
            return (
                np.zeros((1, 4, 3)),
                np.tile(np.array([[[1.0, 0.0, 0.0, 0.0]]]), (1, 4, 1)),
            )

    monkeypatch.setattr(
        optimization,
        "generate_trajectory_cusadi_adapt",
        fake_generate_trajectory_cusadi_adapt,
    )
    monkeypatch.setattr(optimization, "CusadiTrajectoryOptimizer", FakeCusadiTrajectoryOptimizer)
    monkeypatch.setattr(optimization, "HAS_CUSADI", True)
    monkeypatch.setattr(optimization, "HAS_TORCH_CUDA", False)

    result = optimization.generate_trajectory(
        pos_data=np.zeros((4, 3)),
        quat_data=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1)),
        pose_target_init={
            "position": np.zeros(3),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        },
        pose_target_final={
            "position": np.ones(3),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
        },
        traj_length=4,
        backend="cusadi",
        allow_fallback=False,
    )

    assert result["cusadi_decode"] == "cpu"
    assert result["cusadi_gpu"] is False
    assert result["cusadi_decode_fallback_reason"] == "cuda_unavailable"


def test_generate_trajectory_cusadi_requires_adaptation_surface_without_fallback(
    monkeypatch,
):
    import dhb_xr.optimization as optimization

    monkeypatch.setattr(optimization, "generate_trajectory_cusadi_adapt", None)

    with pytest.raises(ImportError, match="CusADi backend requires the CusADi adaptation"):
        optimization.generate_trajectory(
            pos_data=np.zeros((4, 3)),
            quat_data=np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (4, 1)),
            pose_target_init={
                "position": np.zeros(3),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
            },
            pose_target_final={
                "position": np.ones(3),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0]),
            },
            traj_length=4,
            backend="cusadi",
            allow_fallback=False,
        )


def test_cusadi_optimizer_uses_gpu_decode_when_cuda_is_available():
    cusadi_solver = pytest.importorskip("dhb_xr.optimization.cusadi_solver")
    if not cusadi_solver.HAS_TORCH_CUDA or not cusadi_solver.HAS_CUSADI:
        pytest.skip("CusADi CUDA decode requires CUDA plus local CusADi")
    if not Path(cusadi_solver.DEFAULT_FN_PATH).exists():
        pytest.skip(f"compiled CusADi decode function not found: {cusadi_solver.DEFAULT_FN_PATH}")

    optimizer = cusadi_solver.CusadiTrajectoryOptimizer(batch_size=1)
    controls = cusadi_solver.DEFAULT_DECODE_CONTROLS
    linear = np.zeros((1, controls, 4), dtype=np.float64)
    angular = np.zeros((1, controls, 4), dtype=np.float64)
    poses = [
        {
            "position": np.zeros(3, dtype=np.float64),
            "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        }
    ]

    positions, quaternions = optimizer.forward(
        linear,
        angular,
        poses,
        drop_padded=False,
    )

    assert optimizer.last_decode_backend == "gpu"
    assert positions.shape == (1, cusadi_solver.DEFAULT_DECODE_SAMPLES, 3)
    assert quaternions.shape == (1, cusadi_solver.DEFAULT_DECODE_SAMPLES, 4)
    assert np.isfinite(positions).all()
    assert np.isfinite(quaternions).all()


def test_cusadi_decode_artifact_resolution_is_exact(tmp_path):
    from dhb_xr.optimization.cusadi_solver import (
        CusadiDecodeSpec,
        cusadi_decode_filename,
        resolve_cusadi_decode_artifact,
    )

    artifact = tmp_path / cusadi_decode_filename(100)
    artifact.write_text("placeholder")

    resolved = resolve_cusadi_decode_artifact(
        CusadiDecodeSpec(horizon="auto", artifact_dir=str(tmp_path)),
        traj_length=100,
    )
    assert resolved is not None
    assert resolved.horizon == 100
    assert resolved.path == str(artifact)

    assert (
        resolve_cusadi_decode_artifact(
            CusadiDecodeSpec(horizon="auto", artifact_dir=str(tmp_path)),
            traj_length=50,
        )
        is None
    )


def test_cusadi_optimizer_legacy_decode_path_becomes_decode_spec():
    from dhb_xr.optimization.cusadi_solver import (
        DEFAULT_FN_PATH,
        CusadiTrajectoryOptimizer,
    )

    optimizer = CusadiTrajectoryOptimizer(
        batch_size=1,
        decode_casadi_path=DEFAULT_FN_PATH,
    )

    assert optimizer.decode_spec.artifact_path == DEFAULT_FN_PATH
    assert optimizer.decode_spec.horizon == "auto"
    assert optimizer._cusadi_fn is None


def test_cusadi_decode_artifacts_are_bundled_in_dhb_xr_repo():
    from dhb_xr.optimization.cusadi_solver import (
        DEFAULT_CUSADI_FUNCTION_DIR,
        DEFAULT_FN_PATH,
        PACKAGE_CUSADI_CODEGEN_DIR,
        PACKAGE_CUSADI_FUNCTION_DIR,
        SUPPORTED_CUSADI_DECODE_SAMPLES,
        CusadiDecodeSpec,
        cusadi_decode_filename,
        resolve_cusadi_decode_artifact,
    )

    assert DEFAULT_CUSADI_FUNCTION_DIR == PACKAGE_CUSADI_FUNCTION_DIR
    assert Path(DEFAULT_FN_PATH).is_file()
    for samples in SUPPORTED_CUSADI_DECODE_SAMPLES:
        artifact = Path(PACKAGE_CUSADI_FUNCTION_DIR) / cusadi_decode_filename(samples)
        source = Path(PACKAGE_CUSADI_CODEGEN_DIR) / f"fn_dhb_decode_linear_{samples}.cu"
        assert artifact.is_file()
        assert source.is_file()
        resolved = resolve_cusadi_decode_artifact(
            CusadiDecodeSpec(horizon=samples),
            traj_length=samples,
        )
        assert resolved is not None
        assert resolved.horizon == samples
        assert resolved.path == str(artifact)


def test_cusadi_decode_library_resolution_prefers_dhb_xr_cache(tmp_path, monkeypatch):
    from dhb_xr.optimization.cusadi_runtime import (
        cusadi_decode_library_filename,
        resolve_cusadi_decode_library,
    )

    cache_dir = tmp_path / "cache" / "build"
    cache_dir.mkdir(parents=True)
    library = cache_dir / cusadi_decode_library_filename(100)
    library.write_bytes(b"placeholder")
    monkeypatch.setenv("DHB_XR_CUSADI_CACHE", str(tmp_path / "cache"))

    assert resolve_cusadi_decode_library(100) == str(library)


def test_cusadi_decode_build_dry_run_prepares_local_project(tmp_path):
    from dhb_xr.optimization.build_cusadi_decode import build_cusadi_decode_libraries
    from dhb_xr.optimization.cusadi_runtime import cusadi_decode_library_filename

    outputs = build_cusadi_decode_libraries(
        horizons=[50, 100],
        output_dir=str(tmp_path),
        dry_run=True,
    )

    assert outputs == [
        tmp_path / "build" / cusadi_decode_library_filename(50),
        tmp_path / "build" / cusadi_decode_library_filename(100),
    ]
    assert (tmp_path / "CMakeLists.txt").is_file()
    assert (tmp_path / "codegen" / "fn_dhb_decode_linear_50.cu").is_file()
    assert (tmp_path / "codegen" / "fn_dhb_decode_linear_100.cu").is_file()


def test_cusadi_decode_build_forwards_cuda_toolchain_options(tmp_path, monkeypatch):
    from dhb_xr.optimization.build_cusadi_decode import build_cusadi_decode_libraries
    from dhb_xr.optimization.cusadi_runtime import cusadi_decode_library_filename

    commands = []

    def fake_run(command, check):
        commands.append(command)
        if "--build" in command:
            for horizon in (50,):
                output = tmp_path / "build" / cusadi_decode_library_filename(horizon)
                output.touch()

    monkeypatch.setattr("subprocess.run", fake_run)
    build_cusadi_decode_libraries(
        horizons=[50],
        output_dir=str(tmp_path),
        cuda_architectures="120",
        cuda_host_compiler="/usr/bin/g++-13",
        fresh=True,
    )

    configure = commands[0]
    assert "--fresh" in configure
    assert "-DCMAKE_CUDA_ARCHITECTURES=120" in configure
    assert "-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13" in configure


def test_cusadi_optimizer_cpu_fallback_reports_missing_artifact(tmp_path):
    from dhb_xr.optimization.cusadi_solver import (
        CusadiDecodeSpec,
        CusadiTrajectoryOptimizer,
    )

    optimizer = CusadiTrajectoryOptimizer(
        batch_size=1,
        decode_spec=CusadiDecodeSpec(artifact_dir=str(tmp_path)),
    )

    positions, quaternions = optimizer.forward(
        np.zeros((1, 99, 4), dtype=np.float64),
        np.zeros((1, 99, 4), dtype=np.float64),
        [
            {
                "position": np.zeros(3, dtype=np.float64),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            }
        ],
        drop_padded=False,
    )

    assert positions.shape == (1, 99, 3)
    assert quaternions.shape == (1, 99, 4)
    assert optimizer.last_decode_backend == "cpu"
    assert optimizer.last_decode_metadata["cusadi_decode_requested_horizon"] == 100
    assert optimizer.last_decode_metadata["cusadi_decode_artifact_horizon"] is None
    assert optimizer.last_decode_metadata["cusadi_decode_fallback_reason"] in {
        "artifact_not_found",
        "cuda_unavailable",
        "cusadi_unavailable",
    }


def test_cusadi_optimizer_cpu_fallback_reports_missing_library(tmp_path, monkeypatch):
    import dhb_xr.optimization.cusadi_solver as cusadi_solver

    artifact = tmp_path / cusadi_solver.cusadi_decode_filename(100)
    artifact.write_text("placeholder")
    monkeypatch.setattr(cusadi_solver, "HAS_TORCH_CUDA", True)
    monkeypatch.setattr(cusadi_solver, "EXTERNAL_CUSADI_LIBRARY_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("DHB_XR_CUSADI_CACHE", str(tmp_path / "cache"))

    optimizer = cusadi_solver.CusadiTrajectoryOptimizer(
        batch_size=1,
        decode_spec=cusadi_solver.CusadiDecodeSpec(artifact_dir=str(tmp_path)),
    )

    positions, quaternions = optimizer.forward(
        np.zeros((1, 99, 4), dtype=np.float64),
        np.zeros((1, 99, 4), dtype=np.float64),
        [
            {
                "position": np.zeros(3, dtype=np.float64),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            }
        ],
        drop_padded=False,
    )

    assert positions.shape == (1, 99, 3)
    assert quaternions.shape == (1, 99, 4)
    assert optimizer.last_decode_backend == "cpu"
    assert optimizer.last_decode_metadata["cusadi_decode_fallback_reason"] == "library_not_found"
    assert optimizer.last_decode_metadata["cusadi_decode_library_path"] is None


def test_cusadi_optimizer_gpu_required_errors_when_artifact_is_missing(tmp_path):
    from dhb_xr.optimization.cusadi_solver import (
        CusadiDecodeSpec,
        CusadiTrajectoryOptimizer,
    )

    optimizer = CusadiTrajectoryOptimizer(
        batch_size=1,
        decode_spec=CusadiDecodeSpec(
            mode="gpu_required",
            fallback="error",
            artifact_dir=str(tmp_path),
        ),
    )

    with pytest.raises(RuntimeError, match="CusADi GPU decode unavailable"):
        optimizer.forward(
            np.zeros((1, 99, 4), dtype=np.float64),
            np.zeros((1, 99, 4), dtype=np.float64),
            [
                {
                    "position": np.zeros(3, dtype=np.float64),
                    "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                }
            ],
            drop_padded=False,
        )
