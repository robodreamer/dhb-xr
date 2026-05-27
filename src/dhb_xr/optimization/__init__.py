"""Trajectory optimization: FATROP, CasADi, Cusadi, PyTorch."""

from typing import Any, Dict, Optional

import numpy as np

from dhb_xr.core.types import DHBMethod
from dhb_xr.optimization.casadi_solver import (
    generate_trajectory as generate_trajectory_ipopt,
)

__all__ = ["generate_trajectory", "generate_trajectory_ipopt", "get_optimizer"]

try:
    from dhb_xr.optimization.cusadi_solver import (
        HAS_CUSADI,
        HAS_TORCH_CUDA,
        CusadiDecodeSpec,
        CusadiTrajectoryOptimizer,
        batched_decode_dhb_dr,
        generate_trajectory_cusadi_adapt,
    )

    __all__.extend(
        [
            "batched_decode_dhb_dr",
            "CusadiDecodeSpec",
            "CusadiTrajectoryOptimizer",
            "generate_trajectory_cusadi_adapt",
        ]
    )
except ImportError:
    HAS_CUSADI = False
    HAS_TORCH_CUDA = False
    batched_decode_dhb_dr = None
    CusadiDecodeSpec = None
    CusadiTrajectoryOptimizer = None
    generate_trajectory_cusadi_adapt = None

try:
    from dhb_xr.optimization.torch_solver import BatchedTrajectoryOptimizer

    __all__.append("BatchedTrajectoryOptimizer")
except ImportError:
    BatchedTrajectoryOptimizer = None

try:
    from dhb_xr.optimization.fatrop_solver import (
        ConstrainedTrajectoryGenerator,
        FatropTrajectoryGenerator,
        generate_trajectory_fatrop,
    )

    __all__.extend(
        [
            "FatropTrajectoryGenerator",
            "ConstrainedTrajectoryGenerator",
            "generate_trajectory_fatrop",
        ]
    )
except ImportError:
    FatropTrajectoryGenerator = None
    ConstrainedTrajectoryGenerator = None
    generate_trajectory_fatrop = None


def get_optimizer(backend="auto", batch_size=1, device="cpu", **kwargs):
    """Factory for optimal backend selection.

    Args:
        backend: One of "auto", "torch", "cusadi", "fatrop", "ipopt"
        batch_size: Batch size for batched optimizers
        device: Device for torch-based optimizers
        **kwargs: Additional arguments passed to the optimizer

    Returns:
        Optimizer instance or None
    """
    if backend == "auto":
        if batch_size == 1:
            if FatropTrajectoryGenerator is not None:
                return FatropTrajectoryGenerator(use_fatrop=True, **kwargs)
            return None  # Use generate_trajectory_ipopt
        try:
            from dhb_xr.optimization.torch_solver import BatchedTrajectoryOptimizer

            return BatchedTrajectoryOptimizer(device=device)
        except ImportError:
            return None
    if backend == "torch":
        from dhb_xr.optimization.torch_solver import BatchedTrajectoryOptimizer

        return BatchedTrajectoryOptimizer(device=device)
    if backend == "cusadi":
        if CusadiTrajectoryOptimizer is not None:
            return CusadiTrajectoryOptimizer(batch_size=batch_size)
        return None
    if backend == "fatrop":
        if FatropTrajectoryGenerator is not None:
            return FatropTrajectoryGenerator(use_fatrop=True, **kwargs)
        return None
    if backend == "ipopt":
        if FatropTrajectoryGenerator is not None:
            return FatropTrajectoryGenerator(use_fatrop=False, **kwargs)
        return None
    return None


def generate_trajectory(
    pos_data: np.ndarray,
    quat_data: np.ndarray,
    pose_target_init: Dict[str, np.ndarray],
    pose_target_final: Dict[str, np.ndarray],
    traj_length: int,
    smoothing: bool = False,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    enable_smoothing_objective: bool = False,
    enable_collision_constraints: bool = False,
    weights: Optional[np.ndarray] = None,
    use_casadi: bool = True,
    verbose: bool = False,
    backend: str = "auto",
    allow_fallback: bool = False,
    cusadi_decode: str = "auto",
    cusadi_decode_horizon: Any = "auto",
    cusadi_decode_fallback: str = "cpu",
    cusadi_decode_artifact_dir: Optional[str] = None,
    cusadi_decode_library_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate a boundary-adapted trajectory, preferring FATROP for single paths.

    ``backend="auto"`` uses FATROP when the optional Rockit/FATROP stack is
    importable. Optimizer fallback is disabled by default; set
    ``allow_fallback=True`` or choose ``backend="ipopt"`` / ``backend="interpolation"``
    explicitly for legacy behavior. ``backend="cusadi"`` performs boundary
    adaptation through a CusADi-exportable fixed-horizon function, then runs the
    adapted DHB invariants through an exact-horizon CusADi decode artifact when
    available. Decode falls back to CPU by default; use
    ``cusadi_decode="gpu_required"`` or ``cusadi_decode_fallback="error"`` when
    missing CUDA/artifacts/libraries should be hard failures. Set
    ``cusadi_decode_library_dir`` to use prebuilt decode libraries outside the
    default dhb_xr cache. Optimizer fallback never
    falls back to the slower IPOPT/interpolation path unless ``allow_fallback=True``
    is explicitly set.
    """

    backend = backend.lower()
    if backend not in {"auto", "fatrop", "ipopt", "casadi", "cusadi", "interpolation"}:
        raise ValueError(
            "backend must be auto, fatrop, ipopt, casadi, cusadi, or interpolation"
        )

    if backend == "cusadi":
        if generate_trajectory_cusadi_adapt is None:
            if not allow_fallback:
                raise ImportError(
                    "CusADi backend requires the CusADi adaptation surface. "
                    "Install dhb_xr[optimization], request backend='fatrop', or enable "
                    "allow_fallback only for exploratory IPOPT runs."
                )
            result = generate_trajectory_ipopt(
                pos_data=pos_data,
                quat_data=quat_data,
                pose_target_init=pose_target_init,
                pose_target_final=pose_target_final,
                traj_length=traj_length,
                smoothing=smoothing,
                dhb_method=dhb_method,
                enable_smoothing_objective=enable_smoothing_objective,
                enable_collision_constraints=enable_collision_constraints,
                weights=weights,
                use_casadi=True,
                verbose=verbose,
            )
        else:
            result = generate_trajectory_cusadi_adapt(
                pos_data=pos_data,
                quat_data=quat_data,
                pose_target_init=pose_target_init,
                pose_target_final=pose_target_final,
                traj_length=traj_length,
                dhb_method=dhb_method,
            )
        actual_solver = str(result.get("solver", ""))
        if actual_solver not in {"cusadi", "casadi"} and not allow_fallback:
            raise RuntimeError(
                "CusADi backend requires a successful CusADi adaptation before "
                f"decode; got solver={actual_solver!r}. Enable allow_fallback only "
                "for exploratory runs."
            )
        normalized = _normalize_trajectory_result(result, requested_backend="cusadi")
        normalized["optimizer_solver"] = result.get("optimizer_solver", actual_solver)
        normalized["cusadi_decode"] = "unavailable"
        normalized["cusadi_gpu"] = bool(HAS_CUSADI and HAS_TORCH_CUDA)
        if (
            CusadiTrajectoryOptimizer is not None
            and "linear_motion_invariant" in normalized
            and "angular_motion_invariant" in normalized
        ):
            decode_spec = CusadiDecodeSpec(
                mode=cusadi_decode,
                horizon=cusadi_decode_horizon,
                fallback=cusadi_decode_fallback,
                artifact_dir=cusadi_decode_artifact_dir,
                library_dir=cusadi_decode_library_dir,
            )
            optimizer = CusadiTrajectoryOptimizer(
                batch_size=1,
                dhb_method=dhb_method,
                decode_spec=decode_spec,
            )
            decoded_positions, decoded_quats = optimizer.forward(
                np.asarray(normalized["linear_motion_invariant"])[None, ...],
                np.asarray(normalized["angular_motion_invariant"])[None, ...],
                [pose_target_init],
                drop_padded=False,
            )
            normalized["cusadi_decoded_pos_data"] = decoded_positions[0]
            normalized["cusadi_decoded_quat_data"] = decoded_quats[0]
            normalized.update(getattr(optimizer, "last_decode_metadata", {}))
        normalized["solver"] = "cusadi"
        return normalized

    if (
        backend in {"auto", "fatrop"}
        and use_casadi
        and generate_trajectory_fatrop is not None
    ):
        try:
            result = generate_trajectory_fatrop(
                pos_data=pos_data,
                quat_data=quat_data,
                pose_target_init=pose_target_init,
                pose_target_final=pose_target_final,
                traj_length=traj_length,
                use_fatrop=True,
                verbose=verbose,
            )
            return _normalize_trajectory_result(result, requested_backend="fatrop")
        except ImportError:
            if backend == "fatrop" or not allow_fallback:
                raise
        except Exception:
            if backend == "fatrop" or not allow_fallback:
                raise

    if backend == "auto" and use_casadi and not allow_fallback:
        raise ImportError(
            "FATROP backend is unavailable. Install dhb_xr[fatrop], set "
            "allow_fallback=True, or request backend='ipopt'/'interpolation' explicitly."
        )

    return generate_trajectory_ipopt(
        pos_data=pos_data,
        quat_data=quat_data,
        pose_target_init=pose_target_init,
        pose_target_final=pose_target_final,
        traj_length=traj_length,
        smoothing=smoothing,
        dhb_method=dhb_method,
        enable_smoothing_objective=enable_smoothing_objective,
        enable_collision_constraints=enable_collision_constraints,
        weights=weights,
        use_casadi=bool(use_casadi and backend != "interpolation"),
        verbose=verbose,
    )


def _normalize_trajectory_result(
    result: Dict[str, Any], requested_backend: str
) -> Dict[str, Any]:
    normalized = dict(result)
    if "positions" in normalized and "adapted_pos_data" not in normalized:
        normalized["adapted_pos_data"] = normalized["positions"]
    if "quaternions" in normalized and "adapted_quat_data" not in normalized:
        normalized["adapted_quat_data"] = normalized["quaternions"]
    if (
        "linear_invariants" in normalized
        and "linear_motion_invariant" not in normalized
    ):
        normalized["linear_motion_invariant"] = normalized["linear_invariants"]
    if (
        "angular_invariants" in normalized
        and "angular_motion_invariant" not in normalized
    ):
        normalized["angular_motion_invariant"] = normalized["angular_invariants"]
    normalized["requested_backend"] = requested_backend
    return normalized
