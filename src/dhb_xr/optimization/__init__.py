"""Trajectory optimization: CasADi, Cusadi, Fatrop, PyTorch."""

from dhb_xr.optimization.casadi_solver import generate_trajectory

__all__ = ["generate_trajectory", "get_optimizer"]

try:
    from dhb_xr.optimization.cusadi_solver import batched_decode_dhb_dr, CusadiTrajectoryOptimizer
    __all__.extend(["batched_decode_dhb_dr", "CusadiTrajectoryOptimizer"])
except ImportError:
    batched_decode_dhb_dr = None
    CusadiTrajectoryOptimizer = None

try:
    from dhb_xr.optimization.torch_solver import BatchedTrajectoryOptimizer
    __all__.append("BatchedTrajectoryOptimizer")
except ImportError:
    BatchedTrajectoryOptimizer = None

try:
    from dhb_xr.optimization.fatrop_solver import (
        FatropTrajectoryGenerator,
        ConstrainedTrajectoryGenerator,
        generate_trajectory_fatrop,
    )
    __all__.extend([
        "FatropTrajectoryGenerator",
        "ConstrainedTrajectoryGenerator",
        "generate_trajectory_fatrop",
    ])
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
            return None  # Use CasADi generate_trajectory
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
