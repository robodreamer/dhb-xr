"""Trajectory optimization: CasADi, Cusadi, PyTorch."""

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


def get_optimizer(backend="auto", batch_size=1, device="cpu"):
    """Factory for optimal backend selection."""
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
    return None
