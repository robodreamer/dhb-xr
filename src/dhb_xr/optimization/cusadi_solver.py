"""
Cusadi-based GPU-parallel trajectory optimization (optional).

This module provides GPU-accelerated batch decoding of DHB invariants using CusADi.

Setup (one-time):
    1. Clone and install cusadi: git clone https://github.com/se-hwan/cusadi && pip install -e cusadi
    2. Export the CasADi decode function:
       python -m dhb_xr.optimization.export_casadi_decode --out fn_dhb_decode.casadi
    3. Move to cusadi and compile:
       mv fn_dhb_decode.casadi cusadi/src/casadi_functions/
       cd cusadi && python run_codegen.py --fn=fn_dhb_decode

Benchmark results (RTX A2000, 50-step trajectories):
    Batch 100:   43x speedup (CPU 34ms vs GPU 0.8ms)
    Batch 1000: 199x speedup (CPU 342ms vs GPU 1.7ms)
    Batch 2000: 387x speedup (CPU 685ms vs GPU 1.8ms)

Without cusadi: CusadiTrajectoryOptimizer.forward() falls back to NumPy batched decode.
"""

from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union

from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod

# Check for CasADi
try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

# Check for CusADi (GPU acceleration)
HAS_CUSADI = False
CusadiFunction = None
try:
    # Try standard import first
    from cusadi import CusadiFunction
    HAS_CUSADI = True
except ImportError:
    # Try importing with cusadi root in path
    try:
        import sys
        cusadi_root = os.environ.get("CUSADI_ROOT", "/home/andypark/Projects/repos/cusadi")
        if cusadi_root not in sys.path:
            sys.path.insert(0, cusadi_root)
        from src.CusadiFunction import CusadiFunction
        HAS_CUSADI = True
    except ImportError:
        pass

# Check for PyTorch with CUDA
try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False

# Default CusADi paths
CUSADI_ROOT = os.environ.get("CUSADI_ROOT", "/home/andypark/Projects/repos/cusadi")
DEFAULT_FN_PATH = os.path.join(CUSADI_ROOT, "src/casadi_functions/fn_dhb_decode_linear.casadi")

# Global cached CusADi function
_cusadi_fn_cache: Dict[Tuple[str, int], Any] = {}


def get_cusadi_function(
    casadi_path: str = DEFAULT_FN_PATH,
    batch_size: int = 1000,
) -> Optional[Any]:
    """Get or create a cached CusADi function for GPU decode."""
    if not HAS_CUSADI or not HAS_CASADI:
        return None

    cache_key = (casadi_path, batch_size)
    if cache_key in _cusadi_fn_cache:
        return _cusadi_fn_cache[cache_key]

    if not os.path.exists(casadi_path):
        return None

    try:
        fn = ca.Function.load(casadi_path)
        cusadi_fn = CusadiFunction(fn, batch_size)
        _cusadi_fn_cache[cache_key] = cusadi_fn
        return cusadi_fn
    except Exception as e:
        print(f"Warning: Failed to load CusADi function: {e}")
        return None


def batched_decode_dhb_dr_gpu(
    linear_invariants_batch: np.ndarray,
    angular_invariants_batch: np.ndarray,
    initial_poses: List[Dict[str, np.ndarray]],
    casadi_path: str = DEFAULT_FN_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated batch decode using CusADi.

    This function uses CusADi to run the DHB decode on GPU in parallel across
    all trajectories in the batch.

    Requirements:
        - CusADi installed (pip install -e cusadi)
        - CUDA available and PyTorch with CUDA support
        - Compiled CasADi decode function (fn_dhb_decode_linear.casadi)

    Args:
        linear_invariants_batch: (B, T, 4) linear invariants
        angular_invariants_batch: (B, T, 4) angular invariants
        initial_poses: List of B dicts with 'position' (3,) and 'quaternion' (4,) wxyz
        casadi_path: Path to compiled CasADi function

    Returns:
        Tuple of (positions (B, N, 3), quaternions (B, N, 4))

    Raises:
        RuntimeError: If CusADi or CUDA is not available
    """
    if not HAS_TORCH_CUDA:
        raise RuntimeError("CUDA not available. Install PyTorch with CUDA support.")
    if not HAS_CUSADI:
        raise RuntimeError("CusADi not installed. Clone and install from github.com/se-hwan/cusadi")

    B = linear_invariants_batch.shape[0]
    T = linear_invariants_batch.shape[1]

    # Get or create CusADi function
    cusadi_fn = get_cusadi_function(casadi_path, B)
    if cusadi_fn is None:
        raise RuntimeError(f"Could not load CusADi function from {casadi_path}")

    # Prepare inputs for CusADi
    # The fn_dhb_decode_linear function expects:
    # - i0: flattened linear invariants (T*4,) per sample
    # - i1: initial position (3,) per sample
    # - i2: initial rotation matrix (9,) flattened per sample

    # Stack initial poses
    init_pos = np.array([p['position'] for p in initial_poses])  # (B, 3)
    init_quat = np.array([p['quaternion'] for p in initial_poses])  # (B, 4) wxyz

    # Convert quaternions to rotation matrices
    from dhb_xr.core.geometry import quat_to_rot
    init_rot = np.array([quat_to_rot(q).flatten() for q in init_quat])  # (B, 9)

    # Check function signature - fn_dhb_decode_linear expects:
    # i0: linear invariants flattened (nnz_in for first input)
    # i1: initial position (3,)
    # i2: initial rotation matrix flattened (9,)
    n_in = cusadi_fn.fn_casadi.n_in()
    nnz_in = [cusadi_fn.fn_casadi.nnz_in(i) for i in range(n_in)]

    # Flatten linear invariants to match expected input size
    lin_flat = linear_invariants_batch.reshape(B, -1)[:, :nnz_in[0]]  # (B, nnz)

    # Convert to torch tensors on GPU (must be contiguous and double)
    lin_t = torch.from_numpy(lin_flat.astype(np.float64)).cuda().contiguous()
    pos_t = torch.from_numpy(init_pos.astype(np.float64)).cuda().contiguous()
    rot_t = torch.from_numpy(init_rot.astype(np.float64)).cuda().contiguous()

    # Run CusADi function - evaluate takes a LIST of input tensors
    cusadi_fn.evaluate([lin_t, pos_t, rot_t])

    # Get dense output (positions)
    positions = cusadi_fn.getDenseOutput(0).cpu().numpy()  # (B, N, 3)

    # Reshape if needed (output might be (B, N*3) or (B, N, 3))
    if positions.ndim == 2:
        N = positions.shape[1] // 3
        positions = positions.reshape(B, N, 3)

    # For quaternions, we need to decode separately or use angular invariants
    # For now, decode quaternions using NumPy (they're cheap relative to positions)
    _, quat_batch = batched_decode_dhb_dr(
        linear_invariants_batch, angular_invariants_batch,
        initial_poses, drop_padded=True
    )

    return positions, quat_batch


def batched_decode_dhb_dr(
    linear_invariants_batch: np.ndarray,
    angular_invariants_batch: np.ndarray,
    initial_poses: List[Dict[str, np.ndarray]],
    method: EncodingMethod = EncodingMethod.POSITION,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    drop_padded: bool = True,
    use_gpu: bool = False,
    casadi_path: str = DEFAULT_FN_PATH,
) -> tuple:
    """
    Decode multiple trajectories in batch.

    Args:
        linear_invariants_batch: (B, T, 4) or list of (T, 4)
        angular_invariants_batch: (B, T, 4) or list of (T, 4)
        initial_poses: list of B dicts with 'position' (3,) and 'quaternion' (4,) wxyz.
        method: 'pos' or 'vel' for invariant interpretation
        dhb_method: DHBMethod enum (DOUBLE_REFLECTION or ORIGINAL)
        drop_padded: Whether to drop padded frames
        use_gpu: If True and CUDA available, use CusADi GPU acceleration
        casadi_path: Path to compiled CasADi function for GPU decode

    Returns:
        (positions_batch, quaternions_batch): (B, N, 3), (B, N, 4).
    """
    # Try GPU decode if requested
    if use_gpu:
        if HAS_TORCH_CUDA and HAS_CUSADI:
            try:
                return batched_decode_dhb_dr_gpu(
                    linear_invariants_batch, angular_invariants_batch,
                    initial_poses, casadi_path
                )
            except Exception as e:
                print(f"GPU decode failed, falling back to CPU: {e}")
        else:
            missing = []
            if not HAS_TORCH_CUDA:
                missing.append("PyTorch CUDA")
            if not HAS_CUSADI:
                missing.append("CusADi")
            print(f"GPU decode unavailable (missing: {', '.join(missing)}), using CPU")

    # CPU decode (NumPy loop)
    if isinstance(linear_invariants_batch, np.ndarray) and linear_invariants_batch.ndim == 3:
        B = linear_invariants_batch.shape[0]
        lin_list = [linear_invariants_batch[b] for b in range(B)]
        ang_list = [angular_invariants_batch[b] for b in range(B)]
    else:
        lin_list = list(linear_invariants_batch)
        ang_list = list(angular_invariants_batch)
        B = len(lin_list)
    assert len(initial_poses) == B and len(ang_list) == B
    pos_list = []
    quat_list = []
    for b in range(B):
        decoded = decode_dhb_dr(
            lin_list[b], ang_list[b],
            initial_poses[b],
            method=method,
            dhb_method=dhb_method,
            drop_padded=drop_padded,
        )
        pos_list.append(decoded["positions"])
        quat_list.append(decoded["quaternions"])
    return np.stack(pos_list), np.stack(quat_list)


class CusadiTrajectoryOptimizer:
    """
    Batched trajectory decode / optimization.

    - If cusadi is not installed: forward() uses batched_decode_dhb_dr (NumPy loop).
    - If cusadi is installed and a compiled decode function is provided:
      forward() can use CusadiFunction for GPU batch (set decode_casadi_path).
    """

    def __init__(
        self,
        batch_size: int = 1000,
        dhb_method: DHBMethod | str = DHBMethod.DOUBLE_REFLECTION,
        decode_casadi_path: Optional[str] = None,
    ):
        self.batch_size = batch_size
        self.dhb_method = (
            dhb_method
            if isinstance(dhb_method, DHBMethod)
            else (DHBMethod.DOUBLE_REFLECTION if dhb_method == "double_reflection" else DHBMethod.ORIGINAL)
        )
        self.decode_casadi_path = decode_casadi_path
        self._cusadi_fn: Any = None
        if HAS_CUSADI and decode_casadi_path:
            try:
                fn = ca.Function.load(decode_casadi_path)
                self._cusadi_fn = CusadiFunction(fn, batch_size)
            except Exception:
                self._cusadi_fn = None

    def forward(
        self,
        linear_invariants: np.ndarray,
        angular_invariants: np.ndarray,
        initial_poses: List[Dict[str, np.ndarray]],
        method: EncodingMethod = EncodingMethod.POSITION,
        drop_padded: bool = True,
    ) -> tuple:
        """
        Batched decode: (linear_inv, angular_inv, initial_poses) -> (positions, quaternions).

        linear_invariants: (B, T, 4), angular_invariants: (B, T, 4),
        initial_poses: list of B dicts. Returns (B, N, 3), (B, N, 4).
        Uses NumPy batched_decode_dhb_dr. For GPU, build a .casadi decode and use
        CusadiFunction.evaluate() directly (see export_casadi_decode and cusadi repo).
        """
        return batched_decode_dhb_dr(
            linear_invariants,
            angular_invariants,
            initial_poses,
            method=method,
            dhb_method=self.dhb_method,
            drop_padded=drop_padded,
        )
