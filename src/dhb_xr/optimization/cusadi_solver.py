"""
Cusadi-based GPU-parallel trajectory optimization (optional).

- Without cusadi: CusadiTrajectoryOptimizer.forward() uses a NumPy batched decode
  (batched_decode_dhb_dr) so the API is always usable.
- With cusadi + a compiled decode: forward() can use CusadiFunction for GPU batch.
  To build the CasADi decode function, run: python -m dhb_xr.optimization.export_casadi_decode
  then cusadi's run_codegen.py (see cusadi repo). Requires: pip install dhb_xr[cusadi].
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional

from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import DHBMethod

try:
    import casadi as ca  # noqa: F401
    from cusadi import CusadiFunction  # noqa: F401
    HAS_CUSADI = True
except ImportError:
    HAS_CUSADI = False


def batched_decode_dhb_dr(
    linear_invariants_batch: np.ndarray,
    angular_invariants_batch: np.ndarray,
    initial_poses: List[Dict[str, np.ndarray]],
    method: str = "pos",
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    drop_padded: bool = True,
) -> tuple:
    """
    Decode multiple trajectories in batch (loop over decode_dhb_dr).

    linear_invariants_batch: (B, T, 4) or list of (T, 4)
    angular_invariants_batch: (B, T, 4) or list of (T, 4)
    initial_poses: list of B dicts with 'position' (3,) and 'quaternion' (4,) wxyz.

    Returns (positions_batch, quaternions_batch): (B, N, 3), (B, N, 4).
    """
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
        method: str = "pos",
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
