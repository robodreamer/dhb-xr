"""Similarity and distance measures for invariant sequences."""

import numpy as np
from typing import Optional

from dhb_xr.core import geometry as geom


def quaternion_geodesic_distance(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Quaternion geodesic distance (rad). q1, q2: (..., 4) wxyz. Returns scalar or array."""
    q1 = np.asarray(q1).reshape(-1, 4)
    q2 = np.asarray(q2).reshape(-1, 4)
    dot = np.abs(np.sum(q1 * q2, axis=-1))
    dot = np.clip(dot, 0, 1)
    return 2 * np.arccos(dot)


def invariant_distance(
    U: np.ndarray,
    V: np.ndarray,
    weights: Optional[np.ndarray] = None,
    use_quat_geodesic: bool = False,
    quat_start_idx: int = 1,
    quat_len: int = 4,
) -> float:
    """
    Aligned L2 distance between invariant sequences.
    U, V: (N, 2*k) or (N, k_lin + k_ang). If use_quat_geodesic, angular quaternion part uses geodesic.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    assert U.shape == V.shape
    if weights is None:
        weights = np.ones(U.shape[1])
    weights = np.asarray(weights).reshape(-1)
    diff = U - V
    if use_quat_geodesic:
        # Assume linear then angular; each component [m, ...]. Angular has quat at [quat_start_idx:quat_start_idx+quat_len]
        half = U.shape[1] // 2
        for i in range(U.shape[0]):
            for j in [0, 1]:
                base = j * half
                if base + quat_start_idx + quat_len <= U.shape[1]:
                    q1 = U[i, base + quat_start_idx : base + quat_start_idx + quat_len]
                    q2 = V[i, base + quat_start_idx : base + quat_start_idx + quat_len]
                    d = quaternion_geodesic_distance(q1, q2)
                    diff[i, base + quat_start_idx] = d
        dist = np.sqrt(np.sum(weights * (diff ** 2)))
    else:
        dist = np.sqrt(np.sum(weights * (diff ** 2)))
    return float(dist)


def soft_dtw_distance(
    U: np.ndarray,
    V: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Soft-DTW (differentiable) between two sequences. Requires dtaidistance or simple implementation."""
    try:
        from dtaidistance import dtw
        return float(dtw.distance(U, V, use_c=True))
    except ImportError:
        n, m = len(U), len(V)
        D = np.zeros((n + 1, m + 1)) + np.inf
        D[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                d_ij = np.sum((U[i - 1] - V[j - 1]) ** 2)
                D[i, j] = d_ij + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
        return float(D[n, m])
