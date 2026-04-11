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


def _interpolate_sequence(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate a (N, k) sequence to (target_len, k)."""
    n, k = seq.shape
    old_t = np.linspace(0, 1, n)
    new_t = np.linspace(0, 1, target_len)
    result = np.empty((target_len, k), dtype=seq.dtype)
    for col in range(k):
        result[:, col] = np.interp(new_t, old_t, seq[:, col])
    return result


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
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be 2-D arrays (N, k)")
    if U.shape[1] != V.shape[1]:
        raise ValueError(f"Feature dimension mismatch: {U.shape[1]} vs {V.shape[1]}")
    # Interpolate shorter sequence to match the longer one
    if U.shape[0] != V.shape[0]:
        target_len = max(U.shape[0], V.shape[0])
        if U.shape[0] != target_len:
            U = _interpolate_sequence(U, target_len)
        if V.shape[0] != target_len:
            V = _interpolate_sequence(V, target_len)
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


def _soft_min(a: float, b: float, c: float, gamma: float) -> float:
    """Soft minimum: -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma)).

    Uses the log-sum-exp trick for numerical stability.
    """
    vals = np.array([a, b, c])
    min_val = np.min(vals)
    return min_val - gamma * np.log(np.sum(np.exp(-(vals - min_val) / gamma)))


def soft_dtw_distance(
    U: np.ndarray,
    V: np.ndarray,
    gamma: float = 1.0,
) -> float:
    """Soft-DTW (differentiable) between two sequences.

    Uses dtaidistance if available, otherwise a pure-numpy soft-DTW
    implementation with the gamma smoothing parameter.  As gamma -> 0
    this converges to hard DTW.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    n, m = len(U), len(V)
    D = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d_ij = float(np.sum((U[i - 1] - V[j - 1]) ** 2))
            D[i, j] = d_ij + _soft_min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1], gamma)
    return float(D[n, m])
