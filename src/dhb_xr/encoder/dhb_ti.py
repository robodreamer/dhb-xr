"""
DHB-TI: Time-invariant reparameterization.

Reparameterize a trajectory by a geometric progress variable (translational arc-length,
angular, or hybrid) and resample at uniform progress knots so that DHB-DR/DHB-QR
invariants are approximately independent of execution speed and sampling rate.

Progress variables:
  - translation: s_{i+1} = s_i + ||Δp_i||
  - angular:     θ_{i+1} = θ_i + ||Δr_i||
  - hybrid:      σ_{i+1} = σ_i + α||Δp_i|| + (1-α)||Δr_i||, α in [0,1]

Uniform knots σ_k = k * Σ/(M-1), then interpolate poses at σ_k (position spline, quat SLERP).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Optional, Literal
from scipy.interpolate import CubicSpline

from dhb_xr.core import geometry as geom
from dhb_xr.core.types import DHBMethod

ProgressKind = Literal["translation", "angular", "hybrid"]
_EPS = 1e-12
_MIN_STEP = 1e-10  # minimum progress step to avoid degeneracy in interpolation


def compute_progress(
    positions: np.ndarray,
    quaternions: np.ndarray,
    kind: ProgressKind = "hybrid",
    alpha: float = 0.5,
    min_step: float = _MIN_STEP,
) -> np.ndarray:
    """
    Compute cumulative progress along the trajectory.

    positions: (N, 3), quaternions: (N, 4) wxyz.
    kind: 'translation' (arc-length), 'angular' (rotation magnitude), 'hybrid'.
    alpha: weight for translation in hybrid; (1-alpha) for rotation. Ignored if kind != 'hybrid'.
    min_step: minimum increment per segment to avoid degenerate progress (clamped).

    Returns progress (N,) with progress[0] = 0.
    """
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n = positions.shape[0]
    if n < 2:
        return np.zeros(n)

    delta_p = np.diff(positions, axis=0)
    step_p = np.linalg.norm(delta_p, axis=1)
    step_p = np.maximum(step_p, min_step)

    delta_r = np.array([
        geom.quat_relative_axis_angle(quaternions[i], quaternions[i + 1])
        for i in range(n - 1)
    ])
    step_r = np.linalg.norm(delta_r, axis=1)
    step_r = np.maximum(step_r, min_step)

    if kind == "translation":
        steps = step_p
    elif kind == "angular":
        steps = step_r
    else:
        steps = alpha * step_p + (1.0 - alpha) * step_r

    progress = np.concatenate([[0.0], np.cumsum(steps)])
    return progress


def resample_by_progress(
    positions: np.ndarray,
    quaternions: np.ndarray,
    M: int,
    progress_kind: ProgressKind = "hybrid",
    alpha: float = 0.5,
    progress: Optional[np.ndarray] = None,
    min_step: float = _MIN_STEP,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample trajectory to M poses at uniform progress knots.

    progress_knots: σ_k = k * Σ/(M-1), k = 0,...,M-1.
    Positions interpolated with cubic spline in progress; orientations with SLERP.

    Returns (positions_M, quaternions_M) (M, 3), (M, 4) wxyz.
    """
    positions = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions, dtype=np.float64)
    n = positions.shape[0]
    if n < 2:
        raise ValueError("resample_by_progress requires at least 2 poses")

    if progress is None:
        progress = compute_progress(
            positions, quaternions, kind=progress_kind, alpha=alpha, min_step=min_step
        )
    else:
        progress = np.asarray(progress, dtype=np.float64)
        if progress.shape[0] != n:
            raise ValueError("progress length must match positions/quaternions")

    sigma_total = progress[-1]
    if sigma_total <= 0:
        sigma_total = 1.0
    sigma_knots = np.linspace(0, sigma_total, M, dtype=np.float64)

    pos_resample = np.zeros((M, 3))
    quat_resample = np.zeros((M, 4))
    for j in range(3):
        cs = CubicSpline(progress, positions[:, j])
        pos_resample[:, j] = cs(sigma_knots)

    for k in range(M):
        s = sigma_knots[k]
        if s <= progress[0] + _EPS:
            quat_resample[k] = quaternions[0].copy()
            continue
        if s >= progress[-1] - _EPS:
            quat_resample[k] = quaternions[-1].copy()
            continue
        i = np.searchsorted(progress, s, side="right") - 1
        i = min(max(i, 0), n - 2)
        p_lo, p_hi = progress[i], progress[i + 1]
        segment = p_hi - p_lo
        if segment <= _EPS:
            t = 0.0
        else:
            t = float((s - p_lo) / segment)
        t = np.clip(t, 0.0, 1.0)
        quat_resample[k] = geom.quat_slerp(quaternions[i], quaternions[i + 1], t)

    return pos_resample, quat_resample


def encode_dhb_dr_ti(
    positions: np.ndarray,
    quaternions: np.ndarray,
    M: int,
    progress_kind: ProgressKind = "hybrid",
    alpha: float = 0.5,
    method: str = "pos",
    use_default_initial_frames: bool = True,
    init_pose: Optional[Dict[str, np.ndarray]] = None,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    min_step: float = _MIN_STEP,
    **encode_kw,
) -> Dict[str, Any]:
    """
    Time-invariant encode: reparameterize by progress to M samples, then DHB-DR encode.

    Returns same structure as encode_dhb_dr (linear_motion_invariants, angular_motion_invariants,
    initial_pose, ...).
    """
    from dhb_xr.encoder.dhb_dr import encode_dhb_dr

    pos_m, quat_m = resample_by_progress(
        positions, quaternions, M,
        progress_kind=progress_kind, alpha=alpha, min_step=min_step,
    )
    return encode_dhb_dr(
        pos_m, quat_m,
        method=method,
        use_default_initial_frames=use_default_initial_frames,
        init_pose=init_pose,
        dhb_method=dhb_method,
        **encode_kw,
    )


def encode_dhb_qr_ti(
    positions: np.ndarray,
    quaternions: np.ndarray,
    M: int,
    progress_kind: ProgressKind = "hybrid",
    alpha: float = 0.5,
    method: str = "pos",
    use_default_initial_frames: bool = True,
    init_pose: Optional[Dict[str, np.ndarray]] = None,
    min_step: float = _MIN_STEP,
    **encode_kw,
) -> Dict[str, Any]:
    """
    Time-invariant encode: reparameterize by progress to M samples, then DHB-QR encode.

    Returns same structure as encode_dhb_qr.
    """
    from dhb_xr.encoder.dhb_qr import encode_dhb_qr

    pos_m, quat_m = resample_by_progress(
        positions, quaternions, M,
        progress_kind=progress_kind, alpha=alpha, min_step=min_step,
    )
    return encode_dhb_qr(
        pos_m, quat_m,
        method=method,
        use_default_initial_frames=use_default_initial_frames,
        init_pose=init_pose,
        **encode_kw,
    )
