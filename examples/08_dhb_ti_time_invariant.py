"""
DHB-TI: Time-invariant reparameterization.

Reparameterize by geometric progress (translational arc-length, angular, or hybrid)
and resample at uniform progress knots so invariants are approximately independent
of execution speed and sampling rate. Then encode with DHB-DR or DHB-QR.
"""

import numpy as np
from dhb_xr.encoder.dhb_ti import (
    compute_progress,
    resample_by_progress,
    encode_dhb_dr_ti,
    encode_dhb_qr_ti,
)
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.core.types import DHBMethod

def main():
    np.random.seed(42)
    n = 50
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    # Progress variables
    progress_trans = compute_progress(positions, quaternions, kind="translation")
    progress_angular = compute_progress(positions, quaternions, kind="angular")
    progress_hybrid = compute_progress(positions, quaternions, kind="hybrid", alpha=0.5)
    print("Progress (translation) total:", progress_trans[-1])
    print("Progress (angular) total:", progress_angular[-1])
    print("Progress (hybrid, α=0.5) total:", progress_hybrid[-1])

    # Resample to M poses at uniform progress
    M = 30
    pos_m, quat_m = resample_by_progress(
        positions, quaternions, M,
        progress_kind="hybrid", alpha=0.5,
    )
    print("Resampled by progress: positions", pos_m.shape, "quaternions", quat_m.shape)

    # Time-invariant encode (DHB-DR): same motion at different "speeds" gives similar invariants
    out_ti = encode_dhb_dr_ti(
        positions, quaternions, M,
        progress_kind="hybrid", alpha=0.5,
        method="pos", use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    print("DHB-TI (DR) linear invariants shape:", out_ti["linear_motion_invariants"].shape)

    # Compare: raw encode vs TI encode (raw depends on sampling density)
    out_raw = encode_dhb_dr(
        positions, quaternions,
        method="pos", use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    print("DHB-DR (raw) linear invariants shape:", out_raw["linear_motion_invariants"].shape)

    # DHB-TI with DHB-QR
    try:
        out_ti_qr = encode_dhb_qr_ti(
            positions, quaternions, M,
            progress_kind="hybrid", alpha=0.5,
            method="pos", use_default_initial_frames=True,
        )
        print("DHB-TI (QR) linear invariants shape:", out_ti_qr["linear_motion_invariants"].shape)
    except ImportError:
        print("DHB-QR not available; skip DHB-TI (QR).")

    print("Done. Use progress_kind='translation'|'angular'|'hybrid' and alpha for rotation-dominant motions.")

if __name__ == "__main__":
    main()
