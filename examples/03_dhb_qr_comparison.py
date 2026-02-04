"""Compare DHB-DR (Euler) vs DHB-QR (quaternion) encoding."""

import numpy as np
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.encoder.dhb_qr import encode_dhb_qr
from dhb_xr.core.types import DHBMethod, EncodingMethod

def main():
    n = 40
    np.random.seed(1)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    out_dr = encode_dhb_dr(
        positions, quaternions,
        method=EncodingMethod.POSITION, use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    out_qr = encode_dhb_qr(
        positions, quaternions,
        method=EncodingMethod.POSITION, use_default_initial_frames=True,
    )
    print("DHB-DR linear invariants shape:", out_dr["linear_motion_invariants"].shape)
    print("DHB-QR linear invariants shape:", out_qr["linear_motion_invariants"].shape)
    print("DHB-DR: 4 values per component (m + Euler). DHB-QR: 5 values per component (m + quat).")

if __name__ == "__main__":
    main()
