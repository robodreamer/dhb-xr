"""Basic encoding and decoding example."""

import numpy as np
from dhb_xr import encode_dhb_dr, decode_dhb_dr
from dhb_xr.core.types import DHBMethod

def main():
    np.random.seed(42)
    n = 50
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    out = encode_dhb_dr(
        positions, quaternions,
        method="pos", use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    print("Encoded invariants shape:", out["linear_motion_invariants"].shape)

    decoded = decode_dhb_dr(
        out["linear_motion_invariants"],
        out["angular_motion_invariants"],
        out["initial_pose"],
        method="pos",
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    print("Decoded positions shape:", decoded["positions"].shape)
    print("Decoded quaternions shape:", decoded["quaternions"].shape)

if __name__ == "__main__":
    main()
