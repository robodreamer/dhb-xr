"""Benchmark encode/decode backends (numpy, optional torch/cpp)."""

import time
import numpy as np
from dhb_xr import encode_dhb_dr, decode_dhb_dr
from dhb_xr.core.types import DHBMethod

def main():
    n = 100
    np.random.seed(0)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    # Encode
    start = time.perf_counter()
    for _ in range(50):
        out = encode_dhb_dr(
            positions, quaternions,
            method="pos", use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
        )
    elapsed = (time.perf_counter() - start) / 50
    print(f"Encode (n={n}): {elapsed*1000:.2f} ms per call")

    # Decode
    start = time.perf_counter()
    for _ in range(50):
        decoded = decode_dhb_dr(
            out["linear_motion_invariants"],
            out["angular_motion_invariants"],
            out["initial_pose"],
            method="pos", dhb_method=DHBMethod.DOUBLE_REFLECTION, drop_padded=True,
        )
    elapsed = (time.perf_counter() - start) / 50
    print(f"Decode: {elapsed*1000:.2f} ms per call")

if __name__ == "__main__":
    main()
