"""
DHB-DR vs DHB-QR Comparison Example

This example compares the two DHB invariant representations:

DHB-DR (Double-Reflection with Euler angles):
- 4 values per component: magnitude + 3 Euler angles
- Compact representation, may have gimbal lock issues
- Good for most applications

DHB-QR (Double-Reflection with Quaternions):
- 5 values per component: magnitude + unit quaternion (4 values)
- More robust to singularities, smoother interpolation
- Better for applications requiring quaternion-based losses

Both provide SE(3)-invariant representations, but DHB-QR is preferred
for applications involving quaternion operations or optimization.
"""

import numpy as np
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.encoder.dhb_qr import encode_dhb_qr
from dhb_xr.core.types import DHBMethod, EncodingMethod


def run_example():
    """Run the DHB-DR vs DHB-QR comparison example."""
    print("DHB-DR vs DHB-QR Comparison")
    print("=" * 40)

    # Create synthetic trajectory
    n = 40
    np.random.seed(1)
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    print(f"Trajectory: {n} poses")
    print(f"Positions shape: {positions.shape}")
    print(f"Quaternions shape: {quaternions.shape}")

    # Encode with DHB-DR (Euler angles)
    print("\nEncoding with DHB-DR (Euler angles)...")
    out_dr = encode_dhb_dr(
        positions,
        quaternions,
        method=EncodingMethod.POSITION,
        use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    print(f"DHB-DR linear invariants shape: {out_dr['linear_motion_invariants'].shape}")
    print(f"DHB-DR angular invariants shape: {out_dr['angular_motion_invariants'].shape}")

    # Encode with DHB-QR (Quaternions)
    print("\nEncoding with DHB-QR (Quaternions)...")
    out_qr = encode_dhb_qr(
        positions,
        quaternions,
        method=EncodingMethod.POSITION,
        use_default_initial_frames=True,
    )
    print(f"DHB-QR linear invariants shape: {out_qr['linear_motion_invariants'].shape}")
    print(f"DHB-QR angular invariants shape: {out_qr['angular_motion_invariants'].shape}")

    # Compare dimensions
    print("\nDimension comparison:")
    print(f"DHB-DR: {out_dr['linear_motion_invariants'].shape[1]} values per linear component")
    print(f"DHB-QR: {out_qr['linear_motion_invariants'].shape[1]} values per linear component")
    print("\nDHB-DR uses Euler angles (3 values), DHB-QR uses quaternions (4 values)")

    return out_dr, out_qr


def main():
    """Main function for command-line execution."""
    run_example()


if __name__ == "__main__":
    main()
