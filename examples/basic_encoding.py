"""
Basic DHB Encoding and Decoding Example

This example demonstrates the core DHB-XR functionality: encoding SE(3) trajectories
to invariant representations and decoding them back to poses.

The example:
1. Creates a synthetic trajectory (50 poses with random walk positions)
2. Encodes to DHB-DR invariants using double-reflection method
3. Decodes back to reconstruct the original trajectory
4. Shows the shape preservation of the roundtrip encoding/decoding
"""

import numpy as np
from dhb_xr import encode_dhb_dr, decode_dhb_dr
from dhb_xr.core.types import DHBMethod, EncodingMethod


def run_example():
    """Run the basic encoding/decoding example."""
    print("DHB-XR Basic Encoding Example")
    print("=" * 40)

    # Create synthetic trajectory
    np.random.seed(42)
    n = 50
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    print(f"Original trajectory: {n} poses")
    print(f"Positions shape: {positions.shape}")
    print(f"Quaternions shape: {quaternions.shape}")

    # Encode to DHB invariants
    print("\nEncoding to DHB-DR invariants...")
    out = encode_dhb_dr(
        positions,
        quaternions,
        method=EncodingMethod.POSITION,
        use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    print(f"Linear invariants shape: {out['linear_motion_invariants'].shape}")
    print(f"Angular invariants shape: {out['angular_motion_invariants'].shape}")

    # Decode back to poses
    print("\nDecoding back to poses...")
    decoded = decode_dhb_dr(
        out["linear_motion_invariants"],
        out["angular_motion_invariants"],
        out["initial_pose"],
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    print(f"Decoded positions shape: {decoded['positions'].shape}")
    print(f"Decoded quaternions shape: {decoded['quaternions'].shape}")

    # Check reconstruction quality
    pos_error = np.mean(np.abs(positions - decoded["positions"]))
    quat_error = np.mean(np.abs(quaternions - decoded["quaternions"]))

    print("
Reconstruction errors:")
    print(".2e")
    print(".2e")

    if pos_error < 1e-10 and quat_error < 1e-10:
        print("✅ Perfect reconstruction (errors within numerical precision)")
    else:
        print("⚠️  Reconstruction has some error - check parameters")

    return out, decoded


def main():
    """Main function for command-line execution."""
    run_example()


if __name__ == "__main__":
    main()
