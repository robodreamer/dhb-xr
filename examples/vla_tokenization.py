"""
VLA Tokenization Example (DHB-Token)

This example demonstrates VQ-VAE tokenization of DHB invariants for discrete
action representations suitable for VLA (Vision-Language-Action) models.

DHB-Token converts continuous DHB invariants into discrete tokens:
1. Encode trajectory to DHB-DR invariants (8 values per timestep)
2. Use VQ-VAE to quantize to discrete tokens
3. Reconstruct invariants from tokens

Benefits for VLA:
- Discrete action space (like decision transformers)
- Compression: many timesteps → few tokens
- Autoregressive generation capabilities
- Robust to noise and perturbations

Requires: PyTorch (`pip install dhb_xr[gpu]`)
"""

import numpy as np


def run_example():
    """Run the VLA tokenization example."""
    print("DHB-XR VLA Tokenization (DHB-Token) Example")
    print("=" * 45)

    try:
        import torch
        from dhb_xr.encoder.dhb_dr import encode_dhb_dr
        from dhb_xr.core.types import DHBMethod, EncodingMethod
        from dhb_xr.tokenization.vqvae import DHBTokenizer
    except ImportError as e:
        print("❌ PyTorch required for tokenization")
        print(f"Install with: pip install dhb_xr[gpu]")
        print(f"Error: {e}")
        return None

    # Create synthetic trajectory
    np.random.seed(2)
    n = 40
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))

    print(f"Trajectory: {n} poses")
    print(f"Positions shape: {positions.shape}")

    # Encode to DHB invariants
    print("\nEncoding to DHB-DR invariants...")
    out = encode_dhb_dr(
        positions,
        quaternions,
        method=EncodingMethod.POSITION,
        use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )

    # Combine linear and angular invariants
    invariants = np.concatenate([
        out["linear_motion_invariants"],
        out["angular_motion_invariants"],
    ], axis=1).astype(np.float32)

    print(f"Combined invariants shape: {invariants.shape}")
    print(f"Values per timestep: {invariants.shape[1]} (4 linear + 4 angular)")

    # Convert to PyTorch tensor and add batch dimension
    inv_batch = torch.from_numpy(invariants).unsqueeze(0)  # (T, 8) -> (1, T, 8)
    print(f"Batch shape: {inv_batch.shape}")

    # Tokenize with VQ-VAE
    print("\nTokenizing with VQ-VAE...")
    model = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
    indices, reconstructed, z, z_q = model(inv_batch)

    print(f"✅ Tokenization complete")
    print(f"Token indices shape: {indices.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Calculate reconstruction quality
    rec_loss = torch.nn.functional.mse_loss(reconstructed, inv_batch)
    compression_ratio = invariants.size / indices.numel()

    print("
Results:")
    print(".4f")
    print(".1f")

    # Show some token examples
    print(f"\nSample tokens: {indices[0, :10].tolist()}")

    return {
        'invariants': invariants,
        'tokens': indices,
        'reconstructed': reconstructed,
        'loss': rec_loss.item()
    }


def main():
    """Main function for command-line execution."""
    run_example()


if __name__ == "__main__":
    main()
