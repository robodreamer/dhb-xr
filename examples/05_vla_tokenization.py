"""VQ-VAE tokenization of DHB invariants (DHB-Token) for VLA-style action representation."""

import numpy as np

def main():
    try:
        import torch
        from dhb_xr.encoder.dhb_dr import encode_dhb_dr
        from dhb_xr.core.types import DHBMethod
        from dhb_xr.tokenization.vqvae import DHBTokenizer
    except ImportError as e:
        print("Skipping: torch required for tokenization.", e)
        return

    np.random.seed(2)
    n = 40
    positions = np.cumsum(np.random.randn(n, 3) * 0.01, axis=0)
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    out = encode_dhb_dr(
        positions, quaternions,
        method="pos", use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    invariants = np.concatenate([
        out["linear_motion_invariants"],
        out["angular_motion_invariants"],
    ], axis=1).astype(np.float32)
    # (T, 8) -> (1, T, 8)
    inv_batch = torch.from_numpy(invariants).unsqueeze(0)

    model = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
    indices, reconstructed, z, z_q = model(inv_batch)
    print("Invariants shape:", inv_batch.shape)
    print("Token indices shape:", indices.shape)
    print("Reconstructed shape:", reconstructed.shape)
    rec_loss = torch.nn.functional.mse_loss(reconstructed, inv_batch)
    print("Reconstruction MSE:", rec_loss.item())

if __name__ == "__main__":
    main()
