"""Residual VQ (RVQ) tokenizer for higher capacity."""

try:
    import torch
    import torch.nn as nn
    from dhb_xr.tokenization.vqvae import VectorQuantizer
    from dhb_xr.tokenization.causal_encoder import CausalConv1dEncoder
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:

    class ResidualVQTokenizer(nn.Module):
        """RVQ: multiple codebooks on residual. invariants (B, T, C) -> list of (B, T) tokens, (B, T, C) reconstructed."""

        def __init__(
            self,
            invariant_dim: int,
            latent_dim: int,
            codebook_size: int,
            num_codebooks: int = 2,
            num_layers: int = 2,
        ):
            super().__init__()
            self.encoder = CausalConv1dEncoder(
                invariant_dim, latent_dim, latent_dim, num_layers
            )
            self.vqs = nn.ModuleList([
                VectorQuantizer(codebook_size, latent_dim) for _ in range(num_codebooks)
            ])
            self.decoder = CausalConv1dEncoder(
                latent_dim, latent_dim, invariant_dim, num_layers
            )
            self.num_codebooks = num_codebooks

        def forward(self, invariants: torch.Tensor) -> tuple:
            z = self.encoder(invariants)
            residuals = z
            all_indices = []
            z_sum = torch.zeros_like(z)
            for vq in self.vqs:
                indices, z_q_st, z_q = vq(residuals)
                all_indices.append(indices)
                z_sum = z_sum + z_q_st
                residuals = residuals - z_q.detach()
            reconstructed = self.decoder(z_sum)
            return all_indices, reconstructed, z, z_sum

else:
    ResidualVQTokenizer = None
