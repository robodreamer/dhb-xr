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

        # ---- Flow matching integration API ----

        def encode_continuous(self, invariants: torch.Tensor) -> torch.Tensor:
            """
            Encode invariants to continuous latent space (before quantization).

            Args:
                invariants: Input invariant sequences (B, T, C).

            Returns:
                Continuous latent z (B, T, latent_dim).
            """
            return self.encoder(invariants)

        def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
            """
            Decode from continuous latent to invariants.

            Args:
                z: Continuous latent (B, T, latent_dim).

            Returns:
                Reconstructed invariants (B, T, invariant_dim).
            """
            return self.decoder(z)

        def quantize(self, z: torch.Tensor, num_codebooks: int = None) -> tuple:
            """
            Quantize continuous latent using RVQ.

            Args:
                z: Continuous latent (B, T, latent_dim).
                num_codebooks: Number of codebooks to use (default: all).

            Returns:
                Tuple of (all_indices, z_sum) where all_indices is list of (B, T).
            """
            if num_codebooks is None:
                num_codebooks = self.num_codebooks

            residuals = z
            all_indices = []
            z_sum = torch.zeros_like(z)

            for i, vq in enumerate(self.vqs[:num_codebooks]):
                indices, z_q_st, z_q = vq(residuals)
                all_indices.append(indices)
                z_sum = z_sum + z_q_st
                residuals = residuals - z_q.detach()

            return all_indices, z_sum

        def encode_partial(
            self,
            invariants: torch.Tensor,
            num_codebooks: int,
        ) -> tuple:
            """
            Encode with partial RVQ (for hierarchical VFM).

            Uses only the first num_codebooks codebooks.

            Args:
                invariants: Input invariant sequences (B, T, C).
                num_codebooks: Number of codebooks to use.

            Returns:
                Tuple of (all_indices, z_sum, reconstructed).
            """
            z = self.encoder(invariants)
            all_indices, z_sum = self.quantize(z, num_codebooks)
            reconstructed = self.decoder(z_sum)
            return all_indices, z_sum, reconstructed

        def get_codebook_embeddings(self, codebook_idx: int = 0) -> torch.Tensor:
            """
            Get codebook embeddings for a specific codebook.

            Args:
                codebook_idx: Index of the codebook (0 to num_codebooks-1).

            Returns:
                Codebook embeddings (codebook_size, latent_dim).
            """
            return self.vqs[codebook_idx].embedding.weight.data

        def get_all_codebook_embeddings(self) -> list:
            """
            Get all codebook embeddings.

            Returns:
                List of codebook embeddings, each (codebook_size, latent_dim).
            """
            return [vq.embedding.weight.data for vq in self.vqs]

        def embed_tokens(self, all_indices: list) -> torch.Tensor:
            """
            Convert RVQ token indices to summed embeddings.

            Args:
                all_indices: List of token indices, each (B, T).

            Returns:
                Summed embeddings (B, T, latent_dim).
            """
            z_sum = None
            for i, (indices, vq) in enumerate(zip(all_indices, self.vqs)):
                z_q = vq.embedding(indices)
                if z_sum is None:
                    z_sum = z_q
                else:
                    z_sum = z_sum + z_q
            return z_sum

        def decode_tokens(self, all_indices: list) -> torch.Tensor:
            """
            Decode RVQ token indices to invariants.

            Args:
                all_indices: List of token indices, each (B, T).

            Returns:
                Reconstructed invariants (B, T, invariant_dim).
            """
            z_sum = self.embed_tokens(all_indices)
            return self.decoder(z_sum)

else:
    ResidualVQTokenizer = None
