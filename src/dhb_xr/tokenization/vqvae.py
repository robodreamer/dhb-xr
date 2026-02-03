"""VQ-VAE tokenizer for DHB invariant sequences (DHB-Token)."""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from dhb_xr.tokenization.causal_encoder import CausalConv1dEncoder
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:

    class VectorQuantizer(nn.Module):
        def __init__(self, num_embeddings: int, embedding_dim: int):
            super().__init__()
            self.embedding = nn.Embedding(num_embeddings, embedding_dim)
            self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        def forward(self, z: torch.Tensor) -> tuple:
            # z: (B, T, D)
            B, T, D = z.shape
            z_flat = z.reshape(-1, D)
            d = torch.cdist(z_flat, self.embedding.weight)
            indices = d.argmin(dim=1)
            z_q = self.embedding(indices).reshape(B, T, D)
            z_q_st = z + (z_q - z).detach()
            return indices.reshape(B, T), z_q_st, z_q

    class DHBTokenizer(nn.Module):
        """
        Causal VQ-VAE for invariant sequences.
        invariants (B, T, C) -> tokens (B, T), reconstructed (B, T, C).
        """

        def __init__(
            self,
            invariant_dim: int,
            latent_dim: int,
            codebook_size: int,
            num_layers: int = 2,
            kernel_size: int = 3,
        ):
            super().__init__()
            self.encoder = CausalConv1dEncoder(
                invariant_dim, latent_dim, latent_dim, num_layers, kernel_size
            )
            self.vq = VectorQuantizer(codebook_size, latent_dim)
            self.decoder = CausalConv1dEncoder(
                latent_dim, latent_dim, invariant_dim, num_layers, kernel_size
            )
            self.invariant_dim = invariant_dim
            self.latent_dim = latent_dim
            self.codebook_size = codebook_size

        def forward(self, invariants: torch.Tensor) -> tuple:
            z = self.encoder(invariants)
            indices, z_q_st, z_q = self.vq(z)
            reconstructed = self.decoder(z_q_st)
            return indices, reconstructed, z, z_q

        def loss(
            self,
            invariants: torch.Tensor,
            reconstructed: torch.Tensor,
            z: torch.Tensor,
            z_q: torch.Tensor,
            beta: float = 0.25,
        ) -> torch.Tensor:
            rec_loss = F.mse_loss(reconstructed, invariants)
            commitment = F.mse_loss(z, z_q)
            codebook = F.mse_loss(z_q, z.detach())
            return rec_loss + beta * commitment + codebook

        # ---- Flow matching integration API ----

        def encode_continuous(self, invariants: torch.Tensor) -> torch.Tensor:
            """
            Encode invariants to continuous latent space (before quantization).

            This is useful for flow matching which operates in continuous space.

            Args:
                invariants: Input invariant sequences (B, T, C).

            Returns:
                Continuous latent z (B, T, latent_dim).
            """
            return self.encoder(invariants)

        def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
            """
            Decode from continuous latent to invariants.

            Bypasses the VQ step, useful for flow matching generation.

            Args:
                z: Continuous latent (B, T, latent_dim).

            Returns:
                Reconstructed invariants (B, T, invariant_dim).
            """
            return self.decoder(z)

        def quantize(self, z: torch.Tensor) -> tuple:
            """
            Quantize continuous latent to discrete tokens.

            Args:
                z: Continuous latent (B, T, latent_dim).

            Returns:
                Tuple of (indices, z_q_st, z_q).
            """
            return self.vq(z)

        def get_codebook_embeddings(self) -> torch.Tensor:
            """
            Get the VQ codebook embeddings.

            Useful for flow matching in embedding space or visualization.

            Returns:
                Codebook embeddings (codebook_size, latent_dim).
            """
            return self.vq.embedding.weight.data

        def embed_tokens(self, indices: torch.Tensor) -> torch.Tensor:
            """
            Convert token indices to embeddings.

            Args:
                indices: Token indices (B, T).

            Returns:
                Token embeddings (B, T, latent_dim).
            """
            return self.vq.embedding(indices)

        def decode_tokens(self, indices: torch.Tensor) -> torch.Tensor:
            """
            Decode token indices to invariants.

            Args:
                indices: Token indices (B, T).

            Returns:
                Reconstructed invariants (B, T, invariant_dim).
            """
            z_q = self.embed_tokens(indices)
            return self.decoder(z_q)

else:
    DHBTokenizer = None
    VectorQuantizer = None
