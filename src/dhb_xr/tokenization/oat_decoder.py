"""
Cross-attention decoder with prefix decoding support.

Decodes compressed register token latents back to the full invariant/action
sequence using a transformer decoder with cross-attention. Positional queries
attend to the register latents (after optional nested dropout).

Key features:
- Learned positional queries for each output timestep
- Cross-attention from queries to register latent tokens
- Supports prefix decoding: decode with k < K tokens for anytime inference
- Optional causal decoder mask for autoregressive generation

References:
- OAT (Liu et al., 2026): SinglePassDecoder, arXiv:2602.04215
"""

try:
    import math
    from typing import List, Optional

    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class OATDecoder(nn.Module):
        """
        Cross-attention decoder for register-token latents.

        Decodes (B, K, latent_dim) register latents -> (B, T, output_dim).

        Architecture:
        - Learned positional queries of length T (output timesteps)
        - nn.TransformerDecoder with cross-attention to register latents
        - Linear head to project to output dimension

        Works with MaskedNestedDropout: during training/eval, some trailing
        register tokens may be replaced with a mask token, enabling
        variable-quality prefix decoding.

        Args:
            output_dim: Dimension of output features (e.g., 8 for DHB invariants).
            output_horizon: Number of output timesteps T.
            emb_dim: Internal transformer embedding dimension.
            latent_dim: Dimension of input register latents.
            latent_horizon: Number of register tokens K.
            depth: Number of transformer decoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            use_causal_decoder: If True, apply causal mask to output queries
                                (for autoregressive generation).

        Example:
            >>> dec = OATDecoder(output_dim=8, output_horizon=50, emb_dim=64,
            ...                  latent_dim=16, latent_horizon=8, depth=4)
            >>> latents = torch.randn(2, 8, 16)
            >>> recon = dec(latents)  # (2, 50, 8)
            >>> # Prefix decoding: only use first 4 register tokens
            >>> recon_coarse = dec(latents, eval_keep_k=[4, 4])
        """

        def __init__(
            self,
            output_dim: int,
            output_horizon: int,
            emb_dim: int = 64,
            latent_dim: int = 16,
            latent_horizon: int = 8,
            depth: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            use_causal_decoder: bool = False,
        ):
            super().__init__()

            self.output_dim = output_dim
            self.output_horizon = output_horizon
            self.latent_horizon = latent_horizon
            self.emb_dim = emb_dim
            self.use_causal_decoder = use_causal_decoder

            # Learned positional queries for each output timestep
            self.query_pos = nn.Parameter(
                torch.randn(1, output_horizon, emb_dim)
            )
            nn.init.trunc_normal_(self.query_pos, std=0.02)

            # Project register latents to embedding dim
            self.latent_proj = nn.Linear(latent_dim, emb_dim)

            # Positional encoding for latent tokens
            self.latent_pos = nn.Parameter(
                torch.randn(1, latent_horizon, emb_dim)
            )
            nn.init.trunc_normal_(self.latent_pos, std=0.02)

            # Transformer decoder (cross-attention from queries to latents)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=4 * emb_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer, num_layers=depth
            )

            # Output projection
            self.head = nn.Linear(emb_dim, output_dim)

        def forward(
            self,
            latents: torch.Tensor,
            eval_keep_k: Optional[List[int]] = None,
        ) -> torch.Tensor:
            """
            Decode register latents to output sequence.

            Args:
                latents: (B, K, latent_dim) register token latents.
                         May have trailing tokens masked by MaskedNestedDropout.
                eval_keep_k: Optional list of B integers. If provided, only
                             the first keep_k latent tokens per sample are
                             considered "real" (rest are masked). This is
                             handled upstream by MaskedNestedDropout; this
                             parameter is kept for API consistency.

            Returns:
                (B, T, output_dim) reconstructed output sequence.
            """
            B = latents.shape[0]

            # Project latents and add positional encoding
            memory = self.latent_proj(latents)  # (B, K, emb_dim)
            memory = memory + self.latent_pos[:, : latents.shape[1], :]

            # Expand positional queries
            queries = self.query_pos.expand(B, -1, -1)  # (B, T, emb_dim)

            # Optionally apply causal mask on queries
            tgt_mask = None
            if self.use_causal_decoder:
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    self.output_horizon, device=latents.device
                )

            # Cross-attention decoding
            out = self.decoder(
                queries,
                memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=self.use_causal_decoder,
            )

            # Project to output dim
            return self.head(out)  # (B, T, output_dim)

else:
    OATDecoder = None
