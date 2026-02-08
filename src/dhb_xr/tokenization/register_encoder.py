"""
Transformer encoder with register tokens for action/invariant compression.

Compresses a sequence of T invariant timesteps into K << T register tokens
using a transformer with a causal-last attention mask. This achieves massive
temporal compression while preserving the essential trajectory structure.

Key ideas from OAT (Liu et al., 2026, arXiv:2602.04215):
- Learnable register tokens are appended to the input sequence
- A causal-last attention mask ensures:
  * Input tokens attend to all input tokens (bidirectional)
  * Register tokens attend to all input tokens + causally to previous registers
- Only register tokens are extracted as output -> (B, K, D)
- Combined with causal attention on registers, this induces a natural
  coarse-to-fine ordering compatible with autoregressive generation
"""

try:
    import math
    from functools import lru_cache

    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    @lru_cache(maxsize=32)
    def create_causal_last_mask(
        input_len: int,
        register_len: int,
        device: str,
    ) -> torch.Tensor:
        """
        Create causal-last attention mask for [input, registers].

        Attention pattern:
        - Input tokens attend to all input tokens (no masking)
        - Input tokens do NOT attend to register tokens
        - Register tokens attend to all input tokens (full cross-attention)
        - Register tokens attend causally to previous registers

        Args:
            input_len: Length of input sequence.
            register_len: Number of register tokens.
            device: Device string for tensor creation.

        Returns:
            Boolean mask of shape (total_len, total_len).
            True = masked (not attended to), False = attended to.
            Compatible with nn.MultiheadAttention attn_mask convention.
        """
        total_len = input_len + register_len

        # Start with no masking (all False = all attended)
        mask = torch.zeros(
            (total_len, total_len), dtype=torch.bool, device=device
        )

        # Input tokens cannot attend to register tokens
        mask[:input_len, input_len:] = True

        # Register tokens attend causally to other registers
        # (upper triangle = future registers are masked)
        causal_reg = torch.triu(
            torch.ones(
                (register_len, register_len), dtype=torch.bool, device=device
            ),
            diagonal=1,
        )
        mask[input_len:, input_len:] = causal_reg

        return mask

    class SinusoidalPositionalEncoding(nn.Module):
        """Sinusoidal positional encoding (fixed, not learned)."""

        def __init__(self, d_model: int, max_len: int = 512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float()
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
            self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Add positional encoding. x: (B, T, D)."""
            return x + self.pe[:, : x.size(1), :]

    class RegisterEncoder(nn.Module):
        """
        Transformer encoder with register tokens.

        Compresses (B, T, input_dim) -> (B, num_registers, latent_dim).

        The key architectural innovation: learnable register tokens are appended
        to the embedded input, processed through a transformer with a causal-last
        attention mask, and then extracted as the compressed representation.

        Args:
            input_dim: Dimension of input features (e.g., 8 for DHB invariants).
            emb_dim: Internal transformer embedding dimension.
            latent_dim: Output dimension per register token.
            num_registers: Number of register tokens (K). Controls compression.
            depth: Number of transformer encoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            max_seq_len: Maximum input sequence length.

        Example:
            >>> enc = RegisterEncoder(input_dim=8, emb_dim=64, latent_dim=16,
            ...                       num_registers=8, depth=4)
            >>> x = torch.randn(2, 50, 8)  # (B, T=50, D=8)
            >>> z = enc(x)                   # (B, 8, 16) -- 50 timesteps -> 8 tokens
        """

        def __init__(
            self,
            input_dim: int,
            emb_dim: int = 64,
            latent_dim: int = 16,
            num_registers: int = 8,
            depth: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            max_seq_len: int = 512,
        ):
            super().__init__()

            self.input_dim = input_dim
            self.emb_dim = emb_dim
            self.latent_dim = latent_dim
            self.num_registers = num_registers

            # Input projection
            self.input_proj = nn.Linear(input_dim, emb_dim)

            # Positional encoding for input tokens
            self.pos_enc = SinusoidalPositionalEncoding(emb_dim, max_seq_len)

            # Learnable register tokens
            self.registers = nn.Parameter(torch.randn(num_registers, emb_dim))
            nn.init.trunc_normal_(self.registers, std=0.02)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=num_heads,
                dim_feedforward=4 * emb_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=depth
            )

            # Output projection (register embeddings -> latent dim)
            self.head = nn.Linear(emb_dim, latent_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Encode input sequence to register token latents.

            Args:
                x: (B, T, input_dim) input invariant/action sequence.

            Returns:
                (B, num_registers, latent_dim) compressed representation.
            """
            B, T, _ = x.shape

            # Project input and add positional encoding
            x_emb = self.input_proj(x)  # (B, T, emb_dim)
            x_emb = self.pos_enc(x_emb)

            # Append register tokens
            reg = self.registers.unsqueeze(0).expand(B, -1, -1)
            combined = torch.cat([x_emb, reg], dim=1)  # (B, T+K, emb_dim)

            # Create causal-last mask
            mask = create_causal_last_mask(T, self.num_registers, str(x.device))

            # Transformer forward
            out = self.transformer(combined, mask=mask)

            # Extract register tokens and project
            latents = self.head(out[:, T:])  # (B, K, latent_dim)

            return latents

else:
    RegisterEncoder = None
    SinusoidalPositionalEncoding = None
    create_causal_last_mask = None
