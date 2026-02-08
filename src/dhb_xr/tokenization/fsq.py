"""
Finite Scalar Quantization (FSQ) for DHB invariant tokenization.

FSQ replaces learned VQ codebooks with deterministic rounding in a bounded space.
Unlike standard VQ-VAE, FSQ has no codebook collapse, no commitment/codebook losses,
and the codebook is implicitly defined by the level specification.

Key idea:
    Each latent dimension is independently bounded (via tanh) and rounded to one
    of L_i discrete levels. The effective codebook size is prod(L_1, ..., L_d).

References:
- Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple", arXiv:2309.15505
- OAT (Liu et al., 2026): Uses FSQ for ordered action tokenization, arXiv:2602.04215
"""

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
    from torch.amp import autocast
    from typing import List, Optional, Tuple

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    def round_ste(z: Tensor) -> Tensor:
        """Round with straight-through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach()

    class FSQ(nn.Module):
        """
        Finite Scalar Quantization.

        Maps continuous latents to discrete codes by bounding (tanh) and rounding.
        No learned codebook -- the codebook is implicitly defined by the levels.

        Example:
            >>> fsq = FSQ(levels=[8, 5, 5, 5])
            >>> # Effective codebook size: 8 * 5 * 5 * 5 = 1000
            >>> z = torch.randn(2, 10, 4)  # (B, T, D) where D = len(levels)
            >>> z_q, indices = fsq(z)
            >>> # z_q: (2, 10, 4) quantized, indices: (2, 10) codebook indices
            >>> z_recon = fsq.indices_to_embedding(indices)
            >>> assert torch.allclose(z_q, z_recon)

        Args:
            levels: List of quantization levels per dimension.
                    E.g., [8, 5, 5, 5] -> codebook size 1000.
            drop_quant_p: During training, probability of skipping quantization
                          per sample (quantization dropout for regularization).
        """

        def __init__(
            self,
            levels: List[int],
            drop_quant_p: float = 0.0,
        ):
            super().__init__()

            _levels = torch.tensor(levels, dtype=torch.int32)
            self.register_buffer("_levels", _levels, persistent=False)

            _basis = torch.cumprod(
                torch.tensor([1] + levels[:-1], dtype=torch.int32), dim=0
            )
            self.register_buffer("_basis", _basis, persistent=False)

            self.dim = len(levels)
            self.codebook_size = int(_levels.prod().item())
            self.drop_quant_p = drop_quant_p

            # Build implicit codebook for lookup
            implicit_codebook = self.indices_to_embedding(
                torch.arange(self.codebook_size)
            )
            self.register_buffer(
                "implicit_codebook", implicit_codebook, persistent=False
            )

        def __repr__(self) -> str:
            return (
                f"FSQ(levels={self._levels.tolist()}, "
                f"codebook_size={self.codebook_size}, "
                f"drop_quant_p={self.drop_quant_p})"
            )

        @property
        def latent_dim(self) -> int:
            """Dimension of the latent space (= number of FSQ levels)."""
            return self.dim

        def bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
            """Bound z to the valid range for each level via tanh.

            Args:
                z: (..., D) unbounded latent vectors.

            Returns:
                (..., D) bounded latent vectors in [-half_l, half_l].
            """
            half_l = (self._levels - 1) * (1 + eps) / 2
            offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
            shift = (offset / half_l).atanh()
            return (z + shift).tanh() * half_l - offset

        def quantize(self, z: Tensor) -> Tensor:
            """Quantize z: bound, round (STE), normalize to [-1, 1].

            Args:
                z: (..., D) unbounded latent vectors.

            Returns:
                (..., D) quantized and normalized vectors.
            """
            bounded = self.bound(z)

            if self.training and self.drop_quant_p > 0.0:
                # Quantization dropout: skip quantization for some samples
                zhat = round_ste(bounded)
                B = z.shape[0]
                mask = torch.bernoulli(
                    torch.full((B,), self.drop_quant_p, device=z.device)
                )
                mask = mask.view(B, *([1] * (z.ndim - 1)))
                # mask=1 -> keep original (skip quant), mask=0 -> use quantized
                quantized = bounded * mask + zhat * (1 - mask)
            else:
                quantized = round_ste(bounded)

            # Normalize to [-1, 1]
            half_width = self._levels // 2
            return quantized / half_width

        def _scale_and_shift(self, zhat_normalized: Tensor) -> Tensor:
            """Convert normalized codes to integer codes."""
            half_width = self._levels // 2
            return (zhat_normalized * half_width) + half_width

        def _scale_and_shift_inverse(self, zhat: Tensor) -> Tensor:
            """Convert integer codes to normalized codes."""
            half_width = self._levels // 2
            return (zhat - half_width) / half_width

        def codes_to_indices(self, zhat: Tensor) -> Tensor:
            """Convert quantized codes to flat codebook indices.

            Args:
                zhat: (..., D) quantized codes (normalized).

            Returns:
                (...) integer indices.
            """
            assert zhat.shape[-1] == self.dim
            zhat_int = self._scale_and_shift(zhat)
            return (zhat_int * self._basis).sum(dim=-1).to(torch.int64)

        def indices_to_embedding(self, indices: Tensor) -> Tensor:
            """Convert flat codebook indices to normalized code vectors.

            Args:
                indices: (...) integer indices.

            Returns:
                (..., D) normalized code vectors.
            """
            indices = indices.unsqueeze(-1)
            codes_non_centered = (indices // self._basis) % self._levels
            return self._scale_and_shift_inverse(codes_non_centered.float())

        @autocast(device_type="cuda", enabled=False)
        def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
            """Quantize and return codes + indices.

            Args:
                z: (..., D) continuous latent vectors.

            Returns:
                Tuple of:
                - z_q: (..., D) quantized vectors (normalized to [-1, 1]).
                - indices: (...) flat codebook indices.
            """
            assert z.shape[-1] == self.dim, (
                f"Expected last dim {self.dim}, got {z.shape[-1]}"
            )
            z_q = self.quantize(z.float())
            indices = self.codes_to_indices(z_q)
            return z_q, indices.long()

else:
    FSQ = None
    round_ste = None
