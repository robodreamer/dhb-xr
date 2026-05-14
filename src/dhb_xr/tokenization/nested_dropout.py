"""
Masked Nested Dropout for ordered token spaces.

Implements Matryoshka-style nested dropout that forces a coarse-to-fine
ordering on latent token sequences. During training, a random number of
trailing tokens are replaced with a learnable mask token, ensuring that
any prefix of tokens can produce a valid (coarser) reconstruction.

Key properties:
- Any prefix t_{1:k} is decode-worthy (anytime decoding)
- Early tokens capture global structure, later tokens refine details
- Naturally compatible with autoregressive next-token prediction

References:
- OAT (Liu et al., 2026): Nested dropout for ordered action tokenization
- Matryoshka Representation Learning (Kusupati et al., 2022)
"""

try:
    import torch
    import torch.nn as nn
    from torch.nn.init import trunc_normal_
    from typing import List, Literal, Optional

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class MaskedNestedDropout(nn.Module):
        """
        Nested dropout module that replaces trailing tokens with a learnable
        mask token during training.

        During training:
            Randomly sample keep_k in [1, N], replace tokens beyond keep_k
            with a learnable mask token.

        During evaluation:
            If eval_keep_k is provided, mask tokens beyond keep_k.
            Otherwise, pass through without masking (use all tokens).

        Args:
            dim: Embedding dimension of the mask token.
            mode: Sampling strategy for keep_k during training.
                - "disable": No dropout (pass-through).
                - "uniform": Uniform probability across all prefix lengths.
                - "pow2": Only sample power-of-2 prefix lengths.
                - "linear_biased": Linear bias toward longer prefixes.
                - "quadratic_biased": Quadratic bias toward longer prefixes.
                - "cubic_biased": Cubic bias toward longer prefixes.

        Example:
            >>> dropout = MaskedNestedDropout(dim=64, mode="uniform")
            >>> x = torch.randn(2, 8, 64)  # (B, K, D)
            >>> # Training: some trailing tokens replaced with mask
            >>> dropout.train()
            >>> y = dropout(x)
            >>> # Eval with prefix: decode with first 4 tokens only
            >>> dropout.eval()
            >>> y = dropout(x, eval_keep_k=[4, 4])
        """

        def __init__(
            self,
            dim: int,
            mode: str = "uniform",
        ):
            super().__init__()
            self.dim = dim
            self.mode = mode

            if self.mode != "disable":
                self.mask_token = nn.Parameter(
                    torch.randn(dim), requires_grad=True
                )
                trunc_normal_(self.mask_token, std=0.02)

        def _sample_keep_k(
            self,
            batch_size: int,
            num_tokens: int,
            device: torch.device,
        ) -> torch.Tensor:
            """Sample the number of tokens to keep per batch element.

            Args:
                batch_size: Batch size B.
                num_tokens: Total number of tokens N.
                device: Device for tensor creation.

            Returns:
                (B,) tensor of keep_k values in [1, N].
            """
            if self.mode == "uniform":
                return torch.randint(
                    1, num_tokens + 1, (batch_size,), device=device
                )

            elif self.mode == "pow2":
                # Power-of-2 values up to num_tokens
                pow2_vals = []
                v = 1
                while v <= num_tokens:
                    pow2_vals.append(v)
                    v *= 2
                pow2_vals = torch.tensor(pow2_vals, device=device)
                idx = torch.randint(0, len(pow2_vals), (batch_size,), device=device)
                return pow2_vals[idx]

            elif self.mode.endswith("_biased"):
                power_map = {
                    "linear_biased": 1.0,
                    "quadratic_biased": 2.0,
                    "cubic_biased": 3.0,
                }
                power = power_map.get(self.mode, 1.0)
                weights = (
                    torch.arange(1, num_tokens + 1, dtype=torch.float32, device=device)
                    ** power
                )
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, batch_size, replacement=True)
                return indices + 1  # 0-indexed -> 1-indexed

            else:
                raise ValueError(f"Unknown sampling mode: {self.mode}")

        def forward(
            self,
            x: torch.Tensor,
            eval_keep_k: Optional[List[int]] = None,
        ) -> torch.Tensor:
            """
            Apply nested dropout.

            Args:
                x: (B, N, D) token sequence.
                eval_keep_k: Optional list of B integers specifying how many
                             tokens to keep per sample during evaluation.
                             If None during eval, all tokens are kept.

            Returns:
                (B, N, D) token sequence with trailing tokens masked.
            """
            if self.mode == "disable":
                return x

            B, N, D = x.shape
            x = x.clone()  # Don't modify input in-place

            if self.training:
                keep_ks = self._sample_keep_k(B, N, x.device)
                # Create mask: True for positions that should be masked
                positions = torch.arange(N, device=x.device).unsqueeze(0)  # (1, N)
                mask = positions >= keep_ks.unsqueeze(1)  # (B, N)
                x[mask] = self.mask_token.to(device=x.device, dtype=x.dtype)
            elif eval_keep_k is not None:
                keep_ks = torch.tensor(eval_keep_k, device=x.device)
                positions = torch.arange(N, device=x.device).unsqueeze(0)
                mask = positions >= keep_ks.unsqueeze(1)
                x[mask] = self.mask_token.to(device=x.device, dtype=x.dtype)

            return x

else:
    MaskedNestedDropout = None
