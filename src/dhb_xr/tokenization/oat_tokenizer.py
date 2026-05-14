"""
OAT-style tokenizer: Register Encoder + FSQ + Nested Dropout + Cross-Attention Decoder.

Combines all OAT-inspired components into a single learned tokenizer that
produces an ordered, compressed, discrete representation of invariant/action
sequences with anytime prefix decodability.

Pipeline:
    invariants (B, T, D) -> RegisterEncoder -> (B, K, latent_dim)
                          -> FSQ quantizer   -> (B, K, latent_dim) + indices (B, K)
                          -> NestedDropout   -> (B, K, latent_dim) [masked]
                          -> OATDecoder      -> (B, T, D) reconstructed

Key properties:
- Massive compression: T timesteps -> K tokens (K << T)
- No codebook collapse (FSQ)
- Ordered token space (nested dropout)
- Anytime decoding (prefix t_{1:k} produces valid coarser reconstruction)
- Training loss: MSE reconstruction only

References:
- OAT (Liu et al., 2026): arXiv:2602.04215
- FSQ (Mentzer et al., 2024): arXiv:2309.15505
"""

try:
    import math
    from typing import Dict, List, Optional, Sequence, Tuple

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from dhb_xr.tokenization.register_encoder import RegisterEncoder
    from dhb_xr.tokenization.fsq import FSQ
    from dhb_xr.tokenization.nested_dropout import MaskedNestedDropout
    from dhb_xr.tokenization.oat_decoder import OATDecoder

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class OATTokenizer(nn.Module):
        """
        OAT-style tokenizer with register encoding, FSQ, nested dropout, and
        cross-attention decoding.

        Produces an ordered sequence of K discrete tokens from T input timesteps.
        Any prefix of k <= K tokens can be decoded to a valid reconstruction.

        Args:
            input_dim: Dimension of input features (e.g., 8 for DHB invariants).
            input_horizon: Number of input timesteps T.
            emb_dim: Internal transformer embedding dimension.
            latent_dim: FSQ latent dimension (= number of FSQ levels).
            num_registers: Number of register tokens K (compression factor).
            fsq_levels: List of FSQ quantization levels per dimension.
            encoder_depth: Number of transformer encoder layers.
            decoder_depth: Number of transformer decoder layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            nested_dropout_mode: Sampling mode for nested dropout.
            drop_quant_p: FSQ quantization dropout probability.
            use_causal_decoder: Use causal mask in decoder queries.

        Example:
            >>> tok = OATTokenizer(
            ...     input_dim=8, input_horizon=50, emb_dim=64, latent_dim=4,
            ...     num_registers=8, fsq_levels=[8, 5, 5, 5],
            ...     encoder_depth=4, decoder_depth=4
            ... )
            >>> x = torch.randn(2, 50, 8)
            >>> loss = tok(x)
            >>> print(f"Training loss: {loss.item():.4f}")
            >>> # Tokenize
            >>> tokens = tok.tokenize(x)  # (2, 8)
            >>> # Detokenize with prefix
            >>> recon = tok.detokenize(tokens[:, :4])  # use first 4 tokens
        """

        def __init__(
            self,
            input_dim: int = 8,
            input_horizon: int = 50,
            emb_dim: int = 64,
            latent_dim: int = 4,
            num_registers: int = 8,
            fsq_levels: Optional[List[int]] = None,
            encoder_depth: int = 4,
            decoder_depth: int = 4,
            num_heads: int = 4,
            dropout: float = 0.1,
            nested_dropout_mode: str = "uniform",
            drop_quant_p: float = 0.0,
            use_causal_decoder: bool = False,
        ):
            super().__init__()

            if fsq_levels is None:
                fsq_levels = [8, 5, 5, 5]  # Default: codebook size 1000

            assert len(fsq_levels) == latent_dim, (
                f"len(fsq_levels)={len(fsq_levels)} must equal latent_dim={latent_dim}"
            )

            self.input_dim = input_dim
            self.input_horizon = input_horizon
            self.num_registers = num_registers

            # Encoder: compress T timesteps -> K register tokens
            self.encoder = RegisterEncoder(
                input_dim=input_dim,
                emb_dim=emb_dim,
                latent_dim=latent_dim,
                num_registers=num_registers,
                depth=encoder_depth,
                num_heads=num_heads,
                dropout=dropout,
                max_seq_len=input_horizon + 64,
            )

            # Quantizer: FSQ (no learned codebook)
            self.quantizer = FSQ(
                levels=fsq_levels,
                drop_quant_p=drop_quant_p,
            )

            # Nested dropout: force coarse-to-fine ordering
            self.nested_dropout = MaskedNestedDropout(
                dim=latent_dim,
                mode=nested_dropout_mode,
            )

            # Decoder: cross-attention from positional queries to register latents
            self.decoder = OATDecoder(
                output_dim=input_dim,
                output_horizon=input_horizon,
                emb_dim=emb_dim,
                latent_dim=latent_dim,
                latent_horizon=num_registers,
                depth=decoder_depth,
                num_heads=num_heads,
                dropout=dropout,
                use_causal_decoder=use_causal_decoder,
            )

        @property
        def codebook_size(self) -> int:
            """Effective codebook size from FSQ levels."""
            return self.quantizer.codebook_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Training forward pass: encode -> quantize -> dropout -> decode -> loss.

            Args:
                x: (B, T, input_dim) input invariant/action sequence.

            Returns:
                Scalar MSE reconstruction loss.
            """
            # Encode to register latents
            latents = self.encoder(x)  # (B, K, latent_dim)

            # FSQ quantization
            latents_q, _indices = self.quantizer(latents)

            # Nested dropout (forces ordering during training)
            latents_dropped = self.nested_dropout(latents_q)

            # Decode
            recon = self.decoder(latents_dropped)  # (B, T, input_dim)

            # MSE loss (no VQ losses needed with FSQ)
            loss = F.mse_loss(recon, x)
            return loss

        @staticmethod
        def token_stats(tokens: torch.Tensor) -> Dict:
            """Summarize discrete-token usage for collapse diagnostics.

            Args:
                tokens: Integer token IDs with shape (B, K).

            Returns:
                JSON-serializable statistics. A collapsed tokenizer has
                ``global_unique_count == 1`` and ``collapsed_sample_fraction``
                close to 1.0 even if reconstruction loss is low.
            """
            if tokens.ndim != 2:
                raise ValueError(f"Expected tokens with shape (B, K), got {tuple(tokens.shape)}")
            tokens_cpu = tokens.detach().to(device="cpu", dtype=torch.long)
            B, K = tokens_cpu.shape
            if B == 0 or K == 0:
                return {
                    "avg_token_unique_count": 0.0,
                    "global_unique_count": 0,
                    "per_position_unique_counts": [],
                    "collapsed_sample_fraction": 0.0,
                    "token_entropy": 0.0,
                    "token_histogram": {},
                }

            per_sample_unique = [
                int(torch.unique(tokens_cpu[row]).numel())
                for row in range(B)
            ]
            per_position_unique = [
                int(torch.unique(tokens_cpu[:, col]).numel())
                for col in range(K)
            ]
            values, counts = torch.unique(tokens_cpu.reshape(-1), return_counts=True)
            probs = counts.float() / counts.sum().clamp_min(1)
            entropy = float(-(probs * probs.log()).sum().item())
            histogram = {
                str(int(value.item())): int(count.item())
                for value, count in zip(values, counts)
            }
            return {
                "avg_token_unique_count": float(sum(per_sample_unique) / len(per_sample_unique)),
                "global_unique_count": int(values.numel()),
                "per_position_unique_counts": per_position_unique,
                "collapsed_sample_fraction": float(
                    sum(1 for count in per_sample_unique if count <= 1) / len(per_sample_unique)
                ),
                "token_entropy": entropy,
                "token_entropy_normalized": float(
                    entropy / math.log(max(2, int(values.numel())))
                    if values.numel() > 1
                    else 0.0
                ),
                "token_histogram": histogram,
            }

        def register_diversity_loss(
            self,
            x: Optional[torch.Tensor] = None,
            *,
            latents: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Penalize identical register latents before FSQ quantization.

            This is intentionally independent from token counts so it remains
            differentiable during standalone tokenizer pretraining.
            """
            if latents is None:
                if x is None:
                    raise ValueError("register_diversity_loss requires x or latents")
                latents = self.encoder(x)
            if latents.ndim != 3:
                raise ValueError(f"Expected latents with shape (B, K, D), got {tuple(latents.shape)}")
            if latents.shape[1] <= 1:
                return latents.new_zeros(())

            normalized = F.normalize(latents.float(), dim=-1)
            sim = torch.matmul(normalized, normalized.transpose(1, 2))
            K = sim.shape[-1]
            offdiag = ~torch.eye(K, dtype=torch.bool, device=sim.device)
            return sim[:, offdiag].pow(2).mean()

        @torch.no_grad()
        def reconstruction_metrics(
            self,
            x: torch.Tensor,
            *,
            prefix_ks: Optional[Sequence[int]] = None,
        ) -> Dict:
            """Measure full and prefix token reconstruction on a batch.

            The model is evaluated without changing its original training mode.
            Metrics are returned as Python floats so scripts can write JSON
            directly.
            """
            was_training = self.training
            self.eval()
            try:
                latents_q, tokens = self.encode(x)
                full_recon = self.decode(latents_q)
                full_error = (full_recon - x).abs()
                metrics = {
                    "full_recon_mae": float(full_error.mean().item()),
                    "full_recon_mse": float(full_error.pow(2).mean().item()),
                    "token_stats": self.token_stats(tokens),
                    "prefix": {},
                }

                if prefix_ks is None:
                    prefix_ks = [1, 2, 4, self.num_registers]
                for keep_k in prefix_ks:
                    keep_k_int = int(keep_k)
                    if keep_k_int < 1 or keep_k_int > self.num_registers:
                        continue
                    recon = self.decode(
                        latents_q,
                        eval_keep_k=[keep_k_int] * x.shape[0],
                    )
                    error = (recon - x).abs()
                    metrics["prefix"][str(keep_k_int)] = {
                        "mae": float(error.mean().item()),
                        "mse": float(error.pow(2).mean().item()),
                    }
                return metrics
            finally:
                if was_training:
                    self.train()

        def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Encode input to quantized latents and token indices.

            Args:
                x: (B, T, input_dim) input sequence.

            Returns:
                Tuple of:
                - latents_q: (B, K, latent_dim) quantized register latents.
                - indices: (B, K) discrete token indices.
            """
            latents = self.encoder(x)
            latents_q, indices = self.quantizer(latents)
            return latents_q, indices

        def decode(
            self,
            latents: torch.Tensor,
            eval_keep_k: Optional[List[int]] = None,
        ) -> torch.Tensor:
            """
            Decode quantized latents (with optional prefix masking).

            Args:
                latents: (B, K, latent_dim) quantized register latents.
                eval_keep_k: Optional list of B integers for prefix decoding.

            Returns:
                (B, T, input_dim) reconstructed sequence.
            """
            # Apply nested dropout for prefix masking
            latents_masked = self.nested_dropout(latents, eval_keep_k=eval_keep_k)
            return self.decoder(latents_masked)

        def tokenize(self, x: torch.Tensor) -> torch.Tensor:
            """
            Convert input to discrete token indices.

            Args:
                x: (B, T, input_dim) input sequence.

            Returns:
                (B, K) token indices.
            """
            _, indices = self.encode(x)
            return indices

        def detokenize(
            self,
            tokens: torch.Tensor,
            eval_keep_k: Optional[List[int]] = None,
        ) -> torch.Tensor:
            """
            Convert token indices back to reconstructed sequence.

            Supports prefix decoding: pass a subset of tokens (first k columns)
            to get a coarser reconstruction.

            Args:
                tokens: (B, K) or (B, k) token indices where k <= K.
                eval_keep_k: Optional list specifying keep_k per sample.
                             If tokens has fewer columns than num_registers,
                             keep_k is inferred automatically.

            Returns:
                (B, T, input_dim) reconstructed sequence.
            """
            B, K_actual = tokens.shape

            # Pad tokens to full register length if needed
            if K_actual < self.num_registers:
                pad = torch.zeros(
                    B,
                    self.num_registers - K_actual,
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                tokens_full = torch.cat([tokens, pad], dim=1)
                if eval_keep_k is None:
                    eval_keep_k = [K_actual] * B
            else:
                tokens_full = tokens

            # Look up embeddings
            latents = self.quantizer.indices_to_embedding(tokens_full)

            return self.decode(latents, eval_keep_k=eval_keep_k)

        def autoencode(
            self,
            x: torch.Tensor,
            eval_keep_k: Optional[List[int]] = None,
        ) -> torch.Tensor:
            """
            Full encode-decode roundtrip.

            Args:
                x: (B, T, input_dim) input sequence.
                eval_keep_k: Optional prefix lengths for anytime decoding.

            Returns:
                (B, T, input_dim) reconstructed sequence.
            """
            latents_q, _ = self.encode(x)
            return self.decode(latents_q, eval_keep_k=eval_keep_k)

        def get_optimizer(
            self,
            learning_rate: float = 1e-4,
            weight_decay: float = 0.01,
            betas: Tuple[float, float] = (0.9, 0.999),
        ) -> torch.optim.Optimizer:
            """Create AdamW optimizer with proper weight decay groups."""
            decay_params = [
                p
                for n, p in self.named_parameters()
                if p.requires_grad and p.dim() >= 2
            ]
            nodecay_params = [
                p
                for n, p in self.named_parameters()
                if p.requires_grad and p.dim() < 2
            ]
            optim_groups = [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ]
            return torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas
            )

        def get_compression_stats(self) -> Dict:
            """Get compression statistics."""
            return {
                "input_horizon": self.input_horizon,
                "input_dim": self.input_dim,
                "num_registers": self.num_registers,
                "compression_ratio": self.input_horizon / self.num_registers,
                "codebook_size": self.codebook_size,
                "fsq_levels": self.quantizer._levels.tolist(),
                "total_params": sum(
                    p.numel() for p in self.parameters()
                ),
            }

else:
    OATTokenizer = None
