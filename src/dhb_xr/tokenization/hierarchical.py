"""
Hierarchical tokenization for variable-rate DHB compression.

Extends RVQ with multi-level structure for lossy compression:
- Coarse tokens: Global trajectory structure
- Fine tokens: Local details and refinements
- Configurable depth for compression/quality tradeoff

Inspired by:
- BEAST (NeurIPS 2025): B-spline encoded action sequences
- VQ-VLA: Multi-level vector quantization
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from dhb_xr.tokenization.vqvae import VectorQuantizer
    from dhb_xr.tokenization.causal_encoder import CausalConv1dEncoder
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class HierarchicalTokenizer(nn.Module):
        """
        Hierarchical RVQ with variable-rate output.
        
        Provides coarse-to-fine tokenization:
        - Level 0: Low-frequency global structure (high compression)
        - Level 1-N: Residual details (configurable refinement)
        
        For inference, can truncate to fewer levels for faster/coarser output.
        
        Example:
            >>> tokenizer = HierarchicalTokenizer(
            ...     invariant_dim=8, latent_dim=32, codebook_size=256, num_levels=4
            ... )
            >>> tokens, recon = tokenizer(invariants)
            >>> 
            >>> # Coarse only (4x fewer tokens)
            >>> tokens_coarse, recon_coarse = tokenizer(invariants, max_level=1)
        """
        
        def __init__(
            self,
            invariant_dim: int,
            latent_dim: int,
            codebook_size: int,
            num_levels: int = 4,
            temporal_downsample: int = 2,
            num_layers: int = 2,
        ):
            """
            Args:
                invariant_dim: DHB invariant dimension (typically 8)
                latent_dim: Latent embedding dimension
                codebook_size: VQ codebook size per level
                num_levels: Number of hierarchy levels
                temporal_downsample: Downsample factor between levels
                num_layers: Conv layers per encoder/decoder
            """
            super().__init__()
            
            self.invariant_dim = invariant_dim
            self.latent_dim = latent_dim
            self.codebook_size = codebook_size
            self.num_levels = num_levels
            self.temporal_downsample = temporal_downsample
            
            # Per-level encoders (progressively downsample)
            self.encoders = nn.ModuleList()
            self.vqs = nn.ModuleList()
            self.decoders = nn.ModuleList()
            
            for level in range(num_levels):
                # Encoder: downsample temporally at each level
                if level == 0:
                    enc = CausalConv1dEncoder(
                        invariant_dim, latent_dim, latent_dim, num_layers
                    )
                else:
                    enc = nn.Sequential(
                        CausalConv1dEncoder(latent_dim, latent_dim, latent_dim, num_layers),
                        TemporalDownsample(temporal_downsample),
                    )
                self.encoders.append(enc)
                
                # VQ at each level
                self.vqs.append(VectorQuantizer(codebook_size, latent_dim))
                
                # Decoder: upsample to match previous level
                if level == 0:
                    dec = CausalConv1dEncoder(
                        latent_dim, latent_dim, invariant_dim, num_layers
                    )
                else:
                    dec = nn.Sequential(
                        TemporalUpsample(temporal_downsample),
                        CausalConv1dEncoder(latent_dim, latent_dim, latent_dim, num_layers),
                    )
                self.decoders.append(dec)
            
            # Final projection back to invariant space
            self.final_proj = nn.Linear(latent_dim, invariant_dim)
        
        def forward(
            self, 
            invariants: torch.Tensor, 
            max_level: int = None,
            return_all_levels: bool = False,
        ) -> tuple:
            """
            Hierarchical encoding and decoding.
            
            Args:
                invariants: (B, T, invariant_dim) input
                max_level: Stop at this level (None = all levels)
                return_all_levels: Return tokens/recon at each level
                
            Returns:
                all_tokens: List of (B, T_l) tokens per level
                reconstructed: (B, T, invariant_dim) reconstruction
                level_info: Optional dict with per-level details
            """
            B, T, C = invariants.shape
            max_level = max_level or self.num_levels
            
            all_tokens = []
            all_z = []
            all_z_q = []
            level_info = {}
            
            # Encode through hierarchy
            x = invariants
            z_residual = None
            for level in range(max_level):
                if level == 0:
                    encoder_input = x
                else:
                    assert z_residual is not None
                    encoder_input = z_residual
                z = self.encoders[level](encoder_input)
                indices, z_q_st, z_q = self.vqs[level](z)
                
                all_tokens.append(indices)
                all_z.append(z)
                all_z_q.append(z_q)
                
                if level < max_level - 1:
                    z_residual = z - z_q.detach()
                
                level_info[f"level_{level}"] = {
                    "shape": tuple(z.shape),
                    "tokens": indices.shape[-1],
                }
            
            # Decode through hierarchy (reverse order)
            reconstructed = torch.zeros_like(invariants)
            for level in reversed(range(max_level)):
                dec_out = self.decoders[level](all_z_q[level])
                
                # Match temporal dimension
                if dec_out.shape[1] > reconstructed.shape[1]:
                    dec_out = dec_out[:, :reconstructed.shape[1], :]
                elif dec_out.shape[1] < reconstructed.shape[1]:
                    # Upsample to match
                    dec_out = F.interpolate(
                        dec_out.transpose(1, 2), 
                        size=reconstructed.shape[1],
                        mode='linear',
                        align_corners=True
                    ).transpose(1, 2)
                
                if level == 0:
                    reconstructed = dec_out
                else:
                    reconstructed = reconstructed + self.final_proj(dec_out)
            
            if return_all_levels:
                return all_tokens, reconstructed, level_info
            return all_tokens, reconstructed
        
        def loss(
            self,
            invariants: torch.Tensor,
            reconstructed: torch.Tensor,
            all_z: list,
            all_z_q: list,
            beta: float = 0.25,
            level_weights: list = None,
        ) -> torch.Tensor:
            """
            Compute hierarchical loss with per-level weighting.
            
            Args:
                invariants: Original input
                reconstructed: Reconstruction
                all_z: Latents at each level
                all_z_q: Quantized latents at each level
                beta: Commitment loss weight
                level_weights: Optional weights per level (default: exponential decay)
                
            Returns:
                Total loss
            """
            # Reconstruction loss
            rec_loss = F.mse_loss(reconstructed, invariants)
            
            # Per-level VQ losses
            if level_weights is None:
                level_weights = [0.5 ** i for i in range(len(all_z))]
            
            commitment = 0
            codebook = 0
            for i, (z, z_q) in enumerate(zip(all_z, all_z_q)):
                commitment += level_weights[i] * F.mse_loss(z, z_q.detach())
                codebook += level_weights[i] * F.mse_loss(z_q, z.detach())
            
            return rec_loss + beta * commitment + codebook
        
        def get_compression_stats(self, T: int, max_level: int = None) -> dict:
            """
            Compute compression statistics.
            
            Args:
                T: Original sequence length
                max_level: Number of levels to use
                
            Returns:
                Compression statistics
            """
            max_level = max_level or self.num_levels
            
            total_tokens = 0
            for level in range(max_level):
                level_T = T // (self.temporal_downsample ** level)
                total_tokens += level_T
            
            original_values = T * self.invariant_dim
            token_values = total_tokens  # Each token is one index
            
            return {
                "original_values": original_values,
                "total_tokens": total_tokens,
                "tokens_per_level": [T // (self.temporal_downsample ** l) for l in range(max_level)],
                "compression_ratio": original_values / token_values if token_values > 0 else 1,
                "bits_per_value": (total_tokens * np.log2(self.codebook_size)) / original_values,
            }


    class TemporalDownsample(nn.Module):
        """Temporal downsampling via strided convolution."""
        
        def __init__(self, factor: int = 2):
            super().__init__()
            self.factor = factor
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, D)
            return x[:, ::self.factor, :]


    class TemporalUpsample(nn.Module):
        """Temporal upsampling via interpolation."""
        
        def __init__(self, factor: int = 2):
            super().__init__()
            self.factor = factor
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, D)
            B, T, D = x.shape
            x = x.transpose(1, 2)  # (B, D, T)
            x = F.interpolate(x, size=T * self.factor, mode='linear', align_corners=True)
            return x.transpose(1, 2)  # (B, T*factor, D)


    class ProgressiveTokenizer(nn.Module):
        """
        Progressive refinement tokenizer.
        
        Outputs can be truncated at any level for variable-rate decoding:
        - 1 level: ~4x compression, coarse motion
        - 2 levels: ~2x compression, medium detail
        - 4 levels: ~1x compression, full fidelity
        
        Ideal for streaming/bandwidth-adaptive applications.
        """
        
        def __init__(
            self,
            invariant_dim: int,
            latent_dim: int,
            codebook_size: int,
            num_refinements: int = 3,
        ):
            super().__init__()
            
            self.base_tokenizer = nn.Sequential(
                CausalConv1dEncoder(invariant_dim, latent_dim, latent_dim, 2),
            )
            self.base_vq = VectorQuantizer(codebook_size, latent_dim)
            self.base_decoder = CausalConv1dEncoder(latent_dim, latent_dim, invariant_dim, 2)
            
            # Refinement stages
            self.refinements = nn.ModuleList()
            self.refine_vqs = nn.ModuleList()
            self.refine_decoders = nn.ModuleList()
            
            for _ in range(num_refinements):
                self.refinements.append(
                    CausalConv1dEncoder(invariant_dim, latent_dim, latent_dim, 1)
                )
                self.refine_vqs.append(VectorQuantizer(codebook_size, latent_dim))
                self.refine_decoders.append(
                    CausalConv1dEncoder(latent_dim, latent_dim, invariant_dim, 1)
                )
        
        def forward(self, invariants: torch.Tensor, num_refine: int = None) -> tuple:
            """
            Progressive tokenization.
            
            Args:
                invariants: (B, T, D) input
                num_refine: Number of refinement levels (0 = base only)
                
            Returns:
                all_tokens: List of token tensors
                reconstructed: Final reconstruction
            """
            if num_refine is None:
                num_refine = len(self.refinements)
            
            # Base encoding
            z_base = self.base_tokenizer(invariants)
            tokens_base, z_q_st, z_q = self.base_vq(z_base)
            recon = self.base_decoder(z_q_st)
            
            all_tokens = [tokens_base]
            
            # Progressive refinements
            residual = invariants - recon
            for i in range(min(num_refine, len(self.refinements))):
                z_ref = self.refinements[i](residual)
                tokens_ref, z_ref_st, z_ref_q = self.refine_vqs[i](z_ref)
                
                all_tokens.append(tokens_ref)
                
                recon = recon + self.refine_decoders[i](z_ref_st)
                residual = invariants - recon
            
            return all_tokens, recon

else:
    HierarchicalTokenizer = None
    ProgressiveTokenizer = None
    TemporalDownsample = None
    TemporalUpsample = None


# Import numpy for compression stats (needed even without torch)
import numpy as np
