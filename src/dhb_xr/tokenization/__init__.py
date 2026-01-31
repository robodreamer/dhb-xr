"""
VQ-VAE / RVQ tokenization and compression for DHB invariants (VLA).

Modules:
- vqvae: Basic VQ-VAE tokenizer (DHBTokenizer)
- rvq: Residual VQ for higher capacity (ResidualVQTokenizer)
- hierarchical: Multi-level hierarchical tokenization
- compression: BPE, entropy coding, RLE for token sequences
"""

__all__ = [
    # Core tokenizers
    "DHBTokenizer",
    "CausalConv1dEncoder",
    "ResidualVQTokenizer",
    # Hierarchical
    "HierarchicalTokenizer",
    "ProgressiveTokenizer",
    # Compression
    "BPECompressor",
    "EntropyCompressor",
    "RLECompressor",
    "TokenCompressor",
    "TokenReuser",
    "compress_token_sequence",
]

# Core tokenizers (require torch)
try:
    from dhb_xr.tokenization.vqvae import DHBTokenizer
    from dhb_xr.tokenization.causal_encoder import CausalConv1dEncoder
    from dhb_xr.tokenization.rvq import ResidualVQTokenizer
except ImportError:
    DHBTokenizer = None
    CausalConv1dEncoder = None
    ResidualVQTokenizer = None

# Hierarchical tokenizers (require torch)
try:
    from dhb_xr.tokenization.hierarchical import (
        HierarchicalTokenizer,
        ProgressiveTokenizer,
    )
except ImportError:
    HierarchicalTokenizer = None
    ProgressiveTokenizer = None

# Compression (pure Python, always available)
from dhb_xr.tokenization.compression import (
    BPECompressor,
    EntropyCompressor,
    RLECompressor,
    TokenCompressor,
    TokenReuser,
    compress_token_sequence,
)
