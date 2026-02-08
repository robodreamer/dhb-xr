"""
Tokenization and compression for DHB invariants (VLA).

Modules:
- vqvae: Basic VQ-VAE tokenizer (DHBTokenizer)
- rvq: Residual VQ for higher capacity (ResidualVQTokenizer)
- hierarchical: Multi-level hierarchical tokenization
- compression: BPE, entropy coding, RLE for token sequences
- fast_tokenizer: FAST-style DCT + BPE tokenizer (no PyTorch needed)
- fsq: Finite Scalar Quantization (no learned codebook)
- register_encoder: Transformer encoder with register tokens
- nested_dropout: Masked nested dropout for ordered token spaces
- oat_decoder: Cross-attention decoder with prefix support
- oat_tokenizer: Combined OAT-style tokenizer
"""

__all__ = [
    # Core tokenizers (VQ-based, require torch)
    "DHBTokenizer",
    "CausalConv1dEncoder",
    "ResidualVQTokenizer",
    # Hierarchical (require torch)
    "HierarchicalTokenizer",
    "ProgressiveTokenizer",
    # FAST-style (no torch needed)
    "FASTTokenizer",
    # OAT-style components (require torch)
    "FSQ",
    "RegisterEncoder",
    "MaskedNestedDropout",
    "OATDecoder",
    "OATTokenizer",
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

# FAST-style tokenizer (pure numpy/scipy, no torch needed)
try:
    from dhb_xr.tokenization.fast_tokenizer import FASTTokenizer
except ImportError:
    FASTTokenizer = None

# OAT-style components (require torch)
try:
    from dhb_xr.tokenization.fsq import FSQ
    from dhb_xr.tokenization.register_encoder import RegisterEncoder
    from dhb_xr.tokenization.nested_dropout import MaskedNestedDropout
    from dhb_xr.tokenization.oat_decoder import OATDecoder
    from dhb_xr.tokenization.oat_tokenizer import OATTokenizer
except ImportError:
    FSQ = None
    RegisterEncoder = None
    MaskedNestedDropout = None
    OATDecoder = None
    OATTokenizer = None

# Compression (pure Python, always available)
from dhb_xr.tokenization.compression import (
    BPECompressor,
    EntropyCompressor,
    RLECompressor,
    TokenCompressor,
    TokenReuser,
    compress_token_sequence,
)
