# Tokenization

::: dhb_xr.tokenization

## Overview

VQ-VAE-based tokenization for discrete DHB invariant representations.

## Main Classes

### DHBTokenizer

::: dhb_xr.tokenization.vqvae.DHBTokenizer

## Usage Example

```python
import torch
from dhb_xr.tokenization.vqvae import DHBTokenizer

# Create tokenizer
tokenizer = DHBTokenizer(
    invariant_dim=8,
    latent_dim=16,
    codebook_size=64
)

# Tokenize invariants
invariants = torch.randn(10, 20, 8)  # (batch, time, features)
tokens = tokenizer.encode(invariants)

# Decode back
reconstructed = tokenizer.decode(tokens)
```