"""Tests for tokenization (VQ-VAE)."""

import numpy as np
import pytest

pytest.importorskip("torch")
from dhb_xr.tokenization.vqvae import DHBTokenizer


def test_dhb_tokenizer_forward():
    B, T, C = 2, 20, 8
    inv = np.random.randn(B, T, C).astype(np.float32) * 0.1
    model = DHBTokenizer(invariant_dim=C, latent_dim=16, codebook_size=64)
    import torch
    x = torch.from_numpy(inv)
    indices, reconstructed, z, z_q = model(x)
    assert indices.shape == (B, T)
    assert reconstructed.shape == (B, T, C)
