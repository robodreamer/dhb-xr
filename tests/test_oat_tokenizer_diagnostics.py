"""Diagnostics for OAT invariant-token collapse."""

import pytest

torch = pytest.importorskip("torch")

from dhb_xr.tokenization.oat_tokenizer import OATTokenizer


def test_token_stats_detects_collapsed_tokens():
    tokens = torch.full((3, 4), 7, dtype=torch.long)

    stats = OATTokenizer.token_stats(tokens)

    assert stats["avg_token_unique_count"] == 1.0
    assert stats["global_unique_count"] == 1
    assert stats["per_position_unique_counts"] == [1, 1, 1, 1]
    assert stats["collapsed_sample_fraction"] == 1.0
    assert stats["token_entropy"] == 0.0
    assert stats["token_histogram"] == {"7": 12}


def test_token_stats_reports_position_diversity():
    tokens = torch.tensor(
        [
            [1, 1, 2, 3],
            [1, 2, 2, 4],
            [1, 3, 5, 5],
        ],
        dtype=torch.long,
    )

    stats = OATTokenizer.token_stats(tokens)

    assert stats["global_unique_count"] == 5
    assert stats["per_position_unique_counts"] == [1, 3, 2, 3]
    assert stats["collapsed_sample_fraction"] == 0.0
    assert stats["token_entropy"] > 0.0


def test_register_diversity_loss_is_near_zero_for_orthogonal_registers():
    tokenizer = OATTokenizer(
        input_dim=8,
        input_horizon=4,
        emb_dim=16,
        latent_dim=4,
        num_registers=4,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=4,
        dropout=0.0,
    )
    latents = torch.eye(4).unsqueeze(0)

    loss = tokenizer.register_diversity_loss(latents=latents)

    assert loss.item() < 1e-7


def test_reconstruction_metrics_include_prefix_and_token_stats():
    tokenizer = OATTokenizer(
        input_dim=8,
        input_horizon=4,
        emb_dim=16,
        latent_dim=4,
        num_registers=4,
        encoder_depth=1,
        decoder_depth=1,
        num_heads=4,
        dropout=0.0,
    )
    x = torch.randn(2, 4, 8)

    metrics = tokenizer.reconstruction_metrics(x, prefix_ks=[1, 2, 4])

    assert metrics["full_recon_mae"] >= 0.0
    assert metrics["full_recon_mse"] >= 0.0
    assert set(metrics["prefix"].keys()) == {"1", "2", "4"}
    assert "token_stats" in metrics
    assert metrics["token_stats"]["global_unique_count"] >= 1
