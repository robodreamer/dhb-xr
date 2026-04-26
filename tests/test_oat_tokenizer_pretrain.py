"""Tests for standalone OAT tokenizer pretrain/eval helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "starvla_dhb_pretrain_oat_tokenizer.py"
SPEC = importlib.util.spec_from_file_location("starvla_dhb_pretrain_oat_tokenizer", SCRIPT_PATH)
pretrain = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pretrain)


def test_stack_dhb_invariants_concatenates_linear_and_angular():
    sample = {
        "dhb.linear_invariants": np.ones((3, 4), dtype=np.float16),
        "dhb.angular_invariants": np.full((3, 4), 2.0, dtype=np.float16),
    }

    chunk = pretrain.stack_dhb_invariants(sample)

    assert chunk.shape == (3, 8)
    assert chunk.dtype == np.float32
    np.testing.assert_allclose(chunk[:, :4], 1.0)
    np.testing.assert_allclose(chunk[:, 4:], 2.0)


def test_split_by_holdout_is_deterministic():
    chunks = [np.full((2, 1), i, dtype=np.float32) for i in range(6)]

    train, eval_ = pretrain.split_by_holdout(chunks, holdout_mod=3, holdout_fold=1)

    assert [int(c[0, 0]) for c in train] == [0, 2, 3, 5]
    assert [int(c[0, 0]) for c in eval_] == [1, 4]


def test_pad_or_trim_chunks_uses_final_row_for_padding():
    chunks = [
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([[3.0], [4.0], [5.0], [6.0]], dtype=np.float32),
    ]

    stacked = pretrain.pad_or_trim_chunks(chunks, horizon=3)

    np.testing.assert_allclose(stacked[:, :, 0], [[1.0, 2.0, 2.0], [3.0, 4.0, 5.0]])


def test_reconstruction_metrics_reports_zero_for_exact_match():
    target = np.arange(12, dtype=np.float32).reshape(2, 3, 2)

    metrics = pretrain.reconstruction_metrics(target, target.copy())

    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_token_usage_metrics_reports_histogram_and_collapse():
    tokens = np.array(
        [
            [1, 1, 1, 1],
            [1, 2, 1, 2],
            [3, 4, 5, 6],
        ],
        dtype=np.int64,
    )

    metrics = pretrain.token_usage_metrics(tokens, codebook_size=10)

    assert metrics["unique_token_count"] == 6
    assert metrics["unique_token_fraction"] == pytest.approx(0.6)
    assert metrics["per_position_unique_counts"] == [2, 3, 2, 3]
    assert metrics["collapsed_sample_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["token_histogram"]["1"] == 6


def test_maybe_normalize_roundtrips_eval_array():
    train = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    eval_ = train + 10.0

    _, eval_norm, stats = pretrain.maybe_normalize(train, eval_, enabled=True)
    restored = pretrain.unnormalize(eval_norm, stats)

    np.testing.assert_allclose(restored, eval_, rtol=1e-6, atol=1e-6)


def test_evaluate_gate_requires_finite_non_collapsed_tokens():
    metrics = {
        "reconstruction": {"mse": 0.1, "r2": 0.2},
        "token_usage": {
            "collapsed_sample_fraction": 0.2,
            "unique_token_count": 8,
            "effective_num_tokens": 4.0,
            "per_position_unique_counts": [2, 3, 2],
        },
    }
    thresholds = pretrain.gate_thresholds(
        max_collapsed_fraction=0.25,
        min_unique_tokens=8,
        min_effective_tokens=4.0,
        min_per_position_unique=2,
        min_reconstruction_r2=0.0,
    )

    passed, reasons = pretrain.evaluate_gate(metrics, thresholds)

    assert passed
    assert reasons == []


def test_evaluate_gate_reports_collapse_failures():
    metrics = {
        "reconstruction": {"mse": 0.1, "r2": -0.1},
        "token_usage": {
            "collapsed_sample_fraction": 0.5,
            "unique_token_count": 1,
            "effective_num_tokens": 1.0,
            "per_position_unique_counts": [1, 1, 1],
        },
    }
    thresholds = pretrain.gate_thresholds(
        max_collapsed_fraction=0.25,
        min_unique_tokens=2,
        min_effective_tokens=4.0,
        min_per_position_unique=2,
        min_reconstruction_r2=0.0,
    )

    passed, reasons = pretrain.evaluate_gate(metrics, thresholds)

    assert not passed
    assert any("collapsed_sample_fraction" in reason for reason in reasons)
    assert any("unique_token_count" in reason for reason in reasons)
    assert any("effective_num_tokens" in reason for reason in reasons)
    assert any("min_per_position_unique" in reason for reason in reasons)
    assert any("reconstruction_r2" in reason for reason in reasons)


def test_compute_register_diversity_loss_uses_optional_tokenizer_method():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def register_diversity_loss(self, batch):
            return batch.sum() * 0.0 + 3.0

    batch = torch.ones(2, 3, 4)
    weighted, raw, available = pretrain.compute_register_diversity_loss(
        FakeTokenizer(),
        batch,
        weight=0.5,
    )

    assert available
    assert raw.item() == pytest.approx(3.0)
    assert weighted.item() == pytest.approx(1.5)


def test_compute_register_diversity_loss_zero_when_unavailable():
    torch = pytest.importorskip("torch")
    batch = torch.ones(2, 3, 4)

    weighted, raw, available = pretrain.compute_register_diversity_loss(
        object(),
        batch,
        weight=0.5,
    )

    assert not available
    assert raw.item() == 0.0
    assert weighted.item() == 0.0
