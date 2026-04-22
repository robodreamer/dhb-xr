"""Tests for the minimal VLA token pipeline."""

from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from dhb_xr.integration.vla.pipeline import DHBVLAPipeline, DHBVLAPipelineConfig


def _sample_invariants(T: int = 12, D: int = 8) -> np.ndarray:
    rng = np.random.RandomState(7)
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    data = np.stack(
        [
            0.08 * np.sin((i + 1) * np.pi * t) + 0.01 * np.cos((i + 2) * np.pi * t)
            for i in range(D)
        ],
        axis=1,
    ).astype(np.float32)
    data += rng.randn(T, D).astype(np.float32) * 0.002
    return data


class TestDHBVLAPipeline:
    def test_oat_streaming_prefix_decode_matches_pipeline_contract(self):
        invariants = _sample_invariants(T=12, D=8)
        pipeline = DHBVLAPipeline(
            DHBVLAPipelineConfig(
                tokenizer_kind="oat",
                oat_num_registers=4,
                oat_emb_dim=32,
                oat_num_heads=4,
                oat_encoder_depth=1,
                oat_decoder_depth=1,
            )
        )

        tokens = pipeline.tokenize_invariants(invariants)
        assert tokens.shape == (4,)

        full_recon = pipeline.detokenize_tokens(tokens, invariant_dim=8, horizon=12)
        assert full_recon.shape == (1, 12, 8)
        assert np.isfinite(full_recon).all()

        # Exact streaming contract: a streamed prefix and an explicit keep_k on
        # the full token sequence must decode identically.
        for keep_k in [1, 2, 4]:
            recon_from_prefix = pipeline.detokenize_tokens(
                tokens[:keep_k],
                invariant_dim=8,
                horizon=12,
                prefix_keep_k=keep_k,
            )
            recon_from_full = pipeline.detokenize_tokens(
                tokens,
                invariant_dim=8,
                horizon=12,
                prefix_keep_k=keep_k,
            )
            assert recon_from_prefix.shape == (1, 12, 8)
            assert np.allclose(recon_from_prefix, recon_from_full)

        streamed = pipeline.decode_token_stream(
            tokens,
            invariant_dim=8,
            horizon=12,
            prefix_lengths=[1, 2, 4],
        )
        assert [item["keep_k"] for item in streamed] == [1, 2, 4]
        for item in streamed:
            recon = item["reconstruction"]
            assert recon.shape == (1, 12, 8)
            assert np.isfinite(recon).all()

        assert np.allclose(streamed[-1]["reconstruction"], full_recon)

    def test_vq_pipeline_rejects_prefix_streaming_decode(self):
        invariants = _sample_invariants(T=10, D=8)
        pipeline = DHBVLAPipeline(DHBVLAPipelineConfig(tokenizer_kind="vq"))

        tokens = pipeline.tokenize_invariants(invariants)
        assert tokens.shape == (10,)

        with pytest.raises(ValueError, match="prefix_keep_k"):
            pipeline.detokenize_tokens(
                tokens[:4],
                invariant_dim=8,
                horizon=10,
                prefix_keep_k=4,
            )

    def test_process_dataset_tracks_tokenizer_kind(self):
        positions = np.stack(
            [
                np.linspace(0.0, 0.1, 6, dtype=np.float32),
                np.linspace(0.0, -0.05, 6, dtype=np.float32),
                np.linspace(0.2, 0.25, 6, dtype=np.float32),
            ],
            axis=1,
        )
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (6, 1))

        pipeline = DHBVLAPipeline(
            DHBVLAPipelineConfig(
                tokenizer_kind="oat",
                oat_num_registers=4,
                oat_emb_dim=32,
                oat_num_heads=4,
                oat_encoder_depth=1,
                oat_decoder_depth=1,
            )
        )
        out = pipeline.process_dataset(
            [{"positions": positions, "quaternions": quaternions, "metadata": {"episode": 3}}]
        )
        assert len(out) == 1
        assert out[0]["tokenizer_kind"] == "oat"
        assert out[0]["tokens"].shape == (4,)
        assert out[0]["invariants"].shape[1] == 8

    def test_oat_pipeline_handles_variable_horizons_without_stale_cache(self):
        pipeline = DHBVLAPipeline(
            DHBVLAPipelineConfig(
                tokenizer_kind="oat",
                oat_num_registers=4,
                oat_emb_dim=32,
                oat_num_heads=4,
                oat_encoder_depth=1,
                oat_decoder_depth=1,
            )
        )

        inv_a = _sample_invariants(T=12, D=8)
        inv_b = _sample_invariants(T=9, D=8)

        tokens_a = pipeline.tokenize_invariants(inv_a)
        recon_a = pipeline.detokenize_tokens(tokens_a, invariant_dim=8, horizon=12)
        assert recon_a.shape == (1, 12, 8)

        tokens_b = pipeline.tokenize_invariants(inv_b)
        recon_b = pipeline.detokenize_tokens(tokens_b, invariant_dim=8, horizon=9)
        assert recon_b.shape == (1, 9, 8)

    def test_decode_token_stream_to_trajectories_returns_executable_shapes(self):
        T = 12
        positions = np.stack(
            [
                np.linspace(0.0, 0.08, T, dtype=np.float32),
                np.linspace(0.0, -0.03, T, dtype=np.float32),
                np.linspace(0.2, 0.26, T, dtype=np.float32),
            ],
            axis=1,
        )
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (T, 1))

        pipeline = DHBVLAPipeline(
            DHBVLAPipelineConfig(
                tokenizer_kind="oat",
                oat_num_registers=4,
                oat_emb_dim=32,
                oat_num_heads=4,
                oat_encoder_depth=1,
                oat_decoder_depth=1,
            )
        )
        encoded = pipeline.encode_trajectory(positions, quaternions)
        invariants = np.concatenate(
            [encoded["linear_motion_invariants"], encoded["angular_motion_invariants"]],
            axis=1,
        ).astype(np.float32)
        tokens = pipeline.tokenize_invariants(invariants)

        streamed = pipeline.decode_token_stream_to_trajectories(
            tokens,
            invariant_dim=8,
            horizon=invariants.shape[0],
            initial_pose=encoded["initial_pose"],
            prefix_lengths=[1, 2, 4],
            return_diagnostics=True,
        )
        assert [item["keep_k"] for item in streamed] == [1, 2, 4]
        for item in streamed:
            assert item["positions"].shape[1] == 3
            assert item["quaternions"].shape[1] == 4
            assert np.isfinite(item["positions"]).all()
            assert np.isfinite(item["quaternions"]).all()
            assert item["invariants"].shape == invariants.shape
            assert "diagnostics" in item
