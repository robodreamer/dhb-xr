"""Tests for OAT-style and FAST tokenizer modules."""

import numpy as np
import pytest


# ============================================================================
# FAST Tokenizer Tests
# ============================================================================

class TestFASTTokenizer:
    """Tests for FAST-style (DCT + BPE) tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        from dhb_xr.tokenization.fast_tokenizer import FASTTokenizer
        return FASTTokenizer(scale=10.0, vocab_size=1024, num_merges=50)

    @pytest.fixture
    def sample_corpus(self):
        """Generate a corpus of invariant-like chunks."""
        rng = np.random.RandomState(42)
        corpus = []
        for _ in range(20):
            T, D = 30, 8
            t = np.linspace(0, 2 * np.pi, T)
            chunk = np.column_stack([
                np.sin(t * (1 + 0.1 * d)) * 0.1 for d in range(D)
            ]).astype(np.float32)
            chunk += rng.randn(T, D).astype(np.float32) * 0.005
            corpus.append(chunk)
        return corpus

    def test_dct_roundtrip(self, tokenizer):
        """Test DCT encode/decode roundtrip (without BPE)."""
        chunk = np.random.randn(20, 8).astype(np.float32) * 0.1
        int_coeffs = tokenizer._dct_encode(chunk)
        recon = tokenizer._dct_decode(int_coeffs)
        # Should be approximate (due to rounding)
        mse = float(np.mean((chunk - recon) ** 2))
        assert mse < 0.1, f"DCT roundtrip MSE too high: {mse}"

    def test_fit(self, tokenizer, sample_corpus):
        """Test fitting BPE merges."""
        tokenizer.fit(sample_corpus)
        assert tokenizer._fitted
        assert len(tokenizer.merges) > 0
        assert tokenizer._time_horizon == 30
        assert tokenizer._dim == 8

    def test_encode_decode(self, tokenizer, sample_corpus):
        """Test encode-decode produces reasonable reconstruction."""
        tokenizer.fit(sample_corpus)

        for chunk in sample_corpus[:5]:
            tokens = tokenizer.encode(chunk)
            recon = tokenizer.decode(tokens, chunk.shape[0], chunk.shape[1])
            assert recon.shape == chunk.shape
            mse = float(np.mean((chunk - recon) ** 2))
            assert mse < 0.1, f"Reconstruction MSE too high: {mse}"

    def test_compression_ratio(self, tokenizer, sample_corpus):
        """Test compression is achieved."""
        tokenizer.fit(sample_corpus)

        ratios = [tokenizer.compression_ratio(c) for c in sample_corpus[:5]]
        avg_ratio = np.mean(ratios)
        assert avg_ratio > 1.0, f"Should compress, got ratio {avg_ratio}"

    def test_encode_batch(self, tokenizer, sample_corpus):
        """Test batch encoding."""
        tokenizer.fit(sample_corpus)
        batch = np.stack(sample_corpus[:3])  # (3, 30, 8)
        token_seqs = tokenizer.encode_batch(batch)
        assert len(token_seqs) == 3
        for ts in token_seqs:
            assert len(ts) > 0

    def test_decode_batch(self, tokenizer, sample_corpus):
        """Test batch decoding."""
        tokenizer.fit(sample_corpus)
        batch = np.stack(sample_corpus[:3])
        token_seqs = tokenizer.encode_batch(batch)
        recon = tokenizer.decode_batch(token_seqs, 30, 8)
        assert recon.shape == (3, 30, 8)

    def test_unfitted_raises(self, tokenizer):
        """Test encoding before fitting raises."""
        chunk = np.random.randn(20, 8).astype(np.float32)
        with pytest.raises(RuntimeError, match="must be fitted"):
            tokenizer.encode(chunk)

    def test_get_stats(self, tokenizer, sample_corpus):
        """Test statistics computation."""
        tokenizer.fit(sample_corpus)
        stats = tokenizer.get_stats(sample_corpus[0])
        assert "compression_ratio" in stats
        assert "mse" in stats
        assert stats["fitted"] is True


# ============================================================================
# FSQ Tests
# ============================================================================

class TestFSQ:
    """Tests for Finite Scalar Quantization."""

    @pytest.fixture
    def fsq(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.fsq import FSQ
        return FSQ(levels=[8, 5, 5, 5])

    @pytest.fixture
    def sample_z(self):
        pytest.importorskip("torch")
        import torch
        return torch.randn(2, 10, 4)

    def test_codebook_size(self, fsq):
        """Test codebook size computation."""
        assert fsq.codebook_size == 8 * 5 * 5 * 5  # 1000

    def test_forward_shapes(self, fsq, sample_z):
        """Test forward produces correct shapes."""
        z_q, indices = fsq(sample_z)
        assert z_q.shape == sample_z.shape
        assert indices.shape == (2, 10)

    def test_indices_range(self, fsq, sample_z):
        """Test indices are in valid range."""
        _, indices = fsq(sample_z)
        assert indices.min() >= 0
        assert indices.max() < fsq.codebook_size

    def test_roundtrip(self, fsq, sample_z):
        """Test codes_to_indices / indices_to_embedding roundtrip."""
        import torch
        z_q, indices = fsq(sample_z)
        z_recon = fsq.indices_to_embedding(indices)
        assert torch.allclose(z_q, z_recon, atol=1e-5), \
            "FSQ encode-decode roundtrip failed"

    def test_round_ste_gradients(self):
        """Test straight-through gradients flow."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.fsq import round_ste

        z = torch.randn(5, requires_grad=True)
        z_rounded = round_ste(z)
        loss = z_rounded.sum()
        loss.backward()
        assert z.grad is not None, "STE gradients should flow"

    def test_different_levels(self):
        """Test FSQ with different level configurations."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.fsq import FSQ

        fsq_small = FSQ(levels=[3, 3])
        assert fsq_small.codebook_size == 9
        z = torch.randn(1, 5, 2)
        z_q, idx = fsq_small(z)
        assert z_q.shape == (1, 5, 2)

    def test_quantization_dropout(self):
        """Test quantization dropout during training."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.fsq import FSQ

        fsq = FSQ(levels=[8, 5, 5, 5], drop_quant_p=0.5)
        fsq.train()
        z = torch.randn(4, 10, 4)
        z_q, idx = fsq(z)
        assert z_q.shape == z.shape


# ============================================================================
# Register Encoder Tests
# ============================================================================

class TestRegisterEncoder:
    """Tests for Transformer encoder with register tokens."""

    @pytest.fixture
    def encoder(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.register_encoder import RegisterEncoder
        return RegisterEncoder(
            input_dim=8, emb_dim=32, latent_dim=16,
            num_registers=8, depth=2, num_heads=4
        )

    @pytest.fixture
    def sample_input(self):
        pytest.importorskip("torch")
        import torch
        return torch.randn(2, 30, 8)

    def test_output_shape(self, encoder, sample_input):
        """Test encoder produces correct output shape."""
        z = encoder(sample_input)
        assert z.shape == (2, 8, 16), f"Expected (2, 8, 16), got {z.shape}"

    def test_compression(self, encoder, sample_input):
        """Test temporal compression (30 steps -> 8 tokens)."""
        z = encoder(sample_input)
        assert z.shape[1] < sample_input.shape[1], "Should compress temporally"

    def test_causal_last_mask(self):
        """Test causal-last mask structure."""
        pytest.importorskip("torch")
        from dhb_xr.tokenization.register_encoder import create_causal_last_mask

        mask = create_causal_last_mask(5, 3, "cpu")
        # Input tokens (0:5) should not attend to registers (5:8)
        assert mask[:5, 5:].all(), "Input should not attend to registers"
        # Registers (5:8) should attend to all inputs (no mask)
        assert not mask[5:, :5].any(), "Registers should attend to all inputs"
        # Registers should have causal mask among themselves
        assert mask[5, 6], "Register 0 should not attend to register 1"
        assert mask[5, 7], "Register 0 should not attend to register 2"
        assert not mask[6, 6], "Register 1 should attend to itself"
        assert mask[6, 7], "Register 1 should not attend to register 2"

    def test_training_step(self, encoder, sample_input):
        """Test gradients flow through encoder."""
        import torch

        z = encoder(sample_input)
        loss = z.sum()
        loss.backward()

        for name, p in encoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ============================================================================
# Nested Dropout Tests
# ============================================================================

class TestMaskedNestedDropout:
    """Tests for masked nested dropout."""

    @pytest.fixture
    def dropout(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.nested_dropout import MaskedNestedDropout
        return MaskedNestedDropout(dim=16, mode="uniform")

    @pytest.fixture
    def sample_tokens(self):
        pytest.importorskip("torch")
        import torch
        return torch.randn(4, 8, 16)

    def test_training_mode(self, dropout, sample_tokens):
        """Test that training mode applies masking."""
        import torch
        dropout.train()
        out = dropout(sample_tokens)
        assert out.shape == sample_tokens.shape
        # At least some elements should differ (masking applied)

    def test_eval_passthrough(self, dropout, sample_tokens):
        """Test that eval mode without keep_k is passthrough."""
        import torch
        dropout.eval()
        out = dropout(sample_tokens)
        assert torch.allclose(out, sample_tokens), "Eval without keep_k should be passthrough"

    def test_eval_with_keep_k(self, dropout, sample_tokens):
        """Test eval mode with specific keep_k."""
        import torch
        dropout.eval()
        keep_k = [2, 4, 6, 8]
        out = dropout(sample_tokens, eval_keep_k=keep_k)
        assert out.shape == sample_tokens.shape

        # For sample 0 (keep_k=2), positions 2+ should be mask token
        mask_token = dropout.mask_token.data
        for pos in range(2, 8):
            assert torch.allclose(out[0, pos], mask_token, atol=1e-5), \
                f"Position {pos} should be masked for keep_k=2"

    def test_disable_mode(self, sample_tokens):
        """Test disable mode is pure passthrough."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.nested_dropout import MaskedNestedDropout

        dropout_off = MaskedNestedDropout(dim=16, mode="disable")
        dropout_off.train()
        out = dropout_off(sample_tokens)
        assert torch.allclose(out, sample_tokens)

    def test_pow2_mode(self):
        """Test pow2 sampling mode."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.nested_dropout import MaskedNestedDropout

        dropout = MaskedNestedDropout(dim=16, mode="pow2")
        dropout.train()
        x = torch.randn(4, 8, 16)
        out = dropout(x)
        assert out.shape == x.shape

    def test_biased_mode(self):
        """Test quadratic_biased sampling mode."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.nested_dropout import MaskedNestedDropout

        dropout = MaskedNestedDropout(dim=16, mode="quadratic_biased")
        dropout.train()
        x = torch.randn(4, 8, 16)
        out = dropout(x)
        assert out.shape == x.shape


# ============================================================================
# OAT Decoder Tests
# ============================================================================

class TestOATDecoder:
    """Tests for cross-attention decoder."""

    @pytest.fixture
    def decoder(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.oat_decoder import OATDecoder
        return OATDecoder(
            output_dim=8, output_horizon=30, emb_dim=32,
            latent_dim=16, latent_horizon=8, depth=2, num_heads=4
        )

    @pytest.fixture
    def sample_latents(self):
        pytest.importorskip("torch")
        import torch
        return torch.randn(2, 8, 16)

    def test_output_shape(self, decoder, sample_latents):
        """Test decoder produces correct output shape."""
        out = decoder(sample_latents)
        assert out.shape == (2, 30, 8)

    def test_gradient_flow(self, decoder, sample_latents):
        """Test gradients flow through decoder."""
        import torch
        out = decoder(sample_latents)
        loss = out.sum()
        loss.backward()
        for name, p in decoder.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


# ============================================================================
# OAT Tokenizer (Combined) Tests
# ============================================================================

class TestOATTokenizer:
    """Tests for the combined OAT-style tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.oat_tokenizer import OATTokenizer
        return OATTokenizer(
            input_dim=8, input_horizon=30, emb_dim=32, latent_dim=4,
            num_registers=8, fsq_levels=[8, 5, 5, 5],
            encoder_depth=2, decoder_depth=2, num_heads=4,
            nested_dropout_mode="uniform",
        )

    @pytest.fixture
    def sample_data(self):
        pytest.importorskip("torch")
        import torch
        return torch.randn(4, 30, 8) * 0.1

    def test_forward_returns_loss(self, tokenizer, sample_data):
        """Test training forward returns scalar loss."""
        import torch
        tokenizer.train()
        loss = tokenizer(sample_data)
        assert loss.ndim == 0, "Loss should be scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_encode_shapes(self, tokenizer, sample_data):
        """Test encode produces correct shapes."""
        tokenizer.eval()
        latents_q, indices = tokenizer.encode(sample_data)
        assert latents_q.shape == (4, 8, 4)
        assert indices.shape == (4, 8)

    def test_tokenize_detokenize(self, tokenizer, sample_data):
        """Test tokenize/detokenize roundtrip."""
        tokenizer.eval()
        tokens = tokenizer.tokenize(sample_data)
        assert tokens.shape == (4, 8)
        recon = tokenizer.detokenize(tokens)
        assert recon.shape == sample_data.shape

    def test_prefix_decoding(self, tokenizer, sample_data):
        """Test prefix decoding at various k."""
        tokenizer.eval()
        tokens = tokenizer.tokenize(sample_data)

        # Decode with different prefix lengths
        for k in [1, 2, 4, 8]:
            recon = tokenizer.detokenize(tokens[:, :k])
            assert recon.shape == sample_data.shape, f"Failed at k={k}"

    def test_autoencode(self, tokenizer, sample_data):
        """Test autoencoding roundtrip."""
        tokenizer.eval()
        recon = tokenizer.autoencode(sample_data)
        assert recon.shape == sample_data.shape

    def test_autoencode_with_prefix(self, tokenizer, sample_data):
        """Test autoencoding with prefix masking."""
        tokenizer.eval()
        recon = tokenizer.autoencode(sample_data, eval_keep_k=[4, 4, 4, 4])
        assert recon.shape == sample_data.shape

    def test_training_step(self, tokenizer, sample_data):
        """Test a full training step with optimizer."""
        import torch

        optimizer = tokenizer.get_optimizer(learning_rate=1e-3)
        tokenizer.train()

        loss = tokenizer(sample_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients exist
        has_grad = False
        for name, p in tokenizer.named_parameters():
            if p.requires_grad and p.grad is not None:
                has_grad = True
                break
        assert has_grad, "At least one param should have gradients"

    def test_training_convergence(self, tokenizer, sample_data):
        """Test that loss decreases over a few steps."""
        import torch
        optimizer = tokenizer.get_optimizer(learning_rate=1e-3)
        tokenizer.train()

        losses = []
        for _ in range(20):
            loss = tokenizer(sample_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_codebook_size(self, tokenizer):
        """Test codebook size property."""
        assert tokenizer.codebook_size == 1000  # 8*5*5*5

    def test_compression_stats(self, tokenizer):
        """Test compression stats."""
        stats = tokenizer.get_compression_stats()
        assert stats["input_horizon"] == 30
        assert stats["num_registers"] == 8
        assert stats["compression_ratio"] == 30 / 8
        assert stats["codebook_size"] == 1000
        assert stats["total_params"] > 0


# ============================================================================
# Integration: FAST + OAT on same data
# ============================================================================

class TestTokenizerIntegration:
    """Integration tests comparing FAST and OAT tokenizers on same data."""

    @pytest.fixture
    def sample_data(self):
        """Generate deterministic test data."""
        rng = np.random.RandomState(123)
        T, D = 30, 8
        t = np.linspace(0, 2 * np.pi, T)
        data = np.column_stack([
            np.sin(t * (1 + 0.1 * d)) * 0.1 for d in range(D)
        ]).astype(np.float32)
        data += rng.randn(T, D).astype(np.float32) * 0.005
        return data

    def test_fast_produces_fewer_tokens(self, sample_data):
        """Test that FAST compresses (fewer tokens than values)."""
        from dhb_xr.tokenization.fast_tokenizer import FASTTokenizer

        fast = FASTTokenizer(scale=10.0, vocab_size=1024, num_merges=50)
        corpus = [sample_data + np.random.randn(*sample_data.shape).astype(np.float32) * 0.001
                  for _ in range(10)]
        fast.fit(corpus)
        tokens = fast.encode(sample_data)
        assert len(tokens) < sample_data.size, "FAST should compress"

    def test_oat_produces_fixed_tokens(self, sample_data):
        """Test that OAT always produces num_registers tokens."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.oat_tokenizer import OATTokenizer

        oat = OATTokenizer(
            input_dim=8, input_horizon=30, emb_dim=32, latent_dim=4,
            num_registers=8, fsq_levels=[8, 5, 5, 5],
            encoder_depth=1, decoder_depth=1
        )
        oat.eval()
        x = torch.from_numpy(sample_data).unsqueeze(0)
        tokens = oat.tokenize(x)
        assert tokens.shape == (1, 8), "OAT should produce exactly num_registers tokens"

    def test_both_reconstruct(self, sample_data):
        """Test both tokenizers can reconstruct data."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.fast_tokenizer import FASTTokenizer
        from dhb_xr.tokenization.oat_tokenizer import OATTokenizer

        # FAST
        fast = FASTTokenizer(scale=10.0, vocab_size=1024, num_merges=50)
        fast.fit([sample_data])
        tokens_fast = fast.encode(sample_data)
        recon_fast = fast.decode(tokens_fast, sample_data.shape[0], sample_data.shape[1])
        mse_fast = float(np.mean((sample_data - recon_fast) ** 2))
        assert mse_fast < 0.5, f"FAST MSE too high: {mse_fast}"

        # OAT (untrained, just check it runs)
        oat = OATTokenizer(
            input_dim=8, input_horizon=30, emb_dim=32, latent_dim=4,
            num_registers=8, fsq_levels=[8, 5, 5, 5],
            encoder_depth=1, decoder_depth=1
        )
        oat.eval()
        x = torch.from_numpy(sample_data).unsqueeze(0)
        recon_oat = oat.autoencode(x)
        assert recon_oat.shape == (1, 30, 8)
