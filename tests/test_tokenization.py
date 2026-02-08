"""Tests for tokenization (VQ-VAE, RVQ, FAST, OAT) and compression modules."""

import numpy as np
import pytest


# ============================================================================
# VQ-VAE Tokenizer Tests
# ============================================================================

class TestDHBTokenizer:
    """Tests for basic VQ-VAE tokenizer."""

    @pytest.fixture
    def tokenizer(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.vqvae import DHBTokenizer
        return DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)

    @pytest.fixture
    def sample_invariants(self):
        pytest.importorskip("torch")
        import torch
        B, T, C = 2, 20, 8
        return torch.randn(B, T, C) * 0.1

    def test_forward_shape(self, tokenizer, sample_invariants):
        """Test forward pass produces correct shapes."""
        indices, reconstructed, z, z_q = tokenizer(sample_invariants)
        B, T, C = sample_invariants.shape

        assert indices.shape == (B, T), f"Expected indices shape {(B, T)}, got {indices.shape}"
        assert reconstructed.shape == (B, T, C), f"Expected recon shape {(B, T, C)}, got {reconstructed.shape}"
        assert z.shape == z_q.shape, "Latent and quantized should have same shape"

    def test_indices_in_range(self, tokenizer, sample_invariants):
        """Test token indices are within codebook range."""
        indices, _, _, _ = tokenizer(sample_invariants)

        assert indices.min() >= 0, "Token indices should be non-negative"
        assert indices.max() < tokenizer.codebook_size, "Token indices should be < codebook size"

    def test_loss_computation(self, tokenizer, sample_invariants):
        """Test loss computation."""
        indices, reconstructed, z, z_q = tokenizer(sample_invariants)
        loss = tokenizer.loss(sample_invariants, reconstructed, z, z_q)

        assert loss.ndim == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert not np.isnan(loss.item()), "Loss should not be NaN"

    def test_training_step(self, tokenizer, sample_invariants):
        """Test a single training step."""
        import torch

        optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)

        indices, reconstructed, z, z_q = tokenizer(sample_invariants)
        loss = tokenizer.loss(sample_invariants, reconstructed, z, z_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check gradients exist
        for name, param in tokenizer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestResidualVQTokenizer:
    """Tests for Residual VQ tokenizer."""

    @pytest.fixture
    def rvq_tokenizer(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.rvq import ResidualVQTokenizer
        return ResidualVQTokenizer(
            invariant_dim=8, latent_dim=16, codebook_size=64, num_codebooks=3
        )

    @pytest.fixture
    def sample_invariants(self):
        pytest.importorskip("torch")
        import torch
        return torch.randn(2, 20, 8) * 0.1

    def test_forward_shape(self, rvq_tokenizer, sample_invariants):
        """Test RVQ forward produces correct shapes."""
        all_indices, reconstructed, z, z_sum = rvq_tokenizer(sample_invariants)
        B, T, C = sample_invariants.shape

        assert len(all_indices) == rvq_tokenizer.num_codebooks
        for i, indices in enumerate(all_indices):
            assert indices.shape == (B, T), f"Codebook {i} indices shape mismatch"
        assert reconstructed.shape == (B, T, C)

    def test_residual_decreasing(self, rvq_tokenizer, sample_invariants):
        """Test that residuals decrease through codebooks."""
        import torch

        # Forward to get internal residuals
        z = rvq_tokenizer.encoder(sample_invariants)
        residuals = z
        norms = []

        for vq in rvq_tokenizer.vqs:
            norms.append(torch.norm(residuals).item())
            _, _, z_q = vq(residuals)
            residuals = residuals - z_q.detach()

        # Residual norm should generally decrease
        assert norms[0] > norms[-1], "Residual should decrease through codebooks"


# ============================================================================
# Compression Tests
# ============================================================================

class TestBPECompressor:
    """Tests for Byte-Pair Encoding compression."""

    @pytest.fixture
    def bpe(self):
        from dhb_xr.tokenization.compression import BPECompressor
        return BPECompressor(vocab_size=256, num_merges=50)

    @pytest.fixture
    def token_corpus(self):
        """Generate a corpus with repeating patterns."""
        np.random.seed(42)
        corpus = []
        for _ in range(10):
            # Create sequences with repeating patterns
            base = np.random.randint(0, 50, size=10).tolist()
            seq = base * 3 + base[:5]  # Repeating pattern
            corpus.append(seq)
        return corpus

    def test_fit(self, bpe, token_corpus):
        """Test BPE fitting."""
        bpe.fit(token_corpus)

        assert bpe._fitted, "BPE should be marked as fitted"
        assert len(bpe.merges) > 0, "Should have learned some merges"
        assert bpe.extended_vocab_size > bpe.vocab_size

    def test_encode_decode_lossless(self, bpe, token_corpus):
        """Test BPE is lossless."""
        bpe.fit(token_corpus)

        for seq in token_corpus:
            encoded = bpe.encode(seq)
            decoded = bpe.decode(encoded)
            assert decoded == seq, "BPE encode/decode should be lossless"

    def test_compression_ratio(self, bpe, token_corpus):
        """Test that BPE achieves compression."""
        bpe.fit(token_corpus)

        ratios = []
        for seq in token_corpus:
            ratio = bpe.compression_ratio(seq)
            ratios.append(ratio)

        avg_ratio = np.mean(ratios)
        assert avg_ratio > 1.0, f"BPE should compress, got ratio {avg_ratio}"

    def test_unfitted_raises(self, bpe):
        """Test that encoding before fitting raises error."""
        with pytest.raises(RuntimeError, match="must be fitted"):
            bpe.encode([1, 2, 3])


class TestEntropyCompressor:
    """Tests for Entropy (Huffman) coding."""

    @pytest.fixture
    def entropy(self):
        from dhb_xr.tokenization.compression import EntropyCompressor
        return EntropyCompressor()

    @pytest.fixture
    def token_corpus(self):
        """Create corpus with skewed distribution."""
        np.random.seed(42)
        # Zipf-like distribution (some tokens much more frequent)
        corpus = []
        for _ in range(10):
            weights = 1.0 / (np.arange(1, 51) ** 0.8)
            weights /= weights.sum()
            seq = np.random.choice(50, size=100, p=weights).tolist()
            corpus.append(seq)
        return corpus

    def test_fit(self, entropy, token_corpus):
        """Test entropy fitting."""
        entropy.fit(token_corpus)

        assert entropy._fitted
        assert len(entropy.codes) > 0
        assert len(entropy.frequencies) > 0

    def test_encode_decode_lossless(self, entropy, token_corpus):
        """Test entropy coding is lossless."""
        entropy.fit(token_corpus)

        for seq in token_corpus:
            binary = entropy.encode(seq)
            decoded = entropy.decode(binary)
            assert decoded == seq, "Entropy encode/decode should be lossless"

    def test_bits_per_token(self, entropy, token_corpus):
        """Test bits per token is reasonable."""
        entropy.fit(token_corpus)

        for seq in token_corpus:
            bpt = entropy.bits_per_token(seq)
            # Should be better than naive 6 bits (log2(50) ~ 5.6)
            assert bpt > 0, "Bits per token should be positive"
            assert bpt < 10, "Bits per token should be reasonable"

    def test_theoretical_entropy(self, entropy, token_corpus):
        """Test theoretical entropy computation."""
        entropy.fit(token_corpus)

        H = entropy.theoretical_entropy()
        assert H > 0, "Entropy should be positive"
        # For 50 tokens, max entropy is log2(50) ~ 5.6
        assert H < 6, "Entropy should be less than uniform maximum"


class TestRLECompressor:
    """Tests for Run-Length Encoding."""

    @pytest.fixture
    def rle(self):
        from dhb_xr.tokenization.compression import RLECompressor
        return RLECompressor(min_run=2)

    def test_encode_decode_lossless(self, rle):
        """Test RLE is lossless."""
        test_cases = [
            [1, 2, 3, 4, 5],  # No runs
            [1, 1, 1, 1, 1],  # Single long run
            [1, 1, 2, 2, 3, 3],  # Multiple runs
            [1, 2, 2, 2, 3, 3, 3, 3, 4],  # Mixed
            [],  # Empty
        ]

        for seq in test_cases:
            encoded = rle.encode(seq)
            decoded = rle.decode(encoded)
            assert decoded == seq, f"RLE failed for {seq}"

    def test_compression_on_runs(self, rle):
        """Test RLE compresses long runs."""
        # Sequence with long run
        seq = [1, 2] + [5] * 50 + [3, 4]

        ratio = rle.compression_ratio(seq)
        assert ratio > 1.0, "RLE should compress sequences with runs"

    def test_stats(self, rle):
        """Test RLE statistics."""
        seq = [1, 1, 1, 2, 2, 3, 3, 3, 3]
        stats = rle.get_stats(seq)

        assert "num_runs" in stats
        assert "avg_run_length" in stats
        assert "max_run_length" in stats
        assert stats["max_run_length"] == 4  # The run of 3s


class TestTokenCompressor:
    """Tests for unified compression pipeline."""

    @pytest.fixture
    def compressor(self):
        from dhb_xr.tokenization.compression import TokenCompressor
        return TokenCompressor(
            vocab_size=256,
            use_rle=True,
            use_bpe=True,
            use_entropy=True,
            bpe_merges=20,
        )

    @pytest.fixture
    def token_corpus(self):
        np.random.seed(42)
        corpus = []
        for _ in range(10):
            # Patterns + some runs
            base = np.random.randint(0, 30, size=20).tolist()
            seq = base + [base[0]] * 10 + base  # Add a static segment
            corpus.append(seq)
        return corpus

    def test_fit(self, compressor, token_corpus):
        """Test pipeline fitting."""
        compressor.fit(token_corpus)
        assert compressor._fitted

    def test_compress_decompress_lossless(self, compressor, token_corpus):
        """Test full pipeline is lossless."""
        compressor.fit(token_corpus)

        for seq in token_corpus:
            compressed = compressor.compress(seq)
            recovered = compressor.decompress(compressed)
            assert recovered == seq, "Full pipeline should be lossless"

    def test_compression_ratio(self, compressor, token_corpus):
        """Test overall compression ratio."""
        compressor.fit(token_corpus)

        ratios = []
        for seq in token_corpus:
            compressed = compressor.compress(seq)
            ratios.append(compressed["overall_ratio"])

        avg_ratio = np.mean(ratios)
        assert avg_ratio > 1.0, f"Should achieve compression, got {avg_ratio}"

    def test_stage_info(self, compressor, token_corpus):
        """Test compression provides stage info."""
        compressor.fit(token_corpus)

        compressed = compressor.compress(token_corpus[0])

        assert "original_length" in compressed
        assert "stages" in compressed
        assert "rle" in compressed["stages"]
        assert "bpe" in compressed["stages"]
        assert "entropy" in compressed["stages"]


class TestTokenReuser:
    """Tests for token reuse detection."""

    @pytest.fixture
    def reuser(self):
        from dhb_xr.tokenization.compression import TokenReuser
        return TokenReuser(window_size=3)

    def test_detect_stable_regions(self, reuser):
        """Test stable region detection."""
        # Sequence with a stable region in the middle
        seq = [1, 2, 3, 5, 5, 5, 5, 5, 5, 7, 8, 9]

        regions = reuser.detect_stable_regions(seq)

        # Should detect the run of 5s as stable
        stable_regions = [(s, e) for s, e, is_stable in regions if is_stable]
        assert len(stable_regions) > 0, "Should detect stable region"

    def test_reuse_potential(self, reuser):
        """Test reuse potential computation."""
        # Highly repetitive sequence
        repetitive = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        stats = reuser.compute_reuse_potential(repetitive)

        assert stats["reuse_fraction"] > 0.5, "Repetitive should be mostly reusable"
        assert stats["potential_speedup"] >= 1.0  # Can be 1.0 in edge cases

        # Random sequence
        random_seq = list(range(20))
        stats_random = reuser.compute_reuse_potential(random_seq)

        # Repetitive should have higher reuse fraction than random
        assert stats["reuse_fraction"] > stats_random["reuse_fraction"]


class TestCompressTokenSequence:
    """Tests for convenience function."""

    def test_bpe_method(self):
        from dhb_xr.tokenization.compression import compress_token_sequence

        tokens = [1, 2, 1, 2, 1, 2, 3, 4]
        result = compress_token_sequence(tokens, method="bpe", num_merges=10)

        assert "encoded" in result
        assert "ratio" in result
        assert result["method"] == "bpe"

    def test_entropy_method(self):
        from dhb_xr.tokenization.compression import compress_token_sequence

        tokens = [1, 1, 2, 2, 2, 3]
        result = compress_token_sequence(tokens, method="entropy")

        assert "binary" in result
        assert "bits_per_token" in result

    def test_rle_method(self):
        from dhb_xr.tokenization.compression import compress_token_sequence

        tokens = [5, 5, 5, 5, 5, 1, 2, 3]
        result = compress_token_sequence(tokens, method="rle")

        assert "encoded" in result
        assert "stats" in result

    def test_invalid_method(self):
        from dhb_xr.tokenization.compression import compress_token_sequence

        with pytest.raises(ValueError, match="Unknown method"):
            compress_token_sequence([1, 2, 3], method="invalid")


# ============================================================================
# Hierarchical Tokenizer Tests
# ============================================================================

class TestHierarchicalTokenizer:
    """Tests for hierarchical tokenizer (requires torch)."""

    @pytest.fixture
    def hierarchical(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.hierarchical import HierarchicalTokenizer
        return HierarchicalTokenizer(
            invariant_dim=8,
            latent_dim=16,
            codebook_size=64,
            num_levels=3,
            temporal_downsample=2,
        )

    @pytest.fixture
    def sample_invariants(self):
        pytest.importorskip("torch")
        import torch
        return torch.randn(2, 32, 8) * 0.1  # T=32 for clean downsampling

    def test_forward_shape(self, hierarchical, sample_invariants):
        """Test hierarchical forward produces correct shapes."""
        all_tokens, reconstructed = hierarchical(sample_invariants)
        B, T, C = sample_invariants.shape

        assert len(all_tokens) == hierarchical.num_levels
        assert reconstructed.shape == (B, T, C)

    def test_variable_levels(self, hierarchical, sample_invariants):
        """Test using fewer levels for coarser output."""
        all_tokens_full, _ = hierarchical(sample_invariants, max_level=None)
        all_tokens_coarse, _ = hierarchical(sample_invariants, max_level=1)

        assert len(all_tokens_coarse) == 1
        assert len(all_tokens_full) == hierarchical.num_levels


class TestProgressiveTokenizer:
    """Tests for progressive refinement tokenizer."""

    @pytest.fixture
    def progressive(self):
        pytest.importorskip("torch")
        from dhb_xr.tokenization.hierarchical import ProgressiveTokenizer
        return ProgressiveTokenizer(
            invariant_dim=8,
            latent_dim=16,
            codebook_size=64,
            num_refinements=2,
        )

    @pytest.fixture
    def sample_invariants(self):
        pytest.importorskip("torch")
        import torch
        return torch.randn(2, 20, 8) * 0.1

    def test_forward_shape(self, progressive, sample_invariants):
        """Test progressive forward."""
        all_tokens, reconstructed = progressive(sample_invariants)
        B, T, C = sample_invariants.shape

        assert len(all_tokens) == 3  # base + 2 refinements
        assert reconstructed.shape == (B, T, C)

    def test_refinement_levels(self, progressive, sample_invariants):
        """Test different refinement levels."""
        tokens_0, recon_0 = progressive(sample_invariants, num_refine=0)
        tokens_2, recon_2 = progressive(sample_invariants, num_refine=2)

        assert len(tokens_0) == 1  # Base only
        assert len(tokens_2) == 3  # Base + 2 refinements


# ============================================================================
# Integration Tests
# ============================================================================

class TestTokenizationCompressionIntegration:
    """End-to-end tests combining tokenization and compression."""

    @pytest.fixture
    def setup(self):
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.vqvae import DHBTokenizer
        from dhb_xr.tokenization.compression import TokenCompressor
        from dhb_xr.encoder.dhb_dr import encode_dhb_dr

        return {
            "torch": torch,
            "DHBTokenizer": DHBTokenizer,
            "TokenCompressor": TokenCompressor,
            "encode_dhb_dr": encode_dhb_dr,
        }

    def test_trajectory_to_compressed_tokens(self, setup):
        """Test full pipeline: trajectory -> invariants -> tokens -> compressed."""
        torch = setup["torch"]
        DHBTokenizer = setup["DHBTokenizer"]
        TokenCompressor = setup["TokenCompressor"]
        encode_dhb_dr = setup["encode_dhb_dr"]

        # Generate trajectory
        n = 50
        t = np.linspace(0, 2 * np.pi, n)
        positions = np.column_stack([0.3 * np.cos(t), 0.3 * np.sin(t), 0.1 * t])
        quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(float)

        # Encode to invariants
        result = encode_dhb_dr(
            positions, quaternions,
            init_pose={'position': positions[0], 'quaternion': quaternions[0]}
        )
        invariants = np.hstack([
            result['linear_motion_invariants'],
            result['angular_motion_invariants']
        ])

        # Tokenize
        tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=32, codebook_size=256)
        inv_tensor = torch.tensor(invariants, dtype=torch.float32).unsqueeze(0)
        tokens, _, _, _ = tokenizer(inv_tensor)
        token_list = tokens[0].numpy().tolist()

        # Compress
        compressor = TokenCompressor(vocab_size=256, bpe_merges=20)
        compressor.fit([token_list])
        compressed = compressor.compress(token_list)

        # Verify lossless
        recovered = compressor.decompress(compressed)
        assert recovered == token_list, "Full pipeline should be lossless"

        # Verify compression achieved
        assert compressed["overall_ratio"] > 1.0, "Should achieve compression"

    def test_batch_tokenization_compression(self, setup):
        """Test batch tokenization followed by compression."""
        torch = setup["torch"]
        DHBTokenizer = setup["DHBTokenizer"]
        TokenCompressor = setup["TokenCompressor"]
        encode_dhb_dr = setup["encode_dhb_dr"]

        # Generate multiple trajectories
        np.random.seed(42)
        token_corpus = []

        for _ in range(5):
            n = 30
            t = np.linspace(0, 2 * np.pi, n)
            r = np.random.uniform(0.2, 0.4)
            positions = np.column_stack([r * np.cos(t), r * np.sin(t), 0.1 * t])
            quaternions = np.tile([1, 0, 0, 0], (n, 1)).astype(float)

            result = encode_dhb_dr(
                positions, quaternions,
                init_pose={'position': positions[0], 'quaternion': quaternions[0]}
            )
            invariants = np.hstack([
                result['linear_motion_invariants'],
                result['angular_motion_invariants']
            ])

            tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=32, codebook_size=256)
            inv_tensor = torch.tensor(invariants, dtype=torch.float32).unsqueeze(0)
            tokens, _, _, _ = tokenizer(inv_tensor)
            token_corpus.append(tokens[0].numpy().tolist())

        # Fit compressor on corpus
        compressor = TokenCompressor(vocab_size=256, bpe_merges=30)
        compressor.fit(token_corpus)

        # Verify all sequences compress/decompress correctly
        for seq in token_corpus:
            compressed = compressor.compress(seq)
            recovered = compressor.decompress(compressed)
            assert recovered == seq


# ============================================================================
# FAST Tokenizer Integration Test
# ============================================================================

class TestFASTIntegration:
    """Integration test: FAST tokenizer in the tokenization pipeline."""

    def test_fast_invariant_tokenization(self):
        """Test FAST tokenizer on invariant-like data."""
        from dhb_xr.tokenization.fast_tokenizer import FASTTokenizer

        rng = np.random.RandomState(42)
        corpus = [rng.randn(30, 8).astype(np.float32) * 0.1 for _ in range(10)]

        fast = FASTTokenizer(scale=10.0, vocab_size=512, num_merges=50)
        fast.fit(corpus)

        for chunk in corpus[:3]:
            tokens = fast.encode(chunk)
            recon = fast.decode(tokens, chunk.shape[0], chunk.shape[1])
            assert recon.shape == chunk.shape
            mse = float(np.mean((chunk - recon) ** 2))
            assert mse < 1.0, f"FAST reconstruction MSE too high: {mse}"


# ============================================================================
# OAT Tokenizer Integration Test
# ============================================================================

class TestOATIntegration:
    """Integration test: OAT tokenizer in the tokenization pipeline."""

    def test_oat_encode_decode_roundtrip(self):
        """Test OAT tokenizer encode-decode roundtrip."""
        pytest.importorskip("torch")
        import torch
        from dhb_xr.tokenization.oat_tokenizer import OATTokenizer

        model = OATTokenizer(
            input_dim=8, input_horizon=20, emb_dim=32, latent_dim=4,
            num_registers=4, fsq_levels=[8, 5, 5, 5],
            encoder_depth=1, decoder_depth=1,
        )
        model.eval()

        x = torch.randn(2, 20, 8) * 0.1
        tokens = model.tokenize(x)
        assert tokens.shape == (2, 4)
        assert tokens.min() >= 0
        assert tokens.max() < model.codebook_size

        recon = model.detokenize(tokens)
        assert recon.shape == x.shape
