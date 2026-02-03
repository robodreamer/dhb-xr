"""
Tests for the generative/flow matching module.

Tests cover:
- ODE solvers (Euler, RK4)
- FlowMatcher (base flow matching)
- VariationalFlowMatcher (V-RFM)
- Latent encoders (Gaussian, Categorical)
- VFMTokenGenerator (integration)
"""

import pytest
import numpy as np

# Skip all tests if torch is not available
torch = pytest.importorskip("torch")

from dhb_xr.generative import (
    FlowMatcher,
    VariationalFlowMatcher,
    VelocityNetwork,
    SinusoidalTimeEmbedding,
    euler_solve,
    rk4_solve,
    ode_solve,
    LatentEncoder,
    CategoricalLatentEncoder,
    HybridLatentEncoder,
    VFMTokenGenerator,
)
from dhb_xr.tokenization import DHBTokenizer


# ============ ODE Solver Tests ============

class TestODESolvers:
    """Test ODE solver implementations."""

    def test_euler_basic(self):
        """Test Euler solver on a simple linear ODE."""
        # dx/dt = -x => solution x(t) = x0 * exp(-t)
        def velocity_fn(x, t):
            return -x

        x0 = torch.ones(1, 10)
        x1 = euler_solve(velocity_fn, x0, t_start=0.0, t_end=1.0, num_steps=100)

        # Should approximate x(1) = exp(-1) ≈ 0.368
        expected = x0 * np.exp(-1)
        assert x1.shape == x0.shape
        assert torch.allclose(x1, expected, atol=0.01)

    def test_rk4_more_accurate(self):
        """Test that RK4 is more accurate than Euler for same step count."""
        def velocity_fn(x, t):
            return -x

        x0 = torch.ones(1, 10)
        expected = x0 * np.exp(-1)

        x1_euler = euler_solve(velocity_fn, x0, 0.0, 1.0, num_steps=10)
        x1_rk4 = rk4_solve(velocity_fn, x0, 0.0, 1.0, num_steps=10)

        error_euler = (x1_euler - expected).abs().mean()
        error_rk4 = (x1_rk4 - expected).abs().mean()

        assert error_rk4 < error_euler, "RK4 should be more accurate"

    def test_ode_solve_interface(self):
        """Test unified ode_solve interface."""
        def velocity_fn(x, t):
            return torch.zeros_like(x)

        x0 = torch.randn(2, 5, 10)

        x1_euler = ode_solve(velocity_fn, x0, method="euler", num_steps=5)
        x1_rk4 = ode_solve(velocity_fn, x0, method="rk4", num_steps=5)

        assert x1_euler.shape == x0.shape
        assert x1_rk4.shape == x0.shape

    def test_return_trajectory(self):
        """Test trajectory return option."""
        def velocity_fn(x, t):
            return torch.ones_like(x)

        x0 = torch.zeros(1, 5)
        x1, trajectory = euler_solve(
            velocity_fn, x0, 0.0, 1.0, num_steps=10, return_trajectory=True
        )

        assert len(trajectory) == 11  # Initial + 10 steps
        assert trajectory[0].shape == x0.shape
        assert trajectory[-1].shape == x0.shape


# ============ Time Embedding Tests ============

class TestTimeEmbedding:
    """Test sinusoidal time embedding."""

    def test_output_shape(self):
        """Test embedding output shape."""
        embed = SinusoidalTimeEmbedding(dim=64)
        t = torch.tensor([0.0, 0.5, 1.0])
        out = embed(t)
        assert out.shape == (3, 64)

    def test_scalar_time(self):
        """Test with scalar time input."""
        embed = SinusoidalTimeEmbedding(dim=32)
        t = torch.tensor(0.5)
        out = embed(t)
        assert out.shape == (1, 32)

    def test_different_times_different_embeddings(self):
        """Test that different times produce different embeddings."""
        embed = SinusoidalTimeEmbedding(dim=64)
        t1 = embed(torch.tensor([0.0]))
        t2 = embed(torch.tensor([0.5]))
        t3 = embed(torch.tensor([1.0]))

        assert not torch.allclose(t1, t2)
        assert not torch.allclose(t2, t3)


# ============ VelocityNetwork Tests ============

class TestVelocityNetwork:
    """Test velocity network."""

    def test_forward_2d(self):
        """Test forward pass with 2D input (B, D)."""
        net = VelocityNetwork(latent_dim=16, hidden_dim=32, num_layers=2)
        z = torch.randn(4, 16)
        t = torch.rand(4)

        v = net(z, t)
        assert v.shape == (4, 16)

    def test_forward_3d(self):
        """Test forward pass with 3D input (B, T, D)."""
        net = VelocityNetwork(latent_dim=16, hidden_dim=32, num_layers=2)
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        v = net(z, t)
        assert v.shape == (4, 50, 16)

    def test_forward_with_condition(self):
        """Test forward pass with conditioning."""
        net = VelocityNetwork(
            latent_dim=16, hidden_dim=32, num_layers=2, condition_dim=8
        )
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)
        w = torch.randn(4, 8)

        v = net(z, t, w)
        assert v.shape == (4, 50, 16)

    def test_scalar_time(self):
        """Test with scalar time."""
        net = VelocityNetwork(latent_dim=16, hidden_dim=32, num_layers=2)
        z = torch.randn(4, 16)

        v = net(z, 0.5)
        assert v.shape == (4, 16)


# ============ FlowMatcher Tests ============

class TestFlowMatcher:
    """Test base FlowMatcher."""

    def test_forward(self):
        """Test forward pass."""
        fm = FlowMatcher(latent_dim=16, hidden_dim=32, num_layers=2)
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        v = fm(z, t)
        assert v.shape == (4, 50, 16)

    def test_interpolate(self):
        """Test linear interpolation."""
        fm = FlowMatcher(latent_dim=16)
        z_0 = torch.zeros(4, 50, 16)
        z_1 = torch.ones(4, 50, 16)

        z_mid = fm.interpolate(z_0, z_1, 0.5)
        assert torch.allclose(z_mid, torch.full_like(z_mid, 0.5))

    def test_target_velocity(self):
        """Test target velocity computation."""
        fm = FlowMatcher(latent_dim=16)
        z_0 = torch.zeros(4, 50, 16)
        z_1 = torch.ones(4, 50, 16)

        v_target = fm.target_velocity(z_0, z_1)
        assert torch.allclose(v_target, torch.ones_like(v_target))

    def test_loss(self):
        """Test loss computation."""
        fm = FlowMatcher(latent_dim=16, hidden_dim=32, num_layers=2)
        z_1 = torch.randn(4, 50, 16)

        loss = fm.loss(z_1)
        assert loss.dim() == 0  # Scalar
        assert loss >= 0

    def test_sample(self):
        """Test sampling."""
        fm = FlowMatcher(latent_dim=16, hidden_dim=32, num_layers=2)

        with torch.no_grad():
            samples = fm.sample(num_samples=2, seq_len=50, num_steps=5)

        assert samples.shape == (2, 50, 16)


# ============ VariationalFlowMatcher Tests ============

class TestVariationalFlowMatcher:
    """Test VariationalFlowMatcher (V-RFM)."""

    def test_encode_posterior(self):
        """Test posterior encoding."""
        vfm = VariationalFlowMatcher(latent_dim=16, condition_dim=8)
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        mean, log_std = vfm.encode_posterior(z, t)
        assert mean.shape == (4, 8)
        assert log_std.shape == (4, 8)

    def test_sample_posterior(self):
        """Test posterior sampling."""
        vfm = VariationalFlowMatcher(latent_dim=16, condition_dim=8)
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        w, mean, log_std = vfm.sample_posterior(z, t)
        assert w.shape == (4, 8)

    def test_loss(self):
        """Test V-RFM loss with KL."""
        vfm = VariationalFlowMatcher(latent_dim=16, condition_dim=8, hidden_dim=32)
        z_1 = torch.randn(4, 50, 16)

        losses = vfm.loss(z_1, beta=0.01)
        assert "total" in losses
        assert "reconstruction" in losses
        assert "kl" in losses
        assert losses["total"] >= 0
        assert losses["kl"] >= 0

    def test_sample_with_w(self):
        """Test sampling with explicit w."""
        vfm = VariationalFlowMatcher(latent_dim=16, condition_dim=8, hidden_dim=32)

        w = torch.randn(2, 8)
        with torch.no_grad():
            samples = vfm.sample(num_samples=2, seq_len=50, num_steps=5, w=w)

        assert samples.shape == (2, 50, 16)

    def test_sample_multimodal(self):
        """Test multi-modal sampling."""
        vfm = VariationalFlowMatcher(latent_dim=16, condition_dim=8, hidden_dim=32)

        with torch.no_grad():
            samples = vfm.sample_multimodal(
                num_samples=2, seq_len=50, num_modes=4, num_steps=5
            )

        assert samples.shape == (8, 50, 16)  # 4 modes * 2 samples

    def test_different_w_different_samples(self):
        """Test that different w produces different samples."""
        vfm = VariationalFlowMatcher(latent_dim=16, condition_dim=8, hidden_dim=32)

        with torch.no_grad():
            w1 = torch.randn(1, 8)
            w2 = torch.randn(1, 8)

            # Use same initial noise
            torch.manual_seed(42)
            s1 = vfm.sample(num_samples=1, seq_len=50, num_steps=5, w=w1)
            torch.manual_seed(42)
            s2 = vfm.sample(num_samples=1, seq_len=50, num_steps=5, w=w2)

        # Should be different due to different w
        assert not torch.allclose(s1, s2)


# ============ Latent Encoder Tests ============

class TestLatentEncoder:
    """Test Gaussian latent encoder."""

    def test_forward(self):
        """Test encoding."""
        enc = LatentEncoder(input_dim=16, latent_dim=8)
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        mean, log_std = enc(z, t)
        assert mean.shape == (4, 8)
        assert log_std.shape == (4, 8)

    def test_sample(self):
        """Test sampling."""
        enc = LatentEncoder(input_dim=16, latent_dim=8)
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        w, mean, log_std = enc.sample(z, t)
        assert w.shape == (4, 8)

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        enc = LatentEncoder(input_dim=16, latent_dim=8)
        mean = torch.zeros(4, 8)
        log_std = torch.zeros(4, 8)

        kl = enc.kl_divergence(mean, log_std)
        # KL should be ~0 when posterior = prior
        assert kl.abs() < 0.01


class TestCategoricalLatentEncoder:
    """Test categorical latent encoder."""

    def test_forward(self):
        """Test logit computation."""
        enc = CategoricalLatentEncoder(input_dim=16, num_categories=4)
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        logits = enc(z, t)
        assert logits.shape == (4, 4)

    def test_sample(self):
        """Test Gumbel-Softmax sampling."""
        enc = CategoricalLatentEncoder(input_dim=16, num_categories=4, embedding_dim=8)
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        w, probs, logits = enc.sample(z, t)
        assert w.shape == (4, 8)
        assert probs.shape == (4, 4)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4))

    def test_sample_prior(self):
        """Test prior sampling."""
        enc = CategoricalLatentEncoder(input_dim=16, num_categories=4, embedding_dim=8)
        w = enc.sample_prior(batch_size=10)
        assert w.shape == (10, 8)


class TestHybridLatentEncoder:
    """Test hybrid (categorical + continuous) encoder."""

    def test_sample(self):
        """Test hybrid sampling."""
        enc = HybridLatentEncoder(
            input_dim=16, num_categories=4, continuous_dim=8
        )
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        w, info = enc.sample(z, t)
        assert w.shape == (4, 8)
        assert "categorical_probs" in info
        assert "continuous_mean" in info

    def test_kl_divergence(self):
        """Test combined KL."""
        enc = HybridLatentEncoder(
            input_dim=16, num_categories=4, continuous_dim=8
        )
        z = torch.randn(4, 50, 16)
        t = torch.rand(4)

        w, info = enc.sample(z, t)
        kl = enc.kl_divergence(info)
        assert kl >= 0


# ============ VFMTokenGenerator Tests ============

class TestVFMTokenGenerator:
    """Test end-to-end VFM token generator."""

    @pytest.fixture
    def generator(self):
        """Create a VFMTokenGenerator for testing."""
        tokenizer = DHBTokenizer(
            invariant_dim=8, latent_dim=16, codebook_size=64
        )
        flow_matcher = VariationalFlowMatcher(
            latent_dim=16, condition_dim=8, hidden_dim=32, num_layers=2
        )
        return VFMTokenGenerator(tokenizer, flow_matcher)

    def test_encode_decode(self, generator):
        """Test encode/decode roundtrip."""
        invariants = torch.randn(2, 50, 8)
        z = generator.encode(invariants)
        recon = generator.decode(z)

        assert z.shape == (2, 50, 16)
        assert recon.shape == (2, 50, 8)

    def test_loss(self, generator):
        """Test loss computation."""
        invariants = torch.randn(2, 50, 8)
        losses = generator.loss(invariants)

        assert "total" in losses
        assert "tokenizer" in losses
        assert "flow" in losses
        assert losses["total"] >= 0

    def test_generate(self, generator):
        """Test generation."""
        with torch.no_grad():
            samples = generator.generate(num_samples=2, seq_len=50, num_steps=5)

        assert samples.shape == (2, 50, 8)

    def test_generate_multimodal(self, generator):
        """Test multi-modal generation."""
        with torch.no_grad():
            samples = generator.generate_multimodal(
                num_samples=1, seq_len=50, num_modes=4, num_steps=5
            )

        assert samples.shape == (4, 50, 8)

    def test_generate_continuation(self, generator):
        """Test continuation generation."""
        prefix = torch.randn(1, 20, 8)
        with torch.no_grad():
            continuation = generator.generate_continuation(
                prefix_invariants=prefix,
                continuation_len=30,
                num_modes=2,
                num_steps=5,
            )

        assert continuation.shape == (2, 50, 8)

    def test_interpolate_latent(self, generator):
        """Test latent interpolation."""
        inv_a = torch.randn(1, 50, 8)
        inv_b = torch.randn(1, 50, 8)

        interp = generator.interpolate_latent(inv_a, inv_b, num_interp=5)
        assert interp.shape == (5, 50, 8)


# ============ Tokenizer API Extensions Tests ============

class TestTokenizerAPIExtensions:
    """Test the new tokenizer API methods for VFM integration."""

    def test_encode_continuous(self):
        """Test continuous encoding."""
        tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
        inv = torch.randn(2, 50, 8)
        z = tokenizer.encode_continuous(inv)
        assert z.shape == (2, 50, 16)

    def test_decode_from_latent(self):
        """Test decoding from continuous latent."""
        tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
        z = torch.randn(2, 50, 16)
        inv = tokenizer.decode_from_latent(z)
        assert inv.shape == (2, 50, 8)

    def test_get_codebook_embeddings(self):
        """Test codebook access."""
        tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
        codebook = tokenizer.get_codebook_embeddings()
        assert codebook.shape == (64, 16)

    def test_embed_tokens(self):
        """Test token embedding."""
        tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
        indices = torch.randint(0, 64, (2, 50))
        embeddings = tokenizer.embed_tokens(indices)
        assert embeddings.shape == (2, 50, 16)

    def test_decode_tokens(self):
        """Test token decoding."""
        tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
        indices = torch.randint(0, 64, (2, 50))
        inv = tokenizer.decode_tokens(indices)
        assert inv.shape == (2, 50, 8)


# ============ Integration Tests ============

class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_training_step(self):
        """Test a single training step."""
        tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
        flow_matcher = VariationalFlowMatcher(
            latent_dim=16, condition_dim=8, hidden_dim=32
        )
        generator = VFMTokenGenerator(tokenizer, flow_matcher)

        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)

        # Training step
        invariants = torch.randn(4, 50, 8)
        optimizer.zero_grad()
        losses = generator.loss(invariants)
        losses["total"].backward()
        optimizer.step()

        # Check gradients were computed
        for p in generator.parameters():
            if p.requires_grad:
                assert p.grad is not None or p.numel() == 0

    def test_generate_then_encode(self):
        """Test that generated samples can be re-encoded."""
        tokenizer = DHBTokenizer(invariant_dim=8, latent_dim=16, codebook_size=64)
        flow_matcher = VariationalFlowMatcher(
            latent_dim=16, condition_dim=8, hidden_dim=32
        )
        generator = VFMTokenGenerator(tokenizer, flow_matcher)

        with torch.no_grad():
            # Generate
            samples = generator.generate(num_samples=2, seq_len=50, num_steps=5)

            # Re-encode
            z = generator.encode(samples)

            # Decode again
            recon = generator.decode(z)

        assert recon.shape == samples.shape
