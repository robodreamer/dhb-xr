"""Tests for DHBActionHead."""
import pytest
import numpy as np

torch = pytest.importorskip("torch")

from dhb_xr.integration.vla.action_head import DHBActionHead


class TestDHBActionHead:
    """Test the DHB action head module."""

    def setup_method(self):
        self.head = DHBActionHead(
            vlm_dim=128,
            action_horizon=20,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
        )
        self.B = 2
        self.S = 10  # VLM sequence length

    def test_forward_shape(self):
        """Predicted invariants have correct shape."""
        vlm_hidden = torch.randn(self.B, self.S, 128)
        result = self.head(vlm_hidden)
        assert result["pred_invariants"].shape == (self.B, 20, 8)

    def test_forward_with_state(self):
        """State conditioning doesn't break output shape."""
        vlm_hidden = torch.randn(self.B, self.S, 128)
        state = torch.randn(self.B, 8)
        result = self.head(vlm_hidden, state=state)
        assert result["pred_invariants"].shape == (self.B, 20, 8)

    def test_training_loss(self):
        """Training loss is computed when targets provided."""
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_inv = torch.randn(self.B, 20, 8)
        result = self.head(vlm_hidden, target_invariants=target_inv)
        assert "loss" in result
        assert "loss_inv" in result
        assert result["loss"].requires_grad

    def test_training_loss_with_actions(self):
        """Combined loss computed with both invariant and action targets."""
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_inv = torch.randn(self.B, 20, 8)
        target_act = torch.randn(self.B, 20, 7)
        result = self.head(
            vlm_hidden,
            target_invariants=target_inv,
            target_actions=target_act,
        )
        assert "loss" in result
        assert "loss_inv" in result
        assert "loss_act" in result

    def test_predict_invariants_only(self):
        """Predict mode returns invariants without gradients."""
        vlm_hidden = torch.randn(self.B, self.S, 128)
        result = self.head.predict(vlm_hidden)
        assert not result["pred_invariants"].requires_grad

    def test_predict_with_decode(self):
        """Predict with initial pose decodes invariants to trajectory."""
        vlm_hidden = torch.randn(self.B, self.S, 128)
        initial_pose = {
            "position": torch.zeros(self.B, 3),
            "quaternion": torch.tensor([[1.0, 0, 0, 0]] * self.B),
        }
        result = self.head.predict(vlm_hidden, initial_pose=initial_pose)
        assert "positions" in result
        assert "quaternions" in result
        # Positions should have T+padding timesteps
        assert result["positions"].shape[0] == self.B
        assert result["positions"].shape[2] == 3

    def test_gradient_flow(self):
        """Gradients flow through the entire head."""
        vlm_hidden = torch.randn(self.B, self.S, 128, requires_grad=True)
        target_inv = torch.randn(self.B, 20, 8)
        result = self.head(vlm_hidden, target_invariants=target_inv)
        result["loss"].backward()
        assert vlm_hidden.grad is not None
        assert vlm_hidden.grad.abs().sum() > 0

    def test_optim_params(self):
        """get_optim_params returns valid parameter dict."""
        params = self.head.get_optim_params()
        assert "params" in params
        param_count = sum(p.numel() for p in params["params"])
        assert param_count > 0

    def test_oat_integration(self):
        """OAT tokenizer integrates with the action head."""
        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=20,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_oat=True,
            oat_num_registers=4,
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")

        head_oat.train()
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_inv = torch.randn(self.B, 20, 8)
        result = head_oat(vlm_hidden, target_invariants=target_inv)
        assert "oat_loss" in result
        assert result["loss"].requires_grad
