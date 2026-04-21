"""Tests for DHBActionHead."""
import pytest
import numpy as np
import torch.nn as nn

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
            contact_dim=2,
            lambda_contact=0.5,
        )
        self.B = 2
        self.S = 10  # VLM sequence length

    def test_forward_shape(self):
        """Predicted invariants have correct shape."""
        vlm_hidden = torch.randn(self.B, self.S, 128)
        result = self.head(vlm_hidden)
        assert result["pred_invariants"].shape == (self.B, 20, 8)
        assert result["pred_contact_logits"].shape == (self.B, 20, 2)
        assert result["pred_contact_probs"].shape == (self.B, 20, 2)

    def test_forward_with_state(self):
        """State conditioning doesn't break output shape."""
        vlm_hidden = torch.randn(self.B, self.S, 128)
        state = torch.randn(self.B, 8)
        result = self.head(vlm_hidden, state=state)
        assert result["pred_invariants"].shape == (self.B, 20, 8)

    def test_state_conditioning_changes_predictions(self):
        """Regression test for SmolVLA closed-loop usage: state must affect predictions.

        The earlier SmolVLA evaluation scripts often called ``predict(vlm_feat)``
        without forwarding the current 8D EE state, even though the head exposes
        a dedicated state-conditioning path. This test makes that path
        deterministic and verifies that changing ``state`` changes the predicted
        invariants.
        """

        class RepeatFirstMemoryToken(nn.Module):
            def forward(self, queries, memory):
                return memory[:, :1, :].expand(-1, queries.shape[1], -1)

        head = DHBActionHead(
            vlm_dim=16,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        head.decoder = RepeatFirstMemoryToken()

        with torch.no_grad():
            # Zero out the visual path so only explicit state conditioning remains.
            head.vlm_proj.weight.zero_()
            head.vlm_proj.bias.zero_()
            head.query_embed.weight.zero_()

            # Route state[:8] -> hidden[:8] exactly.
            head.state_proj.weight.zero_()
            head.state_proj.bias.zero_()
            head.state_proj.weight[:8, :8] = torch.eye(8)

            # Route hidden[:8] -> invariant[:8] exactly.
            head.invariant_head.weight.zero_()
            head.invariant_head.bias.zero_()
            head.invariant_head.weight[:, :8] = torch.eye(8)

        vlm_hidden = torch.randn(2, 3, 16)
        state_a = torch.tensor(
            [[1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5],
             [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4, 0.5]]
        )
        state_b = torch.tensor(
            [[9.0, 8.0, 7.0, 0.9, 0.8, 0.7, 0.6, 0.5],
             [9.0, 8.0, 7.0, 0.9, 0.8, 0.7, 0.6, 0.5]]
        )

        pred_no_state = head(vlm_hidden)["pred_invariants"]
        pred_a = head(vlm_hidden, state=state_a)["pred_invariants"]
        pred_b = head(vlm_hidden, state=state_b)["pred_invariants"]

        assert torch.allclose(pred_no_state, torch.zeros_like(pred_no_state))
        assert torch.allclose(pred_a[:, 0, :], state_a, atol=1e-6)
        assert torch.allclose(pred_b[:, 0, :], state_b, atol=1e-6)
        assert not torch.allclose(pred_a, pred_b), (
            "Changing the current EE state must change predicted invariants; "
            "otherwise the state-conditioning path is effectively unused."
        )

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
        target_contact = torch.zeros(self.B, 20, 2)
        target_contact[:, 5, 0] = 1.0
        result = self.head(
            vlm_hidden,
            target_invariants=target_inv,
            target_actions=target_act,
            target_contact=target_contact,
        )
        assert "loss" in result
        assert "loss_inv" in result
        assert "loss_act" in result
        assert "loss_contact" in result

    def test_contact_loss_accepts_per_channel_pos_weights(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            contact_dim=2,
            lambda_contact=1.0,
            contact_pos_weight=[16.0, 4.0],
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_contact = torch.zeros(self.B, 8, 2)
        target_contact[:, 2, 0] = 1.0
        target_contact[:, 5, 1] = 1.0
        result = head(vlm_hidden, target_contact=target_contact)
        assert "loss_contact" in result
        assert result["loss_contact"].item() > 0.0

    def test_contact_loss_supports_three_way_event_classes(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            contact_dim=3,
            lambda_contact=1.0,
            contact_loss_type="cross_entropy",
            contact_class_weight=[1.0, 16.0, 8.0],
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_contact = torch.zeros(self.B, 8, dtype=torch.long)
        target_contact[:, 2] = 1
        target_contact[:, 5] = 2
        result = head(vlm_hidden, target_contact=target_contact)
        assert "pred_contact_logits" in result
        assert result["pred_contact_logits"].shape == (self.B, 8, 3)
        assert "loss_contact" in result
        assert result["loss_contact"].item() > 0.0

    def test_contact_event_and_phase_losses_can_train_together(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            contact_dim=3,
            lambda_contact=1.0,
            contact_loss_type="cross_entropy",
            contact_class_weight=[1.0, 16.0, 8.0],
            contact_phase_dim=3,
            lambda_contact_phase=0.5,
            contact_phase_loss_type="cross_entropy",
            contact_phase_class_weight=[0.5, 8.0, 4.0],
            contact_condition_on_phase=True,
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_contact = torch.zeros(self.B, 8, dtype=torch.long)
        target_contact[:, 2] = 1
        target_contact[:, 5] = 2
        target_contact_phase = torch.zeros(self.B, 8, dtype=torch.long)
        target_contact_phase[:, 2:4] = 1
        target_contact_phase[:, 5:7] = 2
        result = head(
            vlm_hidden,
            target_contact=target_contact,
            target_contact_phase=target_contact_phase,
        )
        assert "pred_contact_logits" in result
        assert "pred_contact_phase_logits" in result
        assert result["pred_contact_phase_logits"].shape == (self.B, 8, 3)
        assert result["pred_contact_phase_probs"].shape == (self.B, 8, 3)
        assert "loss_contact" in result
        assert "loss_contact_phase" in result
        assert result["loss_contact"].item() > 0.0
        assert result["loss_contact_phase"].item() > 0.0

    def test_gripper_state_and_phase_losses_can_train_together(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            contact_dim=2,
            lambda_contact=1.0,
            contact_loss_type="cross_entropy",
            contact_class_weight=[1.0, 1.0],
            contact_mode="gripper_state",
            contact_phase_dim=3,
            lambda_contact_phase=0.5,
            contact_phase_loss_type="cross_entropy",
            contact_phase_class_weight=[0.5, 8.0, 4.0],
            contact_condition_on_phase=True,
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_contact = torch.zeros(self.B, 8, dtype=torch.long)
        target_contact[:, 2:6] = 1
        target_contact_phase = torch.zeros(self.B, 8, dtype=torch.long)
        target_contact_phase[:, 2:4] = 1
        target_contact_phase[:, 5:7] = 2
        result = head(
            vlm_hidden,
            target_contact=target_contact,
            target_contact_phase=target_contact_phase,
        )
        assert result["contact_mode"] == "gripper_state"
        assert "pred_contact_logits" in result
        assert result["pred_contact_logits"].shape == (self.B, 8, 2)
        assert "loss_contact" in result
        assert "loss_contact_phase" in result
        assert result["loss_contact"].item() > 0.0
        assert result["loss_contact_phase"].item() > 0.0

    def test_event_phase_and_gripper_state_can_train_together(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            contact_dim=3,
            lambda_contact=1.0,
            contact_loss_type="cross_entropy",
            contact_class_weight=[0.5, 8.0, 4.0],
            contact_mode="event",
            contact_phase_dim=3,
            lambda_contact_phase=0.5,
            contact_phase_loss_type="cross_entropy",
            contact_phase_class_weight=[0.5, 8.0, 4.0],
            gripper_state_dim=2,
            lambda_gripper_state=1.0,
            gripper_state_loss_type="cross_entropy",
            gripper_state_class_weight=[1.0, 1.0],
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_contact = torch.zeros(self.B, 8, dtype=torch.long)
        target_contact[:, 3] = 1
        target_contact[:, 6] = 2
        target_contact_phase = torch.zeros(self.B, 8, dtype=torch.long)
        target_contact_phase[:, 3:5] = 1
        target_contact_phase[:, 6:8] = 2
        target_gripper_state = torch.zeros(self.B, 8, dtype=torch.long)
        target_gripper_state[:, 3:6] = 1
        result = head(
            vlm_hidden,
            target_contact=target_contact,
            target_contact_phase=target_contact_phase,
            target_gripper_state=target_gripper_state,
        )
        assert result["contact_mode"] == "event"
        assert result["pred_contact_logits"].shape == (self.B, 8, 3)
        assert result["pred_contact_phase_logits"].shape == (self.B, 8, 3)
        assert result["pred_gripper_state_logits"].shape == (self.B, 8, 2)
        assert "loss_contact" in result
        assert "loss_contact_phase" in result
        assert "loss_gripper_state" in result
        assert result["loss_gripper_state"].item() > 0.0

    def test_action_loss_can_ignore_gripper_dim(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            action_dim=7,
            lambda_act=1.0,
            action_loss_ignore_gripper=True,
        )
        head.eval()
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_actions = torch.zeros(self.B, 8, 7)

        with torch.no_grad():
            out = head(vlm_hidden)
            pred = out["pred_actions"].detach().clone()

        target_actions[..., :6] = pred[..., :6]
        target_actions[..., 6] = pred[..., 6] + 10.0

        result = head(vlm_hidden, target_actions=target_actions)
        assert "loss_act" in result
        assert result["loss_act"].item() < 1e-5

    def test_committed_prefix_conditioning_biases_marked_queries_only(self):
        class ReturnQueries(nn.Module):
            def forward(self, queries, memory):
                return queries

        head = DHBActionHead(
            vlm_dim=16,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        head.decoder = ReturnQueries()

        with torch.no_grad():
            head.vlm_proj.weight.zero_()
            head.vlm_proj.bias.zero_()
            head.query_embed.weight.zero_()
            head.committed_prefix_proj.weight.zero_()
            head.committed_prefix_proj.bias.zero_()
            head.committed_prefix_proj.weight[:8, :8] = torch.eye(8)
            head.committed_prefix_bias.zero_()
            head.invariant_head.weight.zero_()
            head.invariant_head.bias.zero_()
            head.invariant_head.weight[:, :8] = torch.eye(8)

        vlm_hidden = torch.randn(1, 3, 16)
        committed_prefix = torch.zeros(1, 4, 8)
        committed_prefix[:, 0, :] = 1.0
        committed_prefix[:, 1, :] = 2.0
        committed_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

        result = head(
            vlm_hidden,
            committed_prefix_invariants=committed_prefix,
            committed_prefix_mask=committed_mask,
        )
        pred = result["pred_invariants"]
        assert torch.allclose(pred[:, 0, :], torch.ones(1, 8), atol=1e-6)
        assert torch.allclose(pred[:, 1, :], torch.full((1, 8), 2.0), atol=1e-6)
        assert torch.allclose(pred[:, 2:, :], torch.zeros(1, 2, 8), atol=1e-6)

    def test_loss_mask_ignores_committed_prefix_steps(self):
        class ReturnQueries(nn.Module):
            def forward(self, queries, memory):
                return queries

        head = DHBActionHead(
            vlm_dim=16,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
        )
        head.decoder = ReturnQueries()

        with torch.no_grad():
            head.vlm_proj.weight.zero_()
            head.vlm_proj.bias.zero_()
            head.query_embed.weight.zero_()
            head.invariant_head.weight.zero_()
            head.invariant_head.bias.zero_()

        vlm_hidden = torch.randn(1, 3, 16)
        target_inv = torch.zeros(1, 4, 8)
        target_inv[:, :2, :] = 5.0
        loss_mask = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        result = head(vlm_hidden, target_invariants=target_inv, loss_mask=loss_mask)
        assert "loss_inv" in result
        assert result["loss_inv"].item() < 1e-7

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
