"""Tests for DHBActionHead."""
import pytest
import numpy as np
import torch.nn as nn

torch = pytest.importorskip("torch")

from dhb_xr.integration.vla.action_head import DHBActionHead
from dhb_xr.tokenization.oat_tokenizer import OATTokenizer


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

    def test_object_target_auxiliary_head_can_condition_closed_residual(self):
        """Offline object-target supervision can train a learned geometry prior."""
        head = DHBActionHead(
            vlm_dim=32,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            action_dim=7,
            lambda_act=0.0,
            object_target_dim=3,
            lambda_object_target=1.5,
            object_target_readout="cross_attention",
            action_closed_residual_condition_on_object_target=True,
            use_action_closed_residual_adapter=True,
            action_closed_residual_gate_source="none",
        )
        vlm_hidden = torch.randn(self.B, self.S, 32)
        target_object = torch.randn(self.B, 4, 3) * 0.05
        result = head(
            vlm_hidden,
            target_object_target=target_object,
            target_actions=torch.zeros(self.B, 4, 7),
        )

        assert result["pred_object_target"].shape == (self.B, 4, 3)
        assert result["pred_action_closed_residual"].shape == (self.B, 4, 7)
        assert head.object_target_cross_attn is not None
        assert head.object_target_context_proj is not None
        assert torch.allclose(
            head.object_target_context_proj.weight,
            torch.zeros_like(head.object_target_context_proj.weight),
        )
        assert "loss_object_target" in result
        assert result["loss_object_target"].requires_grad
        assert result["loss"].requires_grad

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

    def test_head_query_residual_preserves_timestep_identity_when_decoder_collapses(self):
        """Prediction heads can still see timestep identity if decoder slots collapse."""

        class ZeroDecoder(nn.Module):
            def forward(self, queries, memory):
                return torch.zeros_like(queries)

        head = DHBActionHead(
            vlm_dim=16,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            head_query_residual_scale=1.0,
        )
        head.decoder = ZeroDecoder()

        with torch.no_grad():
            head.query_embed.weight.zero_()
            head.query_embed.weight[:, :4] = torch.eye(4)
            head.invariant_head.weight.zero_()
            head.invariant_head.bias.zero_()
            head.invariant_head.weight[:4, :4] = torch.eye(4)

        vlm_hidden = torch.randn(1, 3, 16)
        pred = head(vlm_hidden)["pred_invariants"]

        assert torch.allclose(pred[0, :, :4], torch.eye(4), atol=1e-6)
        assert not torch.allclose(pred[:, 0, :], pred[:, 1, :])

        head.head_query_residual_scale = 0.0
        collapsed = head(vlm_hidden)["pred_invariants"]
        assert torch.allclose(collapsed, torch.zeros_like(collapsed), atol=1e-6)

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

    def test_bce_gripper_state_accepts_categorical_closed_labels(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            gripper_state_dim=1,
            lambda_gripper_state=1.0,
            gripper_state_loss_type="bce",
            gripper_state_pos_weight=3.0,
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_gripper_state = torch.zeros(self.B, 8, dtype=torch.long)
        target_gripper_state[:, 4:] = 1

        result = head(vlm_hidden, target_gripper_state=target_gripper_state)

        assert result["pred_gripper_state_logits"].shape == (self.B, 8, 1)
        assert result["pred_gripper_state_probs"].shape == (self.B, 8, 1)
        assert "loss_gripper_state" in result
        assert torch.isfinite(result["loss_gripper_state"])
        assert result["loss_gripper_state"].item() > 0.0

    def test_bce_contact_gripper_state_accepts_categorical_closed_labels(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            contact_dim=1,
            lambda_contact=1.0,
            contact_mode="gripper_state",
            contact_loss_type="bce",
            contact_pos_weight=3.0,
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_contact = torch.zeros(self.B, 8, dtype=torch.long)
        target_contact[:, 4:] = 1

        result = head(vlm_hidden, target_contact=target_contact)

        assert result["pred_contact_logits"].shape == (self.B, 8, 1)
        assert result["pred_contact_probs"].shape == (self.B, 8, 1)
        assert "loss_contact" in result
        assert torch.isfinite(result["loss_contact"])
        assert result["loss_contact"].item() > 0.0

    def test_gripper_state_smooth_loss_penalizes_same_phase_flicker(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            gripper_state_dim=1,
            lambda_gripper_state=0.0,
            gripper_state_loss_type="bce",
            lambda_gripper_state_smooth=1.0,
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_gripper_state = torch.zeros(self.B, 8, dtype=torch.long)
        target_gripper_state[:, 4:] = 1

        result = head(vlm_hidden, target_gripper_state=target_gripper_state)

        assert "loss_gripper_state_smooth" in result
        assert torch.isfinite(result["loss_gripper_state_smooth"])
        assert result["loss_gripper_state_smooth"].item() > 0.0
        assert "loss" in result

    def test_gripper_state_margin_loss_penalizes_uncertain_state(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            gripper_state_dim=1,
            lambda_gripper_state=0.0,
            gripper_state_loss_type="bce",
            lambda_gripper_state_margin=1.0,
            gripper_state_closed_margin=1.0,
            gripper_state_open_margin=0.0,
        )
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_gripper_state = torch.zeros(self.B, 8, dtype=torch.long)
        target_gripper_state[:, 2:6] = 1

        result = head(vlm_hidden, target_gripper_state=target_gripper_state)

        assert "loss_gripper_state_margin" in result
        assert torch.isfinite(result["loss_gripper_state_margin"])
        assert result["loss_gripper_state_margin"].item() > 0.0
        assert "loss" in result

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

    def test_action_loss_channel_weight_emphasizes_selected_dims(self):
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
            action_loss_channel_weight=[1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        )
        head.eval()
        vlm_hidden = torch.randn(self.B, self.S, 128)

        with torch.no_grad():
            out = head(vlm_hidden)
            pred = out["pred_actions"].detach().clone()

        target_actions = pred.clone()
        target_actions[..., 0] += 1.0
        target_actions[..., 2] += 2.0

        result = head(vlm_hidden, target_actions=target_actions)

        assert "loss_act" in result
        assert torch.allclose(result["loss_act"], torch.tensor(1.4), atol=1e-5)

    def test_action_loss_target_shift_aligns_decode_slots(self):
        class ReturnQueries(nn.Module):
            def forward(self, queries, memory):
                return queries

        head = DHBActionHead(
            vlm_dim=8,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=8,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            action_dim=7,
            lambda_act=1.0,
            action_loss_target_shift=1,
        )
        head.decoder = ReturnQueries()

        with torch.no_grad():
            head.query_embed.weight.zero_()
            head.query_embed.weight[:4, :4] = torch.eye(4)
            head.action_head.weight.zero_()
            head.action_head.bias.zero_()
            head.action_head.weight[0, :4] = torch.tensor([0.0, 1.0, 2.0, 3.0])

        vlm_hidden = torch.zeros(1, 1, 8)
        target_actions = torch.zeros(1, 4, 7)
        target_actions[0, :, 0] = torch.tensor([1.0, 2.0, 3.0, 99.0])

        result = head(vlm_hidden, target_actions=target_actions)

        assert "loss_act" in result
        assert result["loss_act"].item() < 1e-6

    def test_action_residual_adapter_zero_init_preserves_base_actions(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            action_dim=7,
            use_action_residual_adapter=True,
            action_residual_hidden_dim=16,
            action_residual_scale=0.25,
        )
        head.eval()
        vlm_hidden = torch.randn(self.B, self.S, 128)

        with torch.no_grad():
            result = head(vlm_hidden)

        assert "pred_actions_base" in result
        assert "pred_action_residual" in result
        assert torch.allclose(
            result["pred_action_residual"],
            torch.zeros_like(result["pred_action_residual"]),
            atol=1e-7,
        )
        assert torch.allclose(result["pred_actions"], result["pred_actions_base"], atol=1e-7)

    def test_action_residual_l2_loss_is_reported_when_adapter_moves(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            action_dim=7,
            lambda_act=0.0,
            use_action_residual_adapter=True,
            action_residual_hidden_dim=16,
            action_residual_scale=1.0,
            lambda_action_residual_l2=2.0,
        )
        with torch.no_grad():
            head.action_residual_adapter[-1].bias.fill_(0.5)

        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_actions = torch.zeros(self.B, 8, 7)
        result = head(vlm_hidden, target_actions=target_actions)

        assert "loss_action_residual_l2" in result
        assert torch.isclose(
            result["loss_action_residual_l2"],
            torch.tensor(0.25),
            atol=1e-6,
        )
        assert "loss" in result

    def test_action_base_cumsum_loss_penalizes_accumulated_residual_drift(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            action_dim=7,
            lambda_act=0.0,
            use_action_residual_adapter=True,
            action_residual_hidden_dim=16,
            action_residual_scale=1.0,
            lambda_action_base_cumsum_l2=3.0,
            action_base_cumsum_dims=3,
        )
        with torch.no_grad():
            head.action_residual_adapter[-1].bias.zero_()
            head.action_residual_adapter[-1].bias[0] = 1.0

        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_actions = torch.zeros(self.B, 4, 7)
        result = head(vlm_hidden, target_actions=target_actions)

        assert "loss_action_base_cumsum_l2" in result
        assert torch.isclose(
            result["loss_action_base_cumsum_l2"],
            torch.tensor(2.5),
            atol=1e-6,
        )
        assert torch.isclose(result["loss"], torch.tensor(7.5), atol=1e-6)

    def test_closed_residual_adapter_is_gated_by_current_gripper_state(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            action_dim=7,
            use_action_closed_residual_adapter=True,
            action_closed_residual_hidden_dim=16,
            action_closed_residual_scale=1.0,
            action_closed_residual_gripper_threshold=0.015,
            action_closed_residual_gate_temperature=0.001,
        )
        with torch.no_grad():
            head.action_closed_residual_adapter[-1].bias.fill_(1.0)

        head.eval()
        vlm_hidden = torch.randn(1, self.S, 128)
        open_state = torch.zeros(1, 8)
        open_state[:, -1] = 0.04
        closed_state = torch.zeros(1, 8)
        closed_state[:, -1] = 0.0

        with torch.no_grad():
            open_result = head(vlm_hidden, state=open_state)
            closed_result = head(vlm_hidden, state=closed_state)

        assert open_result["action_closed_residual_gate"].max().item() < 1e-4
        assert closed_result["action_closed_residual_gate"].min().item() > 0.999
        assert torch.allclose(
            open_result["pred_actions"],
            open_result["pred_actions_base"],
            atol=1e-4,
        )
        assert torch.allclose(
            closed_result["pred_actions"][..., :6],
            closed_result["pred_actions_base"][..., :6] + 1.0,
            atol=1e-3,
        )
        assert torch.allclose(
            closed_result["pred_actions"][..., 6],
            closed_result["pred_actions_base"][..., 6],
            atol=1e-6,
        )

    def test_closed_residual_l2_loss_is_reported_when_adapter_moves(self):
        head = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            action_dim=7,
            lambda_act=0.0,
            use_action_closed_residual_adapter=True,
            action_closed_residual_hidden_dim=16,
            action_closed_residual_scale=1.0,
            action_closed_residual_gate_source="none",
            action_closed_residual_zero_gripper_dim=False,
            lambda_action_closed_residual_l2=2.0,
        )
        with torch.no_grad():
            head.action_closed_residual_adapter[-1].bias.fill_(0.5)

        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_actions = torch.zeros(self.B, 8, 7)
        result = head(vlm_hidden, target_actions=target_actions)

        assert "loss_action_closed_residual_l2" in result
        assert torch.isclose(
            result["loss_action_closed_residual_l2"],
            torch.tensor(0.25),
            atol=1e-6,
        )
        assert "loss" in result

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

    def test_invariant_loss_channel_scale_and_delta_loss(self):
        class ReturnZeros(nn.Module):
            def forward(self, queries, memory):
                return torch.zeros_like(queries)

        head = DHBActionHead(
            vlm_dim=16,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            invariant_loss_channel_scale=[0.1] + [1.0] * 7,
            lambda_invariant_delta=2.0,
        )
        head.decoder = ReturnZeros()

        with torch.no_grad():
            head.vlm_proj.weight.zero_()
            head.vlm_proj.bias.zero_()
            head.query_embed.weight.zero_()
            head.invariant_head.weight.zero_()
            head.invariant_head.bias.zero_()

        vlm_hidden = torch.randn(1, 3, 16)
        target_inv = torch.zeros(1, 4, 8)
        target_inv[:, :, 0] = torch.tensor([0.0, 0.1, 0.2, 0.3])

        result = head(vlm_hidden, target_invariants=target_inv)

        assert "loss_inv_raw" in result
        assert "loss_inv_delta" in result
        assert torch.isclose(result["loss_inv"], torch.tensor(0.4375), atol=1e-6)
        assert torch.isclose(result["loss_inv_raw"], torch.tensor(0.004375), atol=1e-6)
        assert torch.isclose(result["loss_inv_delta"], torch.tensor(0.125), atol=1e-6)
        assert torch.isclose(result["loss"], torch.tensor(0.6875), atol=1e-6)

    def test_prefix_loss_weight_upweights_executed_steps(self):
        class ReturnZeros(nn.Module):
            def forward(self, queries, memory):
                return torch.zeros_like(queries)

        head = DHBActionHead(
            vlm_dim=16,
            action_horizon=4,
            invariant_dim=8,
            hidden_dim=16,
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            prefix_loss_steps=1,
            prefix_loss_weight=3.0,
        )
        head.decoder = ReturnZeros()

        with torch.no_grad():
            head.vlm_proj.weight.zero_()
            head.vlm_proj.bias.zero_()
            head.query_embed.weight.zero_()
            head.invariant_head.weight.zero_()
            head.invariant_head.bias.zero_()

        vlm_hidden = torch.randn(1, 3, 16)
        target_inv = torch.zeros(1, 4, 8)
        target_inv[:, 0, 0] = 1.0

        result = head(vlm_hidden, target_invariants=target_inv)

        assert torch.isclose(result["loss_inv"], torch.tensor(0.0625), atol=1e-6)

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
            dropout=0.0,
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

    def test_oat_token_supervision_derives_labels_by_default(self):
        """Exported OAT IDs are ignored unless explicitly enabled.

        LeRobot windows exported per-row token vectors into (B, T, K). Those
        IDs are only valid for the tokenizer instance used at export time, so
        the action head must default to deriving labels from target invariants
        with its own tokenizer.
        """
        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.0,
            use_oat=True,
            oat_num_registers=4,
            supervise_oat_tokens=True,
            lambda_oat_token=1.0,
            lambda_oat_autoencode=0.0,
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")
        head_oat.eval()

        target_inv = torch.randn(self.B, 8, 8)
        exported_window = torch.zeros(self.B, 8, 4, dtype=torch.long)
        resolved = head_oat._resolve_target_oat_tokens(
            exported_window,
            target_inv,
            device=target_inv.device,
        )
        expected = head_oat.oat.tokenize(target_inv.detach())
        assert torch.equal(resolved, expected)

        vlm_hidden = torch.randn(self.B, self.S, 128)
        result = head_oat(vlm_hidden, target_invariants=target_inv, target_oat_tokens=exported_window)
        assert "loss_oat_token" in result
        assert result["loss"].requires_grad

    def test_exported_oat_token_windows_select_current_row_when_enabled(self):
        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_oat=True,
            oat_num_registers=4,
            supervise_oat_tokens=True,
            use_exported_oat_tokens=True,
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")

        exported_window = torch.arange(self.B * 8 * 4, dtype=torch.long).reshape(self.B, 8, 4)
        resolved = head_oat._resolve_target_oat_tokens(
            exported_window,
            target_invariants=None,
            device=exported_window.device,
        )
        assert torch.equal(resolved, exported_window[:, 0, :])

    def test_float_windowed_oat_token_targets_are_rejected(self):
        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_oat=True,
            oat_num_registers=4,
            supervise_oat_tokens=True,
            use_exported_oat_tokens=True,
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")

        exported_window = torch.zeros(self.B, 8, 4, dtype=torch.float32)
        with pytest.raises(ValueError, match="ambiguous"):
            head_oat._resolve_target_oat_tokens(
                exported_window,
                target_invariants=None,
                device=exported_window.device,
            )

    def test_oat_register_diversity_loss_can_train(self):
        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_oat=True,
            oat_num_registers=4,
            lambda_oat_autoencode=0.0,
            lambda_oat_register_diversity=0.05,
            oat_autoencode_target="target",
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")

        head_oat.train()
        vlm_hidden = torch.randn(self.B, self.S, 128)
        target_inv = torch.randn(self.B, 8, 8)
        result = head_oat(vlm_hidden, target_invariants=target_inv)

        assert "loss_oat_register_diversity" in result
        assert result["loss_oat_register_diversity"].requires_grad
        assert result["loss_oat_register_diversity"].item() >= 0.0
        assert result["loss"].item() >= result["loss_inv"].item()

    def test_pretrained_oat_checkpoint_loads_with_normalization(self, tmp_path):
        tokenizer = OATTokenizer(
            input_dim=8,
            input_horizon=8,
            emb_dim=16,
            latent_dim=4,
            num_registers=4,
            fsq_levels=[8, 5, 5, 5],
            encoder_depth=1,
            decoder_depth=1,
            num_heads=4,
            dropout=0.0,
        )
        ckpt_path = tmp_path / "oat.pt"
        normalization = {
            "mean": [0.5] * 8,
            "std": [2.0] * 8,
        }
        torch.save(
            {
                "model_state_dict": tokenizer.state_dict(),
                "tokenizer": {
                    "input_dim": 8,
                    "input_horizon": 8,
                    "emb_dim": 16,
                    "latent_dim": 4,
                    "num_registers": 4,
                    "fsq_levels": [8, 5, 5, 5],
                    "encoder_depth": 1,
                    "decoder_depth": 1,
                    "num_heads": 4,
                },
                "normalization": normalization,
            },
            ckpt_path,
        )

        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_oat=True,
            oat_num_registers=4,
            oat_checkpoint_path=str(ckpt_path),
            freeze_oat=True,
            supervise_oat_tokens=True,
            lambda_oat_token=1.0,
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")

        assert head_oat.oat.encoder.emb_dim == 16
        assert head_oat.query_embed.embedding_dim == 64
        assert head_oat.oat_uses_normalization
        assert not any(param.requires_grad for param in head_oat.oat.parameters())

        target_inv = torch.randn(self.B, 8, 8)
        resolved = head_oat._resolve_target_oat_tokens(
            target_oat_tokens=None,
            target_invariants=target_inv,
            device=target_inv.device,
        )
        expected = head_oat.oat.tokenize((target_inv - 0.5) / 2.0)
        assert torch.equal(resolved, expected)

    def test_pretrained_oat_decode_denormalizes_outputs(self, tmp_path):
        tokenizer = OATTokenizer(
            input_dim=8,
            input_horizon=8,
            emb_dim=16,
            latent_dim=4,
            num_registers=4,
            fsq_levels=[8, 5, 5, 5],
            encoder_depth=1,
            decoder_depth=1,
            num_heads=4,
            dropout=0.0,
        )
        ckpt_path = tmp_path / "oat.pt"
        normalization = {
            "mean": [1.0] * 8,
            "std": [0.25] * 8,
        }
        torch.save(
            {
                "model_state_dict": tokenizer.state_dict(),
                "tokenizer": {
                    "input_dim": 8,
                    "input_horizon": 8,
                    "emb_dim": 16,
                    "latent_dim": 4,
                    "num_registers": 4,
                    "fsq_levels": [8, 5, 5, 5],
                    "encoder_depth": 1,
                    "decoder_depth": 1,
                    "num_heads": 4,
                },
                "normalization": normalization,
            },
            ckpt_path,
        )
        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_oat=True,
            oat_checkpoint_path=str(ckpt_path),
            freeze_oat=True,
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")

        tokens = torch.zeros(self.B, 4, dtype=torch.long)
        recon_raw, _ = head_oat._decode_oat_tokens(tokens)
        recon_normalized = head_oat.oat.detokenize(tokens)

        assert torch.allclose(recon_raw, recon_normalized * 0.25 + 1.0)

    def test_predict_can_use_discrete_oat_tokens_for_execution(self):
        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_oat=True,
            oat_num_registers=4,
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")

        head_oat.eval()
        vlm_hidden = torch.randn(self.B, self.S, 128)
        initial_pose = {
            "position": torch.zeros(self.B, 3),
            "quaternion": torch.tensor([[1.0, 0, 0, 0]] * self.B),
        }
        result = head_oat.predict(
            vlm_hidden,
            initial_pose=initial_pose,
            use_discrete_tokens=True,
            token_prefix_keep_k=2,
            emit_oat_tokens=True,
        )
        assert "pred_oat_tokens" in result
        assert result["pred_oat_tokens"].shape == (self.B, 4)
        assert "pred_invariants_token_recon" in result
        assert result["pred_invariants_token_recon"].shape == (self.B, 8, 8)
        assert "pred_invariants_execute" in result
        assert torch.allclose(result["pred_invariants_execute"], result["pred_invariants_token_recon"])
        assert "token_prefix_keep_k" in result
        assert result["token_prefix_keep_k"].tolist() == [2] * self.B
        assert "positions" in result
        assert "quaternions" in result

    def test_predict_can_tokenize_continuous_head_even_with_token_policy(self):
        head_oat = DHBActionHead(
            vlm_dim=128,
            action_horizon=8,
            invariant_dim=8,
            hidden_dim=64,
            num_heads=4,
            num_layers=2,
            use_oat=True,
            oat_num_registers=4,
            supervise_oat_tokens=True,
        )
        if head_oat.oat is None:
            pytest.skip("OAT tokenizer not available")

        head_oat.eval()
        vlm_hidden = torch.randn(self.B, self.S, 128)
        result = head_oat.predict(
            vlm_hidden,
            use_discrete_tokens=True,
            emit_oat_tokens=True,
            tokenize_continuous_head=True,
        )
        expected_tokens = head_oat._tokenize_oat_invariants(result["pred_invariants"])

        assert torch.equal(result["pred_oat_tokens"], expected_tokens)
        assert torch.allclose(result["pred_invariants_execute"], result["pred_invariants_token_recon"])
