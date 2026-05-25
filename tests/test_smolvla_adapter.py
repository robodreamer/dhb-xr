"""Tests for the SmolVLA + DHB adapter."""

import pytest
import numpy as np
from scipy.spatial.transform import Rotation

torch = pytest.importorskip("torch")

from dhb_xr.integration.vla.smolvla_adapter import SmolVLADHBAdapter


class _FakeHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def predict(self, vlm_features, state=None, initial_pose=None):
        self.calls.append(
            {
                "vlm_features": vlm_features,
                "state": state,
                "initial_pose": initial_pose,
            }
        )
        batch = vlm_features.shape[0]
        return {
            "pred_invariants": torch.ones(batch, 5, 8),
            "positions": torch.zeros(batch, 5, 3),
            "quaternions": torch.zeros(batch, 5, 4),
        }


def test_predict_invariants_forwards_state():
    """Regression test for the old SmolVLA eval bug: adapter must forward state."""
    adapter = SmolVLADHBAdapter(vlm_dim=32, action_horizon=5, hidden_dim=16, num_heads=4, num_layers=1)
    fake_head = _FakeHead()
    adapter.dhb_head = fake_head

    vlm_features = torch.randn(2, 7, 32)
    state = torch.randn(2, 8)

    pred = adapter.predict_invariants(vlm_features, state=state)

    assert pred.shape == (2, 5, 8)
    assert len(fake_head.calls) == 1
    assert fake_head.calls[0]["state"] is state
    assert fake_head.calls[0]["initial_pose"] is None


def test_predict_trajectory_forwards_state_and_pose():
    """Adapter trajectory prediction must forward both state and initial pose."""
    adapter = SmolVLADHBAdapter(vlm_dim=32, action_horizon=5, hidden_dim=16, num_heads=4, num_layers=1)
    fake_head = _FakeHead()
    adapter.dhb_head = fake_head

    vlm_features = torch.randn(1, 6, 32)
    state = torch.randn(1, 8)
    initial_pose = {
        "position": torch.randn(1, 3),
        "quaternion": torch.randn(1, 4),
    }

    pred = adapter.predict_trajectory(vlm_features, initial_pose=initial_pose, state=state)

    assert pred["positions"].shape == (1, 5, 3)
    assert pred["quaternions"].shape == (1, 5, 4)
    assert pred["pred_invariants"].shape == (1, 5, 8)
    assert len(fake_head.calls) == 1
    assert fake_head.calls[0]["state"] is state
    assert fake_head.calls[0]["initial_pose"] is initial_pose


def test_observation_helpers_build_state_and_initial_pose():
    """Observation conversion must preserve EE state and quaternion ordering."""
    angle = np.deg2rad(30.0)
    quat_xyzw = Rotation.from_rotvec([0.0, 0.0, angle]).as_quat().astype(np.float32)
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    obs = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": quat_wxyz,
        "robot0_gripper_qpos": np.array([0.04, 0.05], dtype=np.float32),
    }

    state = SmolVLADHBAdapter.observation_to_state(obs)
    init_pose = SmolVLADHBAdapter.observation_to_initial_pose(obs)

    assert state.shape == (1, 8)
    assert torch.allclose(state[0, :3], torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    assert torch.allclose(
        state[0, 3:6], torch.tensor([0.0, 0.0, angle], dtype=torch.float32), atol=1e-6
    )
    assert torch.allclose(state[0, 6:], torch.tensor([0.04, 0.05], dtype=torch.float32))

    assert torch.allclose(init_pose["position"][0], torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    assert torch.allclose(init_pose["quaternion"][0], torch.from_numpy(quat_wxyz))
