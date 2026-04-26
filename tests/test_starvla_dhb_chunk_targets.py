import numpy as np
from scipy.spatial.transform import Rotation

from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from starVLA.dataloader.gr00t_lerobot.datasets import (
    build_chunk_local_dhb_targets_from_delta_actions,
)
from starVLA.model.framework.VLM4A.QwenDHB import _extract_loss_mask


def test_chunk_local_dhb_targets_reconstruct_sampled_action_window_prefix():
    horizon = 8
    positions = np.array(
        [[0.01 * i, 0.002 * i * i, 0.5 + 0.001 * i] for i in range(horizon + 1)],
        dtype=np.float32,
    )
    rotations = Rotation.from_euler(
        "xyz",
        [[0.0, 0.01 * i, 0.02 * i] for i in range(horizon + 1)],
    )

    actions = np.zeros((horizon, 7), dtype=np.float32)
    actions[:, :3] = positions[1:] - positions[:-1]
    delta_rot = rotations[:-1].inv() * rotations[1:]
    actions[:, 3:6] = delta_rot.as_rotvec().astype(np.float32)
    actions[:, 6] = 1.0

    current_state = np.zeros(8, dtype=np.float32)
    current_state[:3] = positions[0]
    current_state[3:6] = rotations[0].as_rotvec().astype(np.float32)
    current_state[7] = 1.0

    linear, angular = build_chunk_local_dhb_targets_from_delta_actions(current_state, actions)
    quat_wxyz = rotations[0].as_quat()[[3, 0, 1, 2]]
    decoded = decode_dhb_dr(
        linear,
        angular,
        {"position": positions[0], "quaternion": quat_wxyz},
    )

    assert linear.shape == (horizon, 4)
    assert angular.shape == (horizon, 4)
    np.testing.assert_allclose(decoded["positions"], positions[1 : horizon - 1], atol=1e-6)
    np.testing.assert_allclose(decoded["positions"][0], positions[1], atol=1e-6)


def test_qwendhb_extracts_dhb_action_valid_mask():
    examples = [
        {"dhb.action_valid_mask": np.array([1, 1, 0], dtype=np.float32)},
        {"dhb.action_valid_mask": np.array([1, 0, 0], dtype=np.float32)},
    ]

    mask = _extract_loss_mask(examples)

    assert mask.shape == (2, 3)
    np.testing.assert_allclose(
        mask.numpy(),
        np.array([[1, 1, 0], [1, 0, 0]], dtype=np.float32),
    )
