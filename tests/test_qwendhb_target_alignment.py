from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
STARVLA_ROOT = REPO_ROOT / "vendor" / "starVLA"
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from starVLA.model.framework.VLM4A.QwenDHB import (
    _extract_gt_contact_phase,
    _extract_gt_gripper_state,
    _extract_gt_object_target,
    _slice_temporal_target,
)


def test_slice_temporal_target_keeps_future_chunk_by_default() -> None:
    target = torch.arange(10, dtype=torch.long).reshape(1, 10)

    aligned = _slice_temporal_target(target, window_size=4)

    torch.testing.assert_close(aligned, torch.tensor([[6, 7, 8, 9]]))


def test_slice_temporal_target_can_broadcast_current_label() -> None:
    target = torch.arange(10, dtype=torch.long).reshape(1, 10)

    aligned = _slice_temporal_target(target, window_size=4, alignment="current")

    torch.testing.assert_close(aligned, torch.tensor([[6, 6, 6, 6]]))


def test_slice_temporal_target_future_preserves_late_contact_positive() -> None:
    target = torch.tensor([[[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0]]])

    aligned = _slice_temporal_target(target, window_size=8, alignment="future")

    assert aligned.shape == (1, 8, 1)
    torch.testing.assert_close(aligned[0, :, 0], torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))


def test_slice_temporal_target_broadcasts_current_multichannel_label() -> None:
    target = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)

    aligned = _slice_temporal_target(target, window_size=4, alignment="current")

    assert aligned.shape == (2, 4, 3)
    torch.testing.assert_close(aligned[:, 0], target[:, 2])
    torch.testing.assert_close(aligned[:, -1], target[:, 2])


def test_slice_temporal_target_can_broadcast_current_prefix_only() -> None:
    target = torch.arange(10, dtype=torch.long).reshape(1, 10)

    aligned = _slice_temporal_target(
        target,
        window_size=6,
        alignment="current_prefix",
        current_prefix_steps=3,
    )

    torch.testing.assert_close(aligned, torch.tensor([[4, 4, 4, 7, 8, 9]]))


def test_slice_temporal_target_rejects_current_prefix_without_steps() -> None:
    target = torch.arange(10, dtype=torch.long).reshape(1, 10)

    with pytest.raises(ValueError, match="current_prefix_steps"):
        _slice_temporal_target(target, window_size=4, alignment="current_prefix")


def test_slice_temporal_target_rejects_unknown_alignment() -> None:
    target = torch.zeros(1, 4)

    with pytest.raises(ValueError, match="target_alignment"):
        _slice_temporal_target(target, window_size=4, alignment="later")


def test_extract_gt_object_target_can_concatenate_configured_fields() -> None:
    examples = [
        {
            "dhb.object_to_eef_pos": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "dhb.target_to_eef_pos": [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
        },
        {
            "dhb.object_to_eef_pos": [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            "dhb.target_to_eef_pos": [[-7.0, -8.0, -9.0], [-10.0, -11.0, -12.0]],
        },
    ]

    target = _extract_gt_object_target(
        examples,
        fields=["dhb.object_to_eef_pos", "dhb.target_to_eef_pos"],
    )

    assert target is not None
    assert target.shape == (2, 2, 6)
    torch.testing.assert_close(
        target[0, 0],
        torch.tensor([1.0, 2.0, 3.0, -1.0, -2.0, -3.0]),
    )


def test_configured_gripper_and_phase_fields_override_defaults() -> None:
    examples = [
        {
            "dhb.gripper_state": [0, 0, 0],
            "dhb.gripper_phase": [0, 0, 0],
            "dhb.object_release_gripper_state": [0, 1, 0],
            "dhb.object_release_phase": [0, 1, 2],
        },
        {
            "dhb.gripper_state": [0, 0, 0],
            "dhb.gripper_phase": [0, 0, 0],
            "dhb.object_release_gripper_state": [1, 1, 0],
            "dhb.object_release_phase": [1, 1, 2],
        },
    ]

    gripper = _extract_gt_gripper_state(
        examples,
        field="dhb.object_release_gripper_state",
    )
    phase = _extract_gt_contact_phase(
        examples,
        field="dhb.object_release_phase",
    )

    torch.testing.assert_close(gripper, torch.tensor([[0, 1, 0], [1, 1, 0]]))
    torch.testing.assert_close(phase, torch.tensor([[0, 1, 2], [1, 1, 2]]))
