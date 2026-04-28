from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "libero_rollout_transport_miner.py"
SPEC = importlib.util.spec_from_file_location("libero_rollout_transport_miner", SCRIPT_PATH)
transport_miner = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(transport_miner)


def test_mines_transport_progress_after_physical_close() -> None:
    rollout = {
        "checkpoint": "ckpt.pt",
        "task_id": 0,
        "episode_idx": 1,
        "offset_cm": 3.0,
        "close_step": 2,
        "reopen_step": 6,
        "step_trace": [
            {"step": 0, "bowl_minus_plate": [0.02, 0.0, 0.0], "observed_state_gripper": 0.04},
            {"step": 2, "bowl_minus_plate": [0.12, 0.0, 0.0], "observed_state_gripper": 0.04},
            {
                "step": 3,
                "bowl_minus_plate": [0.10, 0.0, 0.0],
                "bowl_pos": [0.10, 0.0, 0.0],
                "ee_pos": [0.00, 0.0, 0.0],
                "observed_state_gripper": 0.01,
            },
            {
                "step": 4,
                "bowl_minus_plate": [0.06, 0.0, 0.0],
                "bowl_pos": [0.06, 0.0, 0.0],
                "ee_pos": [-0.04, 0.0, 0.0],
                "observed_state_gripper": 0.01,
            },
            {
                "step": 5,
                "bowl_minus_plate": [0.02, 0.0, 0.0],
                "bowl_pos": [0.02, 0.0, 0.0],
                "ee_pos": [-0.08, 0.0, 0.0],
                "observed_state_gripper": 0.01,
            },
        ],
    }

    result = transport_miner.mine_transport_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        target_xy_tolerance_mm=45.0,
        target_z_min_mm=-20.0,
        target_z_max_mm=40.0,
        gripper_qpos_threshold=0.015,
        correction_top_k=2,
    )

    assert result["physical_close_step"] == 3
    assert result["xy_at_physical_close_mm"] == 100.0
    assert result["best_target_xy_mm"] == 20.0
    assert result["target_progress_mm"] == 80.0
    assert result["target_reached"] is True
    assert result["counts"]["target_reached"] == 1
    assert result["transport_closed_steps"] == 3
    assert np.isclose(result["object_motion_path_mm"], 80.0)
    assert np.isclose(result["ee_motion_path_during_transport_mm"], 80.0)
    assert np.isclose(result["object_ee_coupling_ratio"], 1.0)


def test_exports_correction_requests_when_target_never_reached() -> None:
    rollout = {
        "checkpoint": "ckpt.pt",
        "task_id": 0,
        "episode_idx": 1,
        "offset_cm": 3.0,
        "close_step": 1,
        "step_trace": [
            {"step": 1, "bowl_minus_plate": [0.12, 0.0, 0.0], "observed_state_gripper": 0.04},
            {"step": 2, "bowl_minus_plate": [0.11, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 3, "bowl_minus_plate": [0.09, 0.02, 0.0], "observed_state_gripper": 0.01},
            {"step": 4, "bowl_minus_plate": [0.13, 0.0, 0.0], "observed_state_gripper": 0.01},
        ],
    }

    result = transport_miner.mine_transport_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        target_xy_tolerance_mm=45.0,
        target_z_min_mm=-20.0,
        target_z_max_mm=40.0,
        gripper_qpos_threshold=0.015,
        correction_top_k=2,
    )

    assert result["target_reached"] is False
    assert result["needs_transport_correction"] is True
    assert len(result["correction_requests"]) == 2
    assert result["correction_requests"][0]["step"] == 3
    assert result["correction_requests"][0]["desired_object_delta_to_target_m"] == [
        -0.09,
        -0.02,
        -0.0,
    ]


def test_builds_policy_manifest_rows_with_normalized_corrective_actions() -> None:
    rollout = {
        "checkpoint": "ckpt.pt",
        "task_id": 0,
        "episode_idx": 1,
        "offset_cm": 3.0,
        "close_step": 1,
        "policy_trace": [
            {
                "chunk_index": 0,
                "t_start": 0,
                "n_use": 4,
                "npz_path": "chunk_0000_t0000.npz",
                "rel_live": {"bowl_minus_plate": [0.12, 0.0, 0.0]},
            }
        ],
        "step_trace": [
            {"step": 0, "bowl_minus_plate": [0.12, 0.0, 0.0], "observed_state_gripper": 0.04},
            {"step": 1, "bowl_minus_plate": [0.12, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 2, "bowl_minus_plate": [0.08, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 3, "bowl_minus_plate": [0.06, 0.0, 0.0], "observed_state_gripper": 0.01},
        ],
    }
    result = transport_miner.mine_transport_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        target_xy_tolerance_mm=45.0,
        target_z_min_mm=-20.0,
        target_z_max_mm=40.0,
        gripper_qpos_threshold=0.015,
        correction_top_k=2,
    )
    action_stats = {
        "min": [-0.02, -0.02, -0.02, -1.0, -1.0, -1.0, 0.0],
        "max": [0.02, 0.02, 0.02, 1.0, 1.0, 1.0, 1.0],
        "mask": [True, True, True, True, True, True, False],
    }

    rows = transport_miner.policy_manifest_rows(
        rollout,
        result,
        action_stats=action_stats,
        correction_steps=4,
        max_action_step_mm=18.0,
        include_z=False,
    )

    assert len(rows) == 1
    assert rows[0]["npz_path"] == "chunk_0000_t0000.npz"
    assert rows[0]["target_action_normalized"] is True
    assert rows[0]["train_mask"] == [0.0, 1.0, 1.0, 1.0]
    assert rows[0]["target_close"] == [1.0, 1.0, 1.0, 1.0]
    assert rows[0]["target_gripper_state"] == [[1.0], [1.0], [1.0], [1.0]]
    assert rows[0]["target_action"][1][0] == -0.8999999761581421
    assert rows[0]["target_action"][2][0] == -0.8999999761581421
    assert rows[0]["target_action"][3][0] == -0.75
    np.testing.assert_allclose(rows[0]["target_object_target"][2], [0.08, 0.0, 0.0])


def test_ignores_preclose_target_ready_states() -> None:
    rollout = {
        "close_step": 2,
        "step_trace": [
            {"step": 0, "bowl_minus_plate": [0.01, 0.0, 0.0], "observed_state_gripper": 0.04},
            {"step": 2, "bowl_minus_plate": [0.10, 0.0, 0.0], "observed_state_gripper": 0.01},
        ],
    }

    result = transport_miner.mine_transport_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        target_xy_tolerance_mm=45.0,
        target_z_min_mm=-20.0,
        target_z_max_mm=40.0,
        gripper_qpos_threshold=0.015,
        correction_top_k=1,
    )

    assert result["samples"][0]["label"] == "background"
    assert result["counts"].get("target_reached", 0) == 0
    assert result["best_target_step"] == 2


def test_transport_candidates_require_commanded_closed_gripper() -> None:
    rollout = {
        "close_step": 1,
        "reopen_step": 2,
        "step_trace": [
            {
                "step": 1,
                "bowl_minus_plate": [0.10, 0.0, 0.0],
                "observed_state_gripper": 0.01,
                "effective_grip": 1.0,
            },
            {
                "step": 2,
                "bowl_minus_plate": [0.08, 0.0, 0.0],
                "observed_state_gripper": 0.01,
                "effective_grip": -1.0,
            },
        ],
    }

    result = transport_miner.mine_transport_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        target_xy_tolerance_mm=45.0,
        target_z_min_mm=-20.0,
        target_z_max_mm=55.0,
        gripper_qpos_threshold=0.015,
        correction_top_k=5,
    )

    assert result["counts"]["closed_transport_candidate"] == 1
    assert result["counts"]["post_close_background"] == 1
    assert result["correction_requests"][0]["step"] == 1
    assert result["samples"][1]["observed_closed"] is True
    assert result["samples"][1]["commanded_closed"] is False
    assert result["samples"][1]["transport_closed"] is False


def test_can_select_earliest_correction_requests() -> None:
    rollout = {
        "close_step": 1,
        "step_trace": [
            {"step": 1, "bowl_minus_plate": [0.12, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 2, "bowl_minus_plate": [0.08, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 3, "bowl_minus_plate": [0.07, 0.0, 0.0], "observed_state_gripper": 0.01},
        ],
    }

    result = transport_miner.mine_transport_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        target_xy_tolerance_mm=45.0,
        target_z_min_mm=-20.0,
        target_z_max_mm=55.0,
        gripper_qpos_threshold=0.015,
        correction_top_k=2,
        correction_sort="earliest",
    )

    assert [row["step"] for row in result["correction_requests"]] == [1, 2]
