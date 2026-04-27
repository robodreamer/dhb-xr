from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "libero_rollout_release_miner.py"
SPEC = importlib.util.spec_from_file_location("libero_rollout_release_miner", SCRIPT_PATH)
release_miner = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(release_miner)


def test_mines_release_positive_and_closed_hard_negative_after_physical_close() -> None:
    rollout = {
        "checkpoint": "ckpt.pt",
        "task_id": 0,
        "episode_idx": 1,
        "offset_cm": 3.0,
        "close_step": 2,
        "chunks": [
            {
                "t_start": 0,
                "n_use": 5,
                "open_probs": [0.1, 0.1, 0.1, 0.1, 0.7],
            }
        ],
        "step_trace": [
            {
                "step": 0,
                "bowl_minus_plate": [0.01, 0.01, 0.0],
                "observed_state_gripper": 0.04,
            },
            {
                "step": 2,
                "bowl_minus_plate": [0.12, 0.0, 0.0],
                "observed_state_gripper": 0.04,
            },
            {
                "step": 3,
                "bowl_minus_plate": [0.10, 0.0, 0.0],
                "observed_state_gripper": 0.01,
            },
            {
                "step": 4,
                "bowl_minus_plate": [0.02, 0.01, 0.0],
                "observed_state_gripper": 0.01,
            },
        ],
    }

    result = release_miner.mine_release_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        release_xy_tolerance_mm=40.0,
        release_z_min_mm=-20.0,
        release_z_max_mm=40.0,
        best_release_top_k_per_rollout=0,
        best_release_max_xy_mm=float("inf"),
        best_release_window=0,
        open_prob_threshold=0.5,
        gripper_qpos_threshold=0.015,
    )

    assert result["physical_close_step"] == 3
    assert result["samples"][0]["label"] == "background"
    assert result["samples"][2]["label"] == "closed_hard_negative"
    assert result["samples"][3]["label"] == "release_positive"
    assert result["samples"][3]["target_closed"] == 0.0


def test_builds_policy_manifest_rows_for_dense_gripper_state() -> None:
    rollout = {
        "checkpoint": "ckpt.pt",
        "task_id": 0,
        "episode_idx": 1,
        "offset_cm": 3.0,
        "close_step": 0,
        "policy_trace": [
            {
                "chunk_index": 0,
                "t_start": 0,
                "n_use": 4,
                "npz_path": "chunk_0000_t0000.npz",
                "rel_live": {"bowl_minus_plate": [0.1, 0.0, 0.0]},
            }
        ],
        "step_trace": [
            {"step": 0, "bowl_minus_plate": [0.10, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 1, "bowl_minus_plate": [0.08, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 2, "bowl_minus_plate": [0.02, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 3, "bowl_minus_plate": [0.01, 0.0, 0.0], "observed_state_gripper": 0.01},
        ],
    }
    result = release_miner.mine_release_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        release_xy_tolerance_mm=40.0,
        release_z_min_mm=-20.0,
        release_z_max_mm=40.0,
        best_release_top_k_per_rollout=0,
        best_release_max_xy_mm=float("inf"),
        best_release_window=0,
        open_prob_threshold=0.5,
        gripper_qpos_threshold=0.015,
    )

    rows = release_miner.policy_manifest_rows(rollout, result)

    assert len(rows) == 1
    assert rows[0]["npz_path"] == "chunk_0000_t0000.npz"
    assert rows[0]["target_close"] == [1.0, 1.0, 0.0, 0.0]
    assert rows[0]["train_mask"] == [1.0, 1.0, 1.0, 1.0]
    assert rows[0]["labels"] == [
        "closed_hard_negative",
        "closed_hard_negative",
        "release_positive",
        "release_positive",
    ]


def test_best_release_relabeling_can_create_corrective_positive() -> None:
    rollout = {
        "close_step": 0,
        "step_trace": [
            {"step": 0, "bowl_minus_plate": [0.07, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 1, "bowl_minus_plate": [0.055, 0.0, 0.0], "observed_state_gripper": 0.01},
            {"step": 2, "bowl_minus_plate": [0.09, 0.0, 0.0], "observed_state_gripper": 0.01},
        ],
    }

    result = release_miner.mine_release_samples(
        rollout,
        rollout_json="rollout.json",
        object_target_field="bowl_minus_plate",
        release_xy_tolerance_mm=40.0,
        release_z_min_mm=-20.0,
        release_z_max_mm=40.0,
        best_release_top_k_per_rollout=1,
        best_release_max_xy_mm=60.0,
        best_release_window=0,
        open_prob_threshold=0.5,
        gripper_qpos_threshold=0.015,
    )

    assert result["samples"][1]["label"] == "release_positive"
    assert result["samples"][1]["is_forced_best_release"] is True
