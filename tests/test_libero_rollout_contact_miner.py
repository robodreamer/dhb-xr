from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "libero_rollout_contact_miner.py"
SPEC = importlib.util.spec_from_file_location("libero_rollout_contact_miner", SCRIPT_PATH)
contact_miner = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(contact_miner)


def test_mines_missed_positive_and_hard_negative_from_rollout_trace() -> None:
    rollout = {
        "checkpoint": "ckpt.pt",
        "task_id": 0,
        "episode_idx": 1,
        "offset_cm": 3.0,
        "close_step": 2,
        "success": False,
        "chunks": [
            {
                "t_start": 0,
                "n_use": 4,
                "close_probs": [0.2, 0.3, 0.8, 0.1],
                "open_probs": [0.0, 0.0, 0.0, 0.0],
            }
        ],
        "step_trace": [
            {
                "step": 0,
                "phase": "default",
                "ee_minus_bowl": [0.0, -0.050, 0.026],
                "desired_grip": -1.0,
                "effective_grip": -1.0,
            },
            {
                "step": 1,
                "phase": "default",
                "ee_minus_bowl": [0.06, 0.0, 0.06],
                "desired_grip": -1.0,
                "effective_grip": -1.0,
            },
            {
                "step": 2,
                "phase": "default",
                "ee_minus_bowl": [0.08, 0.0, 0.06],
                "desired_grip": 1.0,
                "effective_grip": 1.0,
            },
        ],
    }

    result = contact_miner.mine_contact_samples(
        rollout,
        rollout_json="rollout.json",
        close_reference_m=[0.0, -0.050, 0.026],
        ee_object_field="ee_minus_bowl",
        positive_tolerance_mm=10.0,
        positive_component_tolerance_mm=None,
        best_positive_top_k_per_rollout=0,
        best_positive_max_error_mm=float("inf"),
        best_positive_window=0,
        positive_prob_margin=0.65,
        hard_negative_prob=0.55,
        near_object_hard_negative_xy_mm=None,
        near_object_hard_negative_z_min_mm=float("-inf"),
        near_object_hard_negative_z_max_mm=float("inf"),
    )

    assert result["counts"]["missed_positive"] == 1
    assert result["counts"]["hard_negative"] == 1
    assert result["best_close_affordance"]["step"] == 0
    assert result["highest_confidence_hard_negative"]["step"] == 2


def test_builds_policy_manifest_rows_from_mined_labels() -> None:
    rollout = {
        "checkpoint": "ckpt.pt",
        "task_id": 0,
        "episode_idx": 1,
        "offset_cm": 3.0,
        "chunks": [
            {
                "t_start": 4,
                "n_use": 4,
                "close_probs": [0.2, 0.7, 0.1, 0.8],
            }
        ],
        "policy_trace": [
            {
                "chunk_index": 0,
                "t_start": 4,
                "n_use": 4,
                "npz_path": "chunk_0000_t0004.npz",
                "rel_live": {"ee_minus_bowl": [0.0, -0.05, 0.026]},
            }
        ],
        "step_trace": [
            {"step": 4, "ee_minus_bowl": [0.0, -0.05, 0.026]},
            {"step": 5, "ee_minus_bowl": [0.08, 0.0, 0.06]},
            {"step": 6, "ee_minus_bowl": [0.10, 0.0, 0.06]},
            {"step": 7, "ee_minus_bowl": [0.09, 0.0, 0.06]},
        ],
    }
    result = contact_miner.mine_contact_samples(
        rollout,
        rollout_json="rollout.json",
        close_reference_m=[0.0, -0.050, 0.026],
        ee_object_field="ee_minus_bowl",
        positive_tolerance_mm=10.0,
        positive_component_tolerance_mm=None,
        best_positive_top_k_per_rollout=0,
        best_positive_max_error_mm=float("inf"),
        best_positive_window=0,
        positive_prob_margin=0.65,
        hard_negative_prob=0.55,
        near_object_hard_negative_xy_mm=None,
        near_object_hard_negative_z_min_mm=float("-inf"),
        near_object_hard_negative_z_max_mm=float("inf"),
    )

    rows = contact_miner.policy_manifest_rows(rollout, result)

    assert len(rows) == 1
    assert rows[0]["npz_path"] == "chunk_0000_t0004.npz"
    assert rows[0]["target_close"] == [1.0, 0.0, 0.0, 0.0]
    assert rows[0]["train_mask"] == [1.0, 1.0, 0.0, 1.0]
    assert rows[0]["hard_negative_mask"] == [0.0, 1.0, 0.0, 1.0]


def test_component_tolerance_filters_broad_vector_positive_to_hard_negative() -> None:
    rollout = {
        "chunks": [
            {
                "t_start": 0,
                "n_use": 2,
                "close_probs": [0.2, 0.2],
            }
        ],
        "step_trace": [
            {
                "step": 0,
                "ee_minus_bowl": [0.020, -0.030, 0.040],
                "desired_grip": -1.0,
                "effective_grip": -1.0,
            },
            {
                "step": 1,
                "ee_minus_bowl": [0.000, -0.049, 0.027],
                "desired_grip": -1.0,
                "effective_grip": -1.0,
            },
        ],
    }

    result = contact_miner.mine_contact_samples(
        rollout,
        rollout_json="rollout.json",
        close_reference_m=[0.0, -0.050, 0.026],
        ee_object_field="ee_minus_bowl",
        positive_tolerance_mm=45.0,
        positive_component_tolerance_mm=[10.0, 10.0, 10.0],
        best_positive_top_k_per_rollout=0,
        best_positive_max_error_mm=float("inf"),
        best_positive_window=0,
        positive_prob_margin=0.65,
        hard_negative_prob=0.55,
        near_object_hard_negative_xy_mm=45.0,
        near_object_hard_negative_z_min_mm=0.0,
        near_object_hard_negative_z_max_mm=80.0,
    )

    assert result["samples"][0]["label"] == "hard_negative"
    assert result["samples"][0]["is_near_object_hard_negative"] is True
    assert result["samples"][1]["label"] == "missed_positive"


def test_best_positive_relabeling_can_override_near_object_hard_negative() -> None:
    rollout = {
        "chunks": [
            {
                "t_start": 0,
                "n_use": 2,
                "close_probs": [0.2, 0.2],
            }
        ],
        "step_trace": [
            {
                "step": 0,
                "ee_minus_bowl": [0.020, -0.030, 0.040],
                "desired_grip": -1.0,
                "effective_grip": -1.0,
            },
            {
                "step": 1,
                "ee_minus_bowl": [0.080, 0.000, 0.040],
                "desired_grip": -1.0,
                "effective_grip": -1.0,
            },
        ],
    }

    result = contact_miner.mine_contact_samples(
        rollout,
        rollout_json="rollout.json",
        close_reference_m=[0.0, -0.050, 0.026],
        ee_object_field="ee_minus_bowl",
        positive_tolerance_mm=10.0,
        positive_component_tolerance_mm=[10.0, 10.0, 10.0],
        best_positive_top_k_per_rollout=1,
        best_positive_max_error_mm=40.0,
        best_positive_window=0,
        positive_prob_margin=0.65,
        hard_negative_prob=0.55,
        near_object_hard_negative_xy_mm=60.0,
        near_object_hard_negative_z_min_mm=0.0,
        near_object_hard_negative_z_max_mm=80.0,
    )

    assert result["samples"][0]["label"] == "missed_positive"
    assert result["samples"][0]["is_forced_best_positive"] is True
    assert result["samples"][0]["is_near_object_hard_negative"] is False
    assert result["samples"][1]["label"] == "background"
