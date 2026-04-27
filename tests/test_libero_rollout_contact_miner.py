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
        positive_prob_margin=0.65,
        hard_negative_prob=0.55,
    )

    assert result["counts"]["missed_positive"] == 1
    assert result["counts"]["hard_negative"] == 1
    assert result["best_close_affordance"]["step"] == 0
    assert result["highest_confidence_hard_negative"]["step"] == 2
