from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "libero_postcontact_correction_plan.py"
SPEC = importlib.util.spec_from_file_location("libero_postcontact_correction_plan", SCRIPT_PATH)
correction_plan = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(correction_plan)


def test_scripted_closed_correction_actions_clip_and_hold_closed() -> None:
    actions = correction_plan.scripted_closed_correction_actions(
        [0.12, -0.04, 0.03],
        correction_steps=4,
        max_action_step_mm=20.0,
        include_z=False,
    )

    assert actions.shape == (4, 7)
    np.testing.assert_allclose(actions[:, 0], [-0.02, -0.02, -0.02, -0.02])
    np.testing.assert_allclose(actions[:, 1], [0.01, 0.01, 0.01, 0.01])
    np.testing.assert_allclose(actions[:, 2], [0.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(actions[:, 6], [1.0, 1.0, 1.0, 1.0])

    gained = correction_plan.scripted_closed_correction_actions(
        [0.04, 0.0, 0.0],
        correction_steps=4,
        max_action_step_mm=20.0,
        include_z=False,
        correction_gain=2.0,
    )
    np.testing.assert_allclose(gained[:, 0], [-0.02, -0.02, -0.02, -0.02])


def test_build_plan_rows_marks_available_sim_state() -> None:
    rollout = {
        "checkpoint": "ckpt.pt",
        "task_suite_name": "libero_spatial",
        "task_id": 0,
        "episode_idx": 1,
        "offset_cm": 3.0,
        "step_trace": [
            {"step": 4, "sim_state": [0.1, 0.2, 0.3]},
            {"step": 5},
        ],
    }
    requests = [
        {
            "rollout_json": "rollout.json",
            "step": 4,
            "object_target_field": "bowl_minus_plate",
            "object_minus_target_m": [0.08, 0.0, 0.0],
            "desired_object_delta_to_target_m": [-0.08, -0.0, -0.0],
            "object_target_xy_mm": 80.0,
        },
        {
            "rollout_json": "other.json",
            "step": 4,
            "object_minus_target_m": [1.0, 0.0, 0.0],
        },
    ]

    rows = correction_plan.build_plan_rows(
        rollout,
        requests,
        source_rollout_json="rollout.json",
        correction_steps=8,
        max_action_step_mm=10.0,
        include_z=False,
    )

    assert len(rows) == 1
    assert rows[0]["has_step_trace_entry"] is True
    assert rows[0]["has_sim_state"] is True
    assert rows[0]["sim_state_dim"] == 3
    assert rows[0]["needs_sim_state_capture"] is False
    np.testing.assert_allclose(rows[0]["expected_ee_delta_m"], [-0.08, 0.0, 0.0], atol=1e-7)


def test_build_plan_rows_flags_missing_sim_state_capture() -> None:
    rollout = {"step_trace": [{"step": 5}]}
    requests = [
        {
            "rollout_json": "rollout.json",
            "step": 5,
            "object_minus_target_m": [0.04, 0.0, 0.0],
        }
    ]

    rows = correction_plan.build_plan_rows(
        rollout,
        requests,
        source_rollout_json="rollout.json",
        correction_steps=4,
        max_action_step_mm=10.0,
        include_z=False,
    )

    assert len(rows) == 1
    assert rows[0]["has_step_trace_entry"] is True
    assert rows[0]["has_sim_state"] is False
    assert rows[0]["needs_sim_state_capture"] is True
