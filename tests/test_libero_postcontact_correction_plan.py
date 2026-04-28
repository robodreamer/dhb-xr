from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "libero_postcontact_correction_plan.py"
SPEC = importlib.util.spec_from_file_location("libero_postcontact_correction_plan", SCRIPT_PATH)
correction_plan = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(correction_plan)

ROLLOUT_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "libero_postcontact_correction_rollout.py"
ROLLOUT_SPEC = importlib.util.spec_from_file_location(
    "libero_postcontact_correction_rollout",
    ROLLOUT_SCRIPT_PATH,
)
correction_rollout = importlib.util.module_from_spec(ROLLOUT_SPEC)
assert ROLLOUT_SPEC and ROLLOUT_SPEC.loader
ROLLOUT_SPEC.loader.exec_module(correction_rollout)


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


def test_rollout_attachment_name_helpers() -> None:
    assert correction_rollout._object_token_from_target_field("bowl_minus_plate") == "bowl"
    assert correction_rollout._object_token_from_target_field("object") == "object"
    assert correction_rollout._body_name_from_ref("body:akita_black_bowl_1_main") == "akita_black_bowl_1_main"
    assert correction_rollout._body_name_from_ref("geom:akita_black_bowl_1_g0") is None


def test_rollout_lift_transport_profile_lifts_before_lowering() -> None:
    lift_action = correction_rollout._closed_loop_action(
        {"bowl_minus_plate": [0.10, 0.0, 0.0]},
        "bowl_minus_plate",
        remaining_steps=20,
        max_action_step_mm=10.0,
        correction_gain=1.0,
        include_z=False,
        transport_profile="lift_transport_place",
        lift_height_mm=40.0,
        place_z_mm=5.0,
        lower_xy_mm=45.0,
        lift_steps=4,
    )

    np.testing.assert_allclose(lift_action[:3], [-0.005, 0.0, 0.01], atol=1e-7)
    assert lift_action[6] == 1.0

    lower_action = correction_rollout._closed_loop_action(
        {"bowl_minus_plate": [0.02, 0.0, 0.04]},
        "bowl_minus_plate",
        remaining_steps=10,
        max_action_step_mm=10.0,
        correction_gain=1.0,
        include_z=False,
        transport_profile="lift_transport_place",
        lift_height_mm=40.0,
        place_z_mm=5.0,
        lower_xy_mm=45.0,
        lift_steps=4,
    )

    np.testing.assert_allclose(lower_action[:3], [-0.002, 0.0, -0.0035], atol=1e-7)
    assert lower_action[6] == 1.0


def test_rollout_policy_trace_export_preserves_lift_transport_targets(tmp_path: Path) -> None:
    records = [
        {
            "agentview_image": np.full((8, 8, 3), 10 + idx, dtype=np.uint8),
            "wrist_image": np.full((8, 8, 3), 20 + idx, dtype=np.uint8),
            "state_window": np.full((4, 8), idx, dtype=np.float32),
            "ee_pos_live": np.array([idx * 0.01, 0.0, 0.9], dtype=np.float32),
            "raw_action": np.array(raw_action, dtype=np.float32),
            "object_target": np.array([0.10 - idx * 0.02, 0.01, 0.04], dtype=np.float32),
            "object_to_eef": np.array([-0.02, -0.04, -0.03], dtype=np.float32),
            "target_to_eef": np.array([0.08, -0.03, -0.03], dtype=np.float32),
            "target_xy_mm": 100.0 - idx * 10.0,
        }
        for idx, raw_action in enumerate(
            [
                [0.0, 0.0, 0.008, 0.0, 0.0, 0.0, 1.0],
                [-0.005, 0.002, 0.004, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    ]
    action_stats = {
        "min": [-0.01, -0.01, -0.01, -1.0, -1.0, -1.0, -1.0],
        "max": [0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 1.0],
        "mask": [True, True, True, True, True, True, True],
    }
    manifest_rows: list[dict] = []

    correction_rollout._export_policy_trace_chunks(
        row={
            "source_rollout_json": "rollout.json",
            "step": 42,
            "checkpoint": "ckpt.pt",
            "task_id": 0,
            "episode_idx": 1,
            "offset_cm": 3.0,
        },
        result={"target_reached_step": 7, "attach_object_to_ee": True},
        records=records,
        policy_trace_dir=tmp_path / "policy_traces",
        manifest_rows=manifest_rows,
        action_stats=action_stats,
        action_horizon=8,
        lift_action_loss_weight=3.0,
        carry_action_loss_weight=1.0,
        lower_action_loss_weight=2.0,
        phase_z_threshold_mm=5.0,
    )

    assert len(manifest_rows) == 1
    row = manifest_rows[0]
    assert row["target_action_normalized"] is True
    assert row["target_reached"] is True
    assert row["train_mask"] == [1.0, 1.0]
    assert row["target_gripper_state"] == [[1.0], [1.0]]
    assert row["target_transport_phase"] == ["lift", "carry"]
    assert row["action_loss_weight"] == [3.0, 1.0]
    np.testing.assert_allclose(row["target_action"][0][:3], [0.0, 0.0, 0.8], atol=1e-6)
    np.testing.assert_allclose(row["target_action"][1][:3], [-0.5, 0.2, 0.4], atol=1e-6)
    np.testing.assert_allclose(row["target_action"][1][6], 1.0, atol=1e-6)
    np.testing.assert_allclose(
        row["target_object_pair_to_eef"][0],
        [-0.02, -0.04, -0.03, 0.08, -0.03, -0.03],
        atol=1e-7,
    )
    np.testing.assert_allclose(row["object_target_xy_mm"], [100.0, 90.0], atol=1e-7)

    with np.load(row["npz_path"]) as npz:
        assert npz["agentview_image"].shape == (8, 8, 3)
        assert npz["state_window"].shape == (4, 8)
        np.testing.assert_allclose(
            npz["chunk_positions"],
            [[0.0, 0.0, 0.9], [0.01, 0.0, 0.9]],
            atol=1e-7,
        )
