from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "libero_rollout_affordance_score.py"
)
SPEC = importlib.util.spec_from_file_location("libero_rollout_affordance_score", SCRIPT_PATH)
affordance_score = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(affordance_score)


def test_scores_close_affordance_and_release_from_snapshots() -> None:
    demo_geometry = {
        "events": [
            {
                "event": "close_phys",
                "ee_minus_object_m": [0.0, -0.05, 0.026],
                "object_minus_target_m": [0.13, 0.0, 0.0],
            },
            {
                "event": "close_phys",
                "ee_minus_object_m": [0.004, -0.052, 0.024],
                "object_minus_target_m": [0.13, 0.0, 0.0],
            },
            {
                "event": "open_phys",
                "ee_minus_object_m": [0.0, -0.04, 0.04],
                "object_minus_target_m": [0.02, 0.0, 0.0],
            },
            {
                "event": "open_phys",
                "ee_minus_object_m": [0.0, -0.04, 0.04],
                "object_minus_target_m": [0.018, 0.004, 0.0],
            },
        ]
    }
    rollout = {
        "close_step": 5,
        "reopen_step": 10,
        "success": False,
        "ik_fail_total": 2,
        "min_bowl_xy_m": 0.006,
        "min_bowl_xyz_m": 0.02,
        "close_snapshot": {
            "ee_minus_bowl": [0.003, -0.051, 0.027],
            "bowl_minus_plate": [0.12, 0.0, 0.0],
        },
        "reopen_snapshot": {
            "ee_minus_bowl": [0.0, -0.04, 0.04],
            "bowl_minus_plate": [0.019, 0.002, 0.0],
        },
        "final_snapshot": {
            "ee_minus_bowl": [0.0, -0.04, 0.04],
            "bowl_minus_plate": [0.019, 0.002, 0.0],
        },
        "step_trace": [
            {"step": 0, "ee_minus_bowl": [0.0, 0.0, 0.10], "bowl_minus_plate": [0.13, 0.0, 0.0]},
            {"step": 5, "ee_minus_bowl": [0.003, -0.051, 0.027], "bowl_minus_plate": [0.12, 0.0, 0.0]},
            {"step": 10, "ee_minus_bowl": [0.0, -0.04, 0.04], "bowl_minus_plate": [0.019, 0.002, 0.0]},
        ],
    }

    score = affordance_score.score_rollout(
        rollout,
        demo_geometry,
        close_vector_tolerance_mm=10.0,
        release_xy_tolerance_mm=30.0,
    )

    assert score["events"]["close"]["vector_error_mm"] == pytest.approx(2.236, rel=1e-3)
    assert score["events"]["reopen"]["xy_mm"] == pytest.approx(19.105, rel=1e-3)
    assert score["gates"]["close_affordance_pass"] is True
    assert score["gates"]["release_xy_pass"] is True


def test_falls_back_to_step_trace_when_snapshot_missing() -> None:
    demo_geometry = {
        "events": [
            {
                "event": "close_phys",
                "ee_minus_object_m": [0.0, -0.05, 0.026],
                "object_minus_target_m": [0.13, 0.0, 0.0],
            },
            {
                "event": "open_phys",
                "ee_minus_object_m": [0.0, -0.04, 0.04],
                "object_minus_target_m": [0.02, 0.0, 0.0],
            },
        ]
    }
    rollout = {
        "close_step": 5,
        "reopen_step": None,
        "success": False,
        "step_trace": [
            {"step": 5, "ee_minus_bowl": [0.0, -0.051, 0.026], "bowl_minus_plate": [0.12, 0.0, 0.0]},
        ],
        "final_snapshot": {"bowl_minus_plate": [0.02, 0.0, 0.0]},
    }

    score = affordance_score.score_rollout(rollout, demo_geometry)

    assert score["events"]["close"]["vector_error_mm"] == pytest.approx(1.0)
    assert score["gates"]["has_close"] is True
    assert score["gates"]["has_reopen"] is False
    assert score["gates"]["release_xy_pass"] is True
