from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "starvla_dhb_live_checkpoint_sweep.py"
SPEC = importlib.util.spec_from_file_location("starvla_dhb_live_checkpoint_sweep", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
live_checkpoint_sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live_checkpoint_sweep)


def test_run_case_forwards_disable_gripper_gates(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(cmd, cwd, check, env):
        calls.append(list(cmd))
        save_path = Path(cmd[cmd.index("--save_json") + 1])
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps({"success": False, "chunks": [], "step_trace": []}))

    monkeypatch.setattr(live_checkpoint_sweep.subprocess, "run", fake_run)

    live_checkpoint_sweep._run_case(
        checkpoint=Path("checkpoints/steps_1_pytorch_model.pt"),
        config_yaml="configs/example.yaml",
        task_suite_name="libero_spatial",
        task_id=0,
        episode_idx=1,
        offset_cm=3.0,
        output_dir=tmp_path,
        replan_every=4,
        max_first_offset_mm=35.0,
        max_step_mm=20.0,
        smooth_window=3,
        blend_overlap_alpha=0.6,
        blend_mode="constant",
        commit_prefix_steps=0,
        adaptive_commit_threshold_mm=0.0,
        adaptive_commit_max_steps=0,
        delta_continuation_alpha=0.0,
        delta_continuation_steps=0,
        gripper_source="model",
        init_state_source="benchmark",
        gripper_threshold=0.55,
        gripper_close_threshold=0.55,
        gripper_open_threshold=0.4,
        gripper_transition_debounce_steps=2,
        image_orientation="raw",
        trajectory_source="direct_action",
        direct_action_xyz_scale=(0.85, 1.55, 1.15),
        direct_action_xyz_bias_mm=(0.0, 0.0, 0.0),
        state_history_update_rate="policy",
        state_gripper_source="commanded",
        disable_gripper_gates=True,
        max_policy_steps=240,
        settle_steps=60,
        reach_only=False,
        reach_threshold_xy_m=0.05,
        reach_threshold_z_m=0.08,
        reach_stop_on_success=False,
    )

    assert calls
    assert "--disable-gripper-gates" in calls[0]
