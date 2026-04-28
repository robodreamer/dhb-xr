#!/usr/bin/env python3
"""Replay scripted post-contact corrections from captured LIBERO sim states.

The input is produced by ``libero_postcontact_correction_plan.py``. Each row
contains a flattened simulator state captured from a closed-loop rollout and a
short closed-gripper delta-action correction plan. This script resets LIBERO to
that exact state and executes the actions with the same embodik IK controller
used by live rollouts.

This remains training/data-generation infrastructure. It is not part of policy
deployment and does not add object-distance gates to the model.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

try:
    from robot_descriptions import panda_description
except ImportError:
    import sys

    _fallback_site = (
        Path("/home/andypark/Projects/repos/embodik_test/ik_venv/lib/python3.10/site-packages")
    )
    if _fallback_site.exists() and str(_fallback_site) not in sys.path:
        sys.path.insert(0, str(_fallback_site))
    from robot_descriptions import panda_description

try:
    import embodik
except ImportError:
    import sys

    _embodik_python = Path("/home/andypark/Projects/repos/embodik/python")
    if _embodik_python.exists() and str(_embodik_python) not in sys.path:
        sys.path.insert(0, str(_embodik_python))
    import embodik

from scripts.starvla_dhb_embodik_live_rollout import (
    LIBERO_ENV_RESOLUTION,
    _capture_scene_refs,
    _libero_torch_load_compat,
    _relative_snapshot,
)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _setup_controller(robot, *, output_max: float, kp: float) -> None:
    ctrl = robot.controller
    ctrl.output_max = ctrl.nums2array(output_max, len(ctrl.qpos_index))
    ctrl.output_min = ctrl.nums2array(-output_max, len(ctrl.qpos_index))
    ctrl.kp = ctrl.nums2array(kp, len(ctrl.qpos_index))
    ctrl.kd = 2 * np.sqrt(ctrl.kp) * 1.0
    ctrl.action_scale = None


def _target_xy_mm(snapshot: dict[str, Any], field: str) -> float | None:
    value = snapshot.get(field)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape[0] < 2:
        return None
    return float(np.linalg.norm(arr[:2]) * 1000.0)


def execute_plan_row(
    *,
    env,
    row: dict[str, Any],
    object_target_field: str,
    target_xy_tolerance_mm: float,
    target_z_min_mm: float,
    target_z_max_mm: float,
    output_max: float,
    kp: float,
    substeps: int,
) -> dict[str, Any]:
    sim_state = row.get("sim_state")
    if sim_state is None:
        raise ValueError(f"Plan row step {row.get('step')} has no sim_state")
    raw_actions = np.asarray(row.get("raw_correction_actions"), dtype=np.float32)
    if raw_actions.ndim != 2 or raw_actions.shape[1] < 7:
        raise ValueError(
            f"Expected raw_correction_actions shape (T, 7), got {tuple(raw_actions.shape)}"
        )

    obs = env.regenerate_obs_from_state(np.asarray(sim_state, dtype=np.float64))
    robot = env.env.robots[0]
    _setup_controller(robot, output_max=output_max, kp=kp)

    robot_model = embodik.RobotModel(urdf_path=panda_description.URDF_PATH)
    solver = embodik.KinematicsSolver(robot_model)
    ik_opts = embodik.PositionIKOptions()
    ik_opts.position_tolerance = 0.0005
    ik_opts.orientation_tolerance = 100.0
    ik_opts.orientation_gain = 0.1
    ik_opts.max_iterations = 200

    q_current = np.zeros(robot_model.nq, dtype=np.float64)
    q_current[:7] = np.asarray(robot._joint_positions, dtype=np.float64)
    robot_model.update_configuration(q_current)
    base_offset = np.asarray(obs["robot0_eef_pos"], dtype=np.float64) - np.asarray(
        robot_model.get_frame_pose("panda_hand_tcp").translation,
        dtype=np.float64,
    )

    scene_refs = _capture_scene_refs(env)
    start_snapshot = _relative_snapshot(obs["robot0_eef_pos"], scene_refs, env=env)
    trace = []
    ik_failures = 0
    min_target_xy_mm = _target_xy_mm(start_snapshot, object_target_field)
    target_reached_step = None
    reward = 0.0

    q_seed = np.zeros(robot_model.nq, dtype=np.float64)
    q_seed[:7] = np.asarray(robot._joint_positions, dtype=np.float64)
    for step_idx, action in enumerate(raw_actions):
        ee_now = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        robot_model.update_configuration(q_seed)
        current_rot = np.asarray(
            robot_model.get_frame_pose("panda_hand_tcp").rotation,
            dtype=np.float64,
        )
        target_T = np.eye(4, order="F")
        target_T[:3, :3] = current_rot
        target_T[:3, 3] = ee_now + np.asarray(action[:3], dtype=np.float64) - base_offset

        result_ik = solver.solve_position(q_seed, target_T, "panda_hand_tcp", ik_opts)
        if result_ik.status == embodik.SolverStatus.SUCCESS:
            q_seed = np.asarray(result_ik.q_solution, dtype=np.float64)
        else:
            ik_failures += 1

        for _ in range(max(1, int(substeps))):
            q_target = q_seed[:7]
            q_now = np.asarray(robot._joint_positions, dtype=np.float64)
            delta = q_target - q_now
            joint_action = np.clip(delta / float(output_max), -1.0, 1.0)
            full_action = np.concatenate([joint_action, [1.0]])
            obs, r, done, info = env.step(full_action)
            reward = max(reward, float(r))

        snapshot = _relative_snapshot(obs["robot0_eef_pos"], scene_refs, env=env)
        target_xy_mm = _target_xy_mm(snapshot, object_target_field)
        if target_xy_mm is not None:
            min_target_xy_mm = (
                target_xy_mm
                if min_target_xy_mm is None
                else min(float(min_target_xy_mm), target_xy_mm)
            )
        target_vec = snapshot.get(object_target_field)
        target_ready = False
        if target_vec is not None:
            arr = np.asarray(target_vec, dtype=np.float32)
            xy_mm = float(np.linalg.norm(arr[:2]) * 1000.0)
            z_mm = float(arr[2] * 1000.0) if arr.shape[0] >= 3 else 0.0
            target_ready = (
                xy_mm <= float(target_xy_tolerance_mm)
                and float(target_z_min_mm) <= z_mm <= float(target_z_max_mm)
            )
        if target_ready and target_reached_step is None:
            target_reached_step = int(step_idx)
        trace.append(
            {
                "step": int(step_idx),
                "raw_action": action.astype(np.float32).tolist(),
                "target_xy_mm": target_xy_mm,
                "target_ready": bool(target_ready),
                **snapshot,
            }
        )

    final_snapshot = _relative_snapshot(obs["robot0_eef_pos"], scene_refs, env=env)
    final_target_xy_mm = _target_xy_mm(final_snapshot, object_target_field)
    start_target_xy_mm = _target_xy_mm(start_snapshot, object_target_field)
    return {
        "source_rollout_json": row.get("source_rollout_json"),
        "source_step": row.get("step"),
        "task_suite_name": row.get("task_suite_name"),
        "task_id": row.get("task_id"),
        "episode_idx": row.get("episode_idx"),
        "offset_cm": row.get("offset_cm"),
        "object_target_field": object_target_field,
        "num_actions": int(len(raw_actions)),
        "ik_failures": int(ik_failures),
        "reward": float(reward),
        "success": bool(env.check_success()),
        "target_reached_step": target_reached_step,
        "start_target_xy_mm": start_target_xy_mm,
        "min_target_xy_mm": min_target_xy_mm,
        "final_target_xy_mm": final_target_xy_mm,
        "target_progress_mm": None
        if start_target_xy_mm is None or min_target_xy_mm is None
        else float(start_target_xy_mm - min_target_xy_mm),
        "start_snapshot": start_snapshot,
        "final_snapshot": final_snapshot,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-json", required=True)
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--row-index", type=int, action="append", default=None)
    parser.add_argument("--max-rows", type=int, default=2)
    parser.add_argument("--rollout-seed", type=int, default=70)
    parser.add_argument("--object-target-field", default=None)
    parser.add_argument("--target-xy-tolerance-mm", type=float, default=45.0)
    parser.add_argument("--target-z-min-mm", type=float, default=-20.0)
    parser.add_argument("--target-z-max-mm", type=float, default=55.0)
    parser.add_argument("--output-max", type=float, default=0.5)
    parser.add_argument("--kp", type=float, default=150.0)
    parser.add_argument("--substeps", type=int, default=3)
    args = parser.parse_args()

    _seed(int(args.rollout_seed))
    plan = json.loads(Path(args.plan_json).read_text(encoding="utf-8"))
    rows = [row for row in plan.get("rows", []) if isinstance(row, dict)]
    if args.row_index:
        wanted = {int(idx) for idx in args.row_index}
        rows = [row for idx, row in enumerate(rows) if idx in wanted]
    rows = rows[: max(0, int(args.max_rows))]
    if not rows:
        raise ValueError(f"No plan rows selected from {args.plan_json}")

    first = rows[0]
    task_suite_name = first.get("task_suite_name") or "libero_spatial"
    task_id = int(first.get("task_id") or 0)
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    task = task_suite.get_task(task_id)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    with _libero_torch_load_compat():
        env = OffScreenRenderEnv(
            bddl_file_name=task_bddl_file,
            controller="JOINT_POSITION",
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(int(args.rollout_seed))
        env.reset()
        results = []
        try:
            for row in rows:
                object_target_field = (
                    args.object_target_field
                    or row.get("object_target_field")
                    or "bowl_minus_plate"
                )
                results.append(
                    execute_plan_row(
                        env=env,
                        row=row,
                        object_target_field=str(object_target_field),
                        target_xy_tolerance_mm=float(args.target_xy_tolerance_mm),
                        target_z_min_mm=float(args.target_z_min_mm),
                        target_z_max_mm=float(args.target_z_max_mm),
                        output_max=float(args.output_max),
                        kp=float(args.kp),
                        substeps=max(1, int(args.substeps)),
                    )
                )
        finally:
            env.close()

    payload = {
        "plan_json": str(args.plan_json),
        "rollout_seed": int(args.rollout_seed),
        "num_rows": int(len(results)),
        "num_target_reached": int(sum(r["target_reached_step"] is not None for r in results)),
        "avg_target_progress_mm": None
        if not results
        else float(
            np.mean(
                [
                    r["target_progress_mm"] or 0.0
                    for r in results
                ]
            )
        ),
        "results": results,
    }
    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
