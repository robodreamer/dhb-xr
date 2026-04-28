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
from typing import Any, Sequence

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from dhb_xr.integration.vla.libero import (
    build_libero_state_vector_from_obs,
    build_state_history_window,
)
from scripts.libero_rollout_transport_miner import _extract_action_stats, _normalize_raw_action

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
    LIBERO_COMMAND_CLOSED_QPOS,
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


def _object_token_from_target_field(field: str) -> str:
    return field.split("_minus_", 1)[0] if "_minus_" in field else field


def _object_pair_to_eef_targets(
    snapshot: dict[str, Any],
    object_target_field: str,
) -> tuple[list[float], list[float]]:
    """Return generic object->EEF and target->EEF vectors for object-minus-target fields."""

    zero = [0.0, 0.0, 0.0]
    if "_minus_" not in object_target_field:
        return zero, zero
    object_token, target_token = object_target_field.split("_minus_", 1)

    def _neg_ee_minus(token: str) -> list[float]:
        value = snapshot.get(f"ee_minus_{token}")
        if value is None:
            return zero
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape[0] < 3:
            return zero
        return (-arr[:3]).astype(np.float32).tolist()

    return _neg_ee_minus(object_token), _neg_ee_minus(target_token)


def _body_name_from_ref(ref: str | None) -> str | None:
    if not ref or ":" not in ref:
        return None
    kind, name = ref.split(":", 1)
    return name if kind == "body" else None


def _freejoint_name_for_body(model, body_name: str) -> str | None:
    prefix = body_name.removesuffix("_main")
    for joint_id in range(model.njnt):
        joint_name = model.joint_id2name(joint_id)
        if not joint_name:
            continue
        if prefix in joint_name and int(model.jnt_type[joint_id]) == 0:
            return joint_name
    return None


def _attachment_joint_name(env, scene_refs: dict[str, Any], object_target_field: str) -> str | None:
    token = _object_token_from_target_field(object_target_field)
    body_name = _body_name_from_ref(scene_refs.get(f"{token}_ref"))
    if body_name is None:
        return None
    return _freejoint_name_for_body(env.env.sim.model, body_name)


def _set_freejoint_pos(env, joint_name: str, pos: np.ndarray) -> None:
    sim = env.env.sim
    model = sim.model
    joint_id = model.joint_name2id(joint_name)
    qpos_addr = int(model.jnt_qposadr[joint_id])
    dof_addr = int(model.jnt_dofadr[joint_id])
    sim.data.qpos[qpos_addr : qpos_addr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    sim.data.qvel[dof_addr : dof_addr + 6] = 0.0
    sim.forward()


def _refresh_obs(env) -> dict[str, Any]:
    return env.regenerate_obs_from_state(env.get_sim_state())


def _apply_attached_object(
    env,
    *,
    joint_name: str,
    ee_pos: np.ndarray,
    object_minus_ee: np.ndarray,
) -> dict[str, Any]:
    _set_freejoint_pos(
        env,
        joint_name,
        np.asarray(ee_pos, dtype=np.float64).reshape(3)
        + np.asarray(object_minus_ee, dtype=np.float64).reshape(3),
    )
    return _refresh_obs(env)


def _closed_policy_state_window(
    history: list[np.ndarray],
    obs: dict[str, Any],
    *,
    history_window: int,
) -> np.ndarray:
    state = build_libero_state_vector_from_obs(obs).astype(np.float32)
    state[-1] = LIBERO_COMMAND_CLOSED_QPOS
    history.append(state)
    return build_state_history_window(history, history_window=history_window)


def _export_policy_trace_chunks(
    *,
    row: dict[str, Any],
    result: dict[str, Any],
    records: list[dict[str, Any]],
    policy_trace_dir: Path,
    manifest_rows: list[dict[str, Any]],
    action_stats: dict[str, Any],
    action_horizon: int,
    lift_action_loss_weight: float = 1.0,
    carry_action_loss_weight: float = 1.0,
    lower_action_loss_weight: float = 1.0,
    phase_z_threshold_mm: float = 1.0,
) -> None:
    action_horizon = max(1, int(action_horizon))
    policy_trace_dir.mkdir(parents=True, exist_ok=True)
    row_tag = f"row{len(manifest_rows):04d}_step{row.get('step')}"
    for chunk_start in range(0, len(records), action_horizon):
        chunk = records[chunk_start : chunk_start + action_horizon]
        if not chunk:
            continue
        npz_path = policy_trace_dir / f"{row_tag}_chunk{chunk_start // action_horizon:03d}.npz"
        first = chunk[0]
        np.savez_compressed(
            npz_path,
            agentview_image=np.asarray(first["agentview_image"], dtype=np.uint8),
            wrist_image=np.asarray(first["wrist_image"], dtype=np.uint8),
            state_window=np.asarray(first["state_window"], dtype=np.float32),
            ee_pos_live=np.asarray(first["ee_pos_live"], dtype=np.float32),
            chunk_positions=np.asarray(
                [record["ee_pos_live"] for record in chunk],
                dtype=np.float32,
            ),
        )
        target_action = [
            _normalize_raw_action(record["raw_action"], action_stats)
            for record in chunk
        ]
        target_object = [record["object_target"] for record in chunk]
        target_object_to_eef = [
            record.get("object_to_eef", [0.0, 0.0, 0.0])
            for record in chunk
        ]
        target_target_to_eef = [
            record.get("target_to_eef", [0.0, 0.0, 0.0])
            for record in chunk
        ]
        target_object_pair_to_eef = [
            list(object_to_eef) + list(target_to_eef)
            for object_to_eef, target_to_eef in zip(target_object_to_eef, target_target_to_eef)
        ]
        target_transport_phase = [
            _transport_phase_from_action(
                record["raw_action"],
                z_threshold_mm=phase_z_threshold_mm,
            )
            for record in chunk
        ]
        action_loss_weight = [
            _transport_phase_loss_weight(
                phase,
                lift_weight=lift_action_loss_weight,
                carry_weight=carry_action_loss_weight,
                lower_weight=lower_action_loss_weight,
            )
            for phase in target_transport_phase
        ]
        n_use = len(chunk)
        manifest_rows.append(
            {
                "rollout_json": row.get("source_rollout_json"),
                "source_plan_step": row.get("step"),
                "source_replay_result_step": result.get("target_reached_step"),
                "checkpoint": row.get("checkpoint"),
                "task_id": row.get("task_id"),
                "episode_idx": row.get("episode_idx"),
                "offset_cm": row.get("offset_cm"),
                "chunk_index": chunk_start // action_horizon,
                "t_start": chunk_start,
                "n_use": n_use,
                "npz_path": str(npz_path),
                "target_action": target_action,
                "target_action_normalized": True,
                "target_object_target": target_object,
                "target_object_to_eef": target_object_to_eef,
                "target_target_to_eef": target_target_to_eef,
                "target_object_pair_to_eef": target_object_pair_to_eef,
                "target_gripper_state": [[1.0] for _ in chunk],
                "target_close": [1.0 for _ in chunk],
                "train_mask": [1.0 for _ in chunk],
                "target_transport_phase": target_transport_phase,
                "action_loss_weight": action_loss_weight,
                "labels": ["attached_transport_positive" for _ in chunk],
                "object_target_xy_mm": [
                    record.get("target_xy_mm")
                    for record in chunk
                ],
                "attached_transport_positive": True,
                "target_reached": result.get("target_reached_step") is not None,
                "attach_object_to_ee": result.get("attach_object_to_ee"),
                "attachment_joint_name": result.get("attachment_joint_name"),
            }
        )


def _transport_phase_from_action(
    raw_action: Sequence[float],
    *,
    z_threshold_mm: float,
) -> str:
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    z_mm = float(action[2] * 1000.0) if action.shape[0] >= 3 else 0.0
    threshold = max(0.0, float(z_threshold_mm))
    if z_mm > threshold:
        return "lift"
    if z_mm < -threshold:
        return "lower"
    return "carry"


def _transport_phase_loss_weight(
    phase: str,
    *,
    lift_weight: float,
    carry_weight: float,
    lower_weight: float,
) -> float:
    weights = {
        "lift": float(lift_weight),
        "carry": float(carry_weight),
        "lower": float(lower_weight),
    }
    return max(0.0, weights.get(str(phase), 1.0))


def _target_ready(
    snapshot: dict[str, Any],
    field: str,
    *,
    target_xy_tolerance_mm: float,
    target_z_min_mm: float,
    target_z_max_mm: float,
) -> bool:
    target_vec = snapshot.get(field)
    if target_vec is None:
        return False
    arr = np.asarray(target_vec, dtype=np.float32)
    if arr.shape[0] < 3:
        return False
    xy_mm = float(np.linalg.norm(arr[:2]) * 1000.0)
    z_mm = float(arr[2] * 1000.0)
    return (
        xy_mm <= float(target_xy_tolerance_mm)
        and float(target_z_min_mm) <= z_mm <= float(target_z_max_mm)
    )


def _closed_loop_action(
    snapshot: dict[str, Any],
    field: str,
    *,
    remaining_steps: int,
    max_action_step_mm: float,
    correction_gain: float,
    include_z: bool,
    transport_profile: str = "target_vector",
    lift_height_mm: float = 45.0,
    place_z_mm: float = 5.0,
    lower_xy_mm: float = 45.0,
    lift_steps: int = 8,
) -> np.ndarray:
    target_vec = snapshot.get(field)
    if target_vec is None:
        raise KeyError(f"Snapshot does not contain object target field '{field}'")

    target = np.asarray(target_vec, dtype=np.float32).reshape(-1)[:3]
    remaining = max(1, int(remaining_steps))
    correction_gain = max(0.0, float(correction_gain))
    transport_profile = str(transport_profile or "target_vector")
    if transport_profile == "target_vector":
        desired = -target
        if not include_z:
            desired[2] = 0.0
        per_step = desired * correction_gain / float(remaining)
    elif transport_profile == "lift_transport_place":
        xy_mm = float(np.linalg.norm(target[:2]) * 1000.0)
        z_target = (
            float(place_z_mm) / 1000.0
            if xy_mm <= float(lower_xy_mm)
            else float(lift_height_mm) / 1000.0
        )
        per_step = np.zeros((3,), dtype=np.float32)
        per_step[:2] = -target[:2] * correction_gain / float(remaining)
        z_remaining = remaining
        if target[2] < z_target:
            z_remaining = max(1, min(remaining, int(lift_steps)))
        per_step[2] = (z_target - float(target[2])) * correction_gain / float(z_remaining)
    else:
        raise ValueError(
            "transport_profile must be 'target_vector' or 'lift_transport_place', "
            f"got {transport_profile!r}"
        )

    max_step_m = max(0.0, float(max_action_step_mm)) / 1000.0
    if max_step_m > 0.0:
        per_step = np.clip(per_step, -max_step_m, max_step_m)
    action = np.zeros((7,), dtype=np.float32)
    action[:3] = per_step.astype(np.float32)
    action[6] = 1.0
    return action


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
    closed_loop_recompute: bool,
    closed_loop_steps: int,
    max_action_step_mm: float,
    correction_gain: float,
    include_z: bool,
    transport_profile: str,
    lift_height_mm: float,
    place_z_mm: float,
    lower_xy_mm: float,
    lift_steps: int,
    attach_object_to_ee: bool,
    policy_trace_dir: Path | None = None,
    manifest_rows: list[dict[str, Any]] | None = None,
    action_stats: dict[str, Any] | None = None,
    export_only_target_reached: bool = False,
    action_horizon: int = 8,
    state_history_window: int = 4,
    lift_action_loss_weight: float = 1.0,
    carry_action_loss_weight: float = 1.0,
    lower_action_loss_weight: float = 1.0,
    phase_z_threshold_mm: float = 1.0,
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
    attach_joint_name = None
    attach_offset = None
    if attach_object_to_ee:
        attach_joint_name = _attachment_joint_name(env, scene_refs, object_target_field)
        if attach_joint_name is None:
            raise RuntimeError(
                f"Could not resolve free joint for object target field '{object_target_field}'"
            )
        token = _object_token_from_target_field(object_target_field)
        object_pos = scene_refs.get(f"{token}_pos")
        if object_pos is None:
            raise RuntimeError(f"Could not resolve object position for token '{token}'")
        attach_offset = np.asarray(object_pos, dtype=np.float64) - np.asarray(
            obs["robot0_eef_pos"],
            dtype=np.float64,
        )
    start_snapshot = _relative_snapshot(obs["robot0_eef_pos"], scene_refs, env=env)
    trace = []
    policy_records: list[dict[str, Any]] = []
    state_history: list[np.ndarray] = []
    ik_failures = 0
    min_target_xy_mm = _target_xy_mm(start_snapshot, object_target_field)
    target_reached_step = None
    reward = 0.0

    q_seed = np.zeros(robot_model.nq, dtype=np.float64)
    q_seed[:7] = np.asarray(robot._joint_positions, dtype=np.float64)
    action_count = max(1, int(closed_loop_steps)) if closed_loop_recompute else len(raw_actions)
    executed_actions: list[list[float]] = []
    for step_idx in range(action_count):
        state_window = _closed_policy_state_window(
            state_history,
            obs,
            history_window=state_history_window,
        )
        if closed_loop_recompute:
            pre_snapshot = _relative_snapshot(obs["robot0_eef_pos"], scene_refs, env=env)
            if _target_ready(
                pre_snapshot,
                object_target_field,
                target_xy_tolerance_mm=target_xy_tolerance_mm,
                target_z_min_mm=target_z_min_mm,
                target_z_max_mm=target_z_max_mm,
            ):
                target_reached_step = int(step_idx)
                break
            action = _closed_loop_action(
                pre_snapshot,
                object_target_field,
                remaining_steps=max(1, action_count - step_idx),
                max_action_step_mm=max_action_step_mm,
                correction_gain=correction_gain,
                include_z=include_z,
                transport_profile=transport_profile,
                lift_height_mm=lift_height_mm,
                place_z_mm=place_z_mm,
                lower_xy_mm=lower_xy_mm,
                lift_steps=lift_steps,
            )
        else:
            pre_snapshot = _relative_snapshot(obs["robot0_eef_pos"], scene_refs, env=env)
            action = raw_actions[step_idx]
        object_target = pre_snapshot.get(object_target_field)
        if object_target is not None:
            object_target = np.asarray(object_target, dtype=np.float32).reshape(-1)[:3].tolist()
        object_to_eef, target_to_eef = _object_pair_to_eef_targets(
            pre_snapshot,
            object_target_field,
        )
        policy_records.append(
            {
                "agentview_image": np.asarray(obs["agentview_image"], dtype=np.uint8),
                "wrist_image": np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8),
                "state_window": state_window.astype(np.float32),
                "ee_pos_live": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                "raw_action": action.astype(np.float32).tolist(),
                "object_target": object_target or [0.0, 0.0, 0.0],
                "object_to_eef": object_to_eef,
                "target_to_eef": target_to_eef,
                "target_xy_mm": _target_xy_mm(pre_snapshot, object_target_field),
            }
        )
        executed_actions.append(action.astype(np.float32).tolist())
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
            if attach_object_to_ee and attach_joint_name is not None and attach_offset is not None:
                obs = _apply_attached_object(
                    env,
                    joint_name=attach_joint_name,
                    ee_pos=np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                    object_minus_ee=attach_offset,
                )

        snapshot = _relative_snapshot(obs["robot0_eef_pos"], scene_refs, env=env)
        target_xy_mm = _target_xy_mm(snapshot, object_target_field)
        if target_xy_mm is not None:
            min_target_xy_mm = (
                target_xy_mm
                if min_target_xy_mm is None
                else min(float(min_target_xy_mm), target_xy_mm)
            )
        target_ready = _target_ready(
            snapshot,
            object_target_field,
            target_xy_tolerance_mm=target_xy_tolerance_mm,
            target_z_min_mm=target_z_min_mm,
            target_z_max_mm=target_z_max_mm,
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
    result = {
        "source_rollout_json": row.get("source_rollout_json"),
        "source_step": row.get("step"),
        "task_suite_name": row.get("task_suite_name"),
        "task_id": row.get("task_id"),
        "episode_idx": row.get("episode_idx"),
        "offset_cm": row.get("offset_cm"),
        "object_target_field": object_target_field,
        "num_actions": int(len(raw_actions)),
        "num_executed_actions": int(len(executed_actions)),
        "closed_loop_recompute": bool(closed_loop_recompute),
        "transport_profile": str(transport_profile),
        "lift_height_mm": float(lift_height_mm),
        "place_z_mm": float(place_z_mm),
        "lower_xy_mm": float(lower_xy_mm),
        "lift_steps": int(lift_steps),
        "attach_object_to_ee": bool(attach_object_to_ee),
        "attachment_joint_name": attach_joint_name,
        "attachment_object_minus_ee": None
        if attach_offset is None
        else np.asarray(attach_offset, dtype=np.float32).tolist(),
        "executed_actions": executed_actions,
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
    if (
        policy_trace_dir is not None
        and manifest_rows is not None
        and action_stats is not None
        and policy_records
        and (
            not bool(export_only_target_reached)
            or result.get("target_reached_step") is not None
        )
    ):
        _export_policy_trace_chunks(
            row=row,
            result=result,
            records=policy_records,
            policy_trace_dir=policy_trace_dir,
            manifest_rows=manifest_rows,
            action_stats=action_stats,
            action_horizon=action_horizon,
            lift_action_loss_weight=lift_action_loss_weight,
            carry_action_loss_weight=carry_action_loss_weight,
            lower_action_loss_weight=lower_action_loss_weight,
            phase_z_threshold_mm=phase_z_threshold_mm,
        )
    return result


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
    parser.add_argument("--closed-loop-recompute", action="store_true")
    parser.add_argument("--closed-loop-steps", type=int, default=48)
    parser.add_argument("--max-action-step-mm", type=float, default=10.0)
    parser.add_argument("--correction-gain", type=float, default=1.0)
    parser.add_argument("--include-z", action="store_true")
    parser.add_argument(
        "--transport-profile",
        choices=("target_vector", "lift_transport_place"),
        default="target_vector",
        help=(
            "Closed-loop correction profile. 'target_vector' preserves the "
            "legacy object-target vector servo. 'lift_transport_place' first "
            "raises the attached object to a generic clearance height, carries "
            "laterally, then lowers near the target."
        ),
    )
    parser.add_argument("--lift-height-mm", type=float, default=45.0)
    parser.add_argument("--place-z-mm", type=float, default=5.0)
    parser.add_argument("--lower-xy-mm", type=float, default=45.0)
    parser.add_argument("--lift-steps", type=int, default=8)
    parser.add_argument(
        "--attach-object-to-ee",
        action="store_true",
        help=(
            "Offline augmentation mode: after each simulator step, keep the "
            "object named by --object-target-field rigidly attached to the EE "
            "at its reset-time offset. This is not deployment logic."
        ),
    )
    parser.add_argument("--save-policy-trace-dir")
    parser.add_argument("--save-policy-manifest-jsonl")
    parser.add_argument("--action-stats-json")
    parser.add_argument("--policy-trace-lift-weight", type=float, default=1.0)
    parser.add_argument("--policy-trace-carry-weight", type=float, default=1.0)
    parser.add_argument("--policy-trace-lower-weight", type=float, default=1.0)
    parser.add_argument("--policy-trace-phase-z-threshold-mm", type=float, default=1.0)
    parser.add_argument(
        "--export-only-target-reached",
        action="store_true",
        help=(
            "When exporting policy-trace positives, skip replay rows that did "
            "not reach the target gate. Use for DAgger-style correction data "
            "so failed replays do not become positive transport supervision."
        ),
    )
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--state-history-window", type=int, default=4)
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
    action_stats = None
    if args.save_policy_manifest_jsonl or args.save_policy_trace_dir:
        if not args.save_policy_manifest_jsonl or not args.save_policy_trace_dir:
            raise ValueError(
                "--save-policy-manifest-jsonl and --save-policy-trace-dir must be used together"
            )
        if not args.action_stats_json:
            raise ValueError("Policy-trace export requires --action-stats-json")
        action_stats_payload = json.loads(Path(args.action_stats_json).read_text(encoding="utf-8"))
        action_stats = _extract_action_stats(action_stats_payload)
    policy_trace_dir = None if args.save_policy_trace_dir is None else Path(args.save_policy_trace_dir)
    manifest_rows: list[dict[str, Any]] = []

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
                        closed_loop_recompute=bool(args.closed_loop_recompute),
                        closed_loop_steps=max(1, int(args.closed_loop_steps)),
                        max_action_step_mm=float(args.max_action_step_mm),
                        correction_gain=max(0.0, float(args.correction_gain)),
                        include_z=bool(args.include_z),
                        transport_profile=str(args.transport_profile),
                        lift_height_mm=float(args.lift_height_mm),
                        place_z_mm=float(args.place_z_mm),
                        lower_xy_mm=float(args.lower_xy_mm),
                        lift_steps=max(1, int(args.lift_steps)),
                        attach_object_to_ee=bool(args.attach_object_to_ee),
                        policy_trace_dir=policy_trace_dir,
                        manifest_rows=manifest_rows,
                        action_stats=action_stats,
                        export_only_target_reached=bool(args.export_only_target_reached),
                        action_horizon=max(1, int(args.action_horizon)),
                        state_history_window=max(1, int(args.state_history_window)),
                        lift_action_loss_weight=float(args.policy_trace_lift_weight),
                        carry_action_loss_weight=float(args.policy_trace_carry_weight),
                        lower_action_loss_weight=float(args.policy_trace_lower_weight),
                        phase_z_threshold_mm=float(args.policy_trace_phase_z_threshold_mm),
                    )
                )
        finally:
            env.close()
    if args.save_policy_manifest_jsonl:
        manifest_path = Path(args.save_policy_manifest_jsonl)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + "\n")

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
        "policy_manifest_jsonl": args.save_policy_manifest_jsonl,
        "policy_manifest_rows": int(len(manifest_rows)),
        "export_only_target_reached": bool(args.export_only_target_reached),
        "results": results,
    }
    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
