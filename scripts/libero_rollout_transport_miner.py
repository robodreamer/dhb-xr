#!/usr/bin/env python3
"""Mine closed-loop LIBERO traces for post-close transport correction targets.

This is training/data-generation infrastructure. It does not add object-distance
logic to deployment. It turns recorded rollout geometry into a compact report:

- when physical close happens;
- whether the held object ever enters the target affordance window;
- which closed states should become simulator/world-model correction starts.

The correction request is intentionally generic: an object-minus-target vector
and the corresponding object-space delta needed to reach the target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _vector3(value: Any) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def _xy_norm(vec: list[float]) -> float:
    return math.hypot(vec[0], vec[1])


def _physical_close_step(
    trace: list[dict[str, Any]],
    *,
    command_close_step: int | None,
    gripper_qpos_threshold: float,
) -> int | None:
    if not isinstance(command_close_step, int):
        return None
    for row in sorted(trace, key=lambda item: int(item.get("step", 10**9))):
        step = row.get("step")
        if not isinstance(step, int) or step < command_close_step:
            continue
        value = row.get("observed_state_gripper")
        if value is None:
            continue
        try:
            if float(value) <= gripper_qpos_threshold:
                return step
        except (TypeError, ValueError):
            continue
    return command_close_step


def _observed_closed(row: dict[str, Any], *, gripper_qpos_threshold: float) -> bool:
    value = row.get("observed_state_gripper", row.get("state_gripper"))
    try:
        return value is not None and float(value) <= gripper_qpos_threshold
    except (TypeError, ValueError):
        return False


def _target_ready(
    vector: list[float],
    *,
    target_xy_tolerance_mm: float,
    target_z_min_mm: float,
    target_z_max_mm: float,
) -> bool:
    xy_mm = _xy_norm(vector) * 1000.0
    z_mm = vector[2] * 1000.0
    return xy_mm <= target_xy_tolerance_mm and target_z_min_mm <= z_mm <= target_z_max_mm


def _extract_action_stats(payload: dict[str, Any]) -> dict[str, Any]:
    if "action" in payload:
        return payload["action"]
    if "franka" in payload and isinstance(payload["franka"], dict):
        return payload["franka"]["action"]
    first_key = next(iter(payload.keys()))
    return payload[first_key]["action"]


def _normalize_raw_action(raw_action: list[float], action_stats: dict[str, Any]) -> list[float]:
    raw = np.asarray(raw_action, dtype=np.float32)
    low = np.asarray(action_stats["min"], dtype=np.float32)
    high = np.asarray(action_stats["max"], dtype=np.float32)
    mask = np.asarray(
        action_stats.get("mask", np.ones_like(low, dtype=bool)),
        dtype=bool,
    )
    denom = np.where(np.abs(high - low) < 1e-8, 1.0, high - low)
    normalized = 2.0 * (raw - low) / denom - 1.0
    normalized = np.where(mask, normalized, raw)
    return np.clip(normalized, -1.0, 1.0).astype(np.float32).tolist()


def mine_transport_samples(
    rollout: dict[str, Any],
    *,
    rollout_json: str,
    object_target_field: str,
    target_xy_tolerance_mm: float,
    target_z_min_mm: float,
    target_z_max_mm: float,
    gripper_qpos_threshold: float,
    correction_top_k: int,
) -> dict[str, Any]:
    trace = [row for row in rollout.get("step_trace", []) if isinstance(row, dict)]
    close_step = rollout.get("close_step")
    if not isinstance(close_step, int):
        close_step = None
    physical_close = _physical_close_step(
        trace,
        command_close_step=close_step,
        gripper_qpos_threshold=gripper_qpos_threshold,
    )
    reopen_step = rollout.get("reopen_step")
    if not isinstance(reopen_step, int):
        reopen_step = None

    samples: list[dict[str, Any]] = []
    for row in trace:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        vector = _vector3(row.get(object_target_field))
        if vector is None:
            continue
        after_physical_close = physical_close is not None and step >= physical_close
        closed = _observed_closed(row, gripper_qpos_threshold=gripper_qpos_threshold)
        ready = after_physical_close and _target_ready(
            vector,
            target_xy_tolerance_mm=target_xy_tolerance_mm,
            target_z_min_mm=target_z_min_mm,
            target_z_max_mm=target_z_max_mm,
        )
        if ready:
            label = "target_reached"
        elif after_physical_close and closed:
            label = "closed_transport_candidate"
        elif after_physical_close:
            label = "post_close_background"
        else:
            label = "background"
        samples.append(
            {
                "rollout_json": rollout_json,
                "checkpoint": rollout.get("checkpoint"),
                "task_id": rollout.get("task_id"),
                "episode_idx": rollout.get("episode_idx"),
                "offset_cm": rollout.get("offset_cm"),
                "step": step,
                "label": label,
                "object_target_field": object_target_field,
                "object_minus_target_m": vector,
                "object_target_xy_mm": _xy_norm(vector) * 1000.0,
                "object_target_z_mm": vector[2] * 1000.0,
                "observed_closed": closed,
                "phase": row.get("phase"),
            }
        )

    post_close = [
        sample
        for sample in samples
        if physical_close is not None and int(sample["step"]) >= physical_close
    ]
    close_sample = next(
        (sample for sample in samples if physical_close is not None and sample["step"] == physical_close),
        None,
    )
    best = min(post_close, key=lambda item: item["object_target_xy_mm"], default=None)
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample["label"]] = counts.get(sample["label"], 0) + 1

    correction_candidates = [
        sample
        for sample in post_close
        if sample["label"] == "closed_transport_candidate"
    ]
    correction_candidates = sorted(
        correction_candidates,
        key=lambda item: item["object_target_xy_mm"],
    )[: max(0, int(correction_top_k))]
    correction_requests = []
    for sample in correction_candidates:
        vector = sample["object_minus_target_m"]
        correction_requests.append(
            {
                "rollout_json": rollout_json,
                "checkpoint": rollout.get("checkpoint"),
                "task_id": rollout.get("task_id"),
                "episode_idx": rollout.get("episode_idx"),
                "offset_cm": rollout.get("offset_cm"),
                "step": sample["step"],
                "object_target_field": object_target_field,
                "object_minus_target_m": vector,
                "desired_object_delta_to_target_m": [-vector[0], -vector[1], -vector[2]],
                "object_target_xy_mm": sample["object_target_xy_mm"],
                "reason": "closed_post_contact_state_needs_target_transport",
            }
        )

    xy_at_close = None if close_sample is None else close_sample["object_target_xy_mm"]
    best_xy = None if best is None else best["object_target_xy_mm"]
    progress_mm = None
    progress_fraction = None
    if xy_at_close is not None and best_xy is not None:
        progress_mm = float(xy_at_close) - float(best_xy)
        progress_fraction = progress_mm / max(float(xy_at_close), 1e-6)

    return {
        "rollout_json": rollout_json,
        "checkpoint": rollout.get("checkpoint"),
        "task_id": rollout.get("task_id"),
        "episode_idx": rollout.get("episode_idx"),
        "offset_cm": rollout.get("offset_cm"),
        "close_step": close_step,
        "physical_close_step": physical_close,
        "reopen_step": reopen_step,
        "success": bool(rollout.get("success")),
        "target_xy_tolerance_mm": target_xy_tolerance_mm,
        "target_z_min_mm": target_z_min_mm,
        "target_z_max_mm": target_z_max_mm,
        "counts": counts,
        "xy_at_physical_close_mm": xy_at_close,
        "best_target_xy_mm": best_xy,
        "best_target_step": None if best is None else best["step"],
        "target_progress_mm": progress_mm,
        "target_progress_fraction": progress_fraction,
        "target_reached": counts.get("target_reached", 0) > 0,
        "needs_transport_correction": counts.get("target_reached", 0) == 0
        and bool(correction_requests),
        "best_target_candidate": best,
        "correction_requests": correction_requests,
        "samples": samples,
    }


def policy_manifest_rows(
    rollout: dict[str, Any],
    mined_result: dict[str, Any],
    *,
    action_stats: dict[str, Any],
    correction_steps: int,
    max_action_step_mm: float,
    include_z: bool,
) -> list[dict[str, Any]]:
    """Build policy-trace rows with normalized corrective action targets."""
    policy_trace = rollout.get("policy_trace")
    if not isinstance(policy_trace, list):
        return []

    sample_by_step = {
        sample["step"]: sample
        for sample in mined_result.get("samples", [])
        if isinstance(sample, dict) and isinstance(sample.get("step"), int)
    }
    max_step_m = float(max_action_step_mm) / 1000.0
    horizon = max(1, int(correction_steps))
    rows: list[dict[str, Any]] = []
    for chunk in policy_trace:
        if not isinstance(chunk, dict):
            continue
        t_start = chunk.get("t_start")
        n_use = chunk.get("n_use")
        npz_path = chunk.get("npz_path")
        if not isinstance(t_start, int) or not isinstance(n_use, int) or not npz_path:
            continue

        target_action: list[list[float]] = []
        target_object: list[list[float]] = []
        target_gripper_state: list[list[float]] = []
        target_close: list[float] = []
        train_mask: list[float] = []
        labels: list[str | None] = []
        object_target_xy_mm: list[float | None] = []
        for step in range(t_start, t_start + n_use):
            sample = sample_by_step.get(step)
            label = sample.get("label") if sample else None
            labels.append(label)
            if sample and label == "closed_transport_candidate":
                vector = np.asarray(sample["object_minus_target_m"], dtype=np.float32)
                desired_delta = -vector
                if not include_z:
                    desired_delta[2] = 0.0
                raw_xyz = np.clip(desired_delta / float(horizon), -max_step_m, max_step_m)
                raw_action = [
                    float(raw_xyz[0]),
                    float(raw_xyz[1]),
                    float(raw_xyz[2]),
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
                target_action.append(_normalize_raw_action(raw_action, action_stats))
                target_object.append(vector.astype(np.float32).tolist())
                target_gripper_state.append([1.0])
                target_close.append(1.0)
                train_mask.append(1.0)
                object_target_xy_mm.append(float(sample["object_target_xy_mm"]))
            else:
                target_action.append([0.0] * 7)
                target_object.append([0.0, 0.0, 0.0])
                target_gripper_state.append([1.0])
                target_close.append(1.0)
                train_mask.append(0.0)
                object_target_xy_mm.append(None)

        if not any(value > 0.5 for value in train_mask):
            continue
        rows.append(
            {
                "rollout_json": mined_result.get("rollout_json"),
                "checkpoint": mined_result.get("checkpoint"),
                "task_id": mined_result.get("task_id"),
                "episode_idx": mined_result.get("episode_idx"),
                "offset_cm": mined_result.get("offset_cm"),
                "chunk_index": chunk.get("chunk_index"),
                "t_start": t_start,
                "n_use": n_use,
                "npz_path": str(npz_path),
                "target_action": target_action,
                "target_action_normalized": True,
                "target_object_target": target_object,
                "target_gripper_state": target_gripper_state,
                "target_close": target_close,
                "train_mask": train_mask,
                "labels": labels,
                "object_target_xy_mm": object_target_xy_mm,
                "correction_steps": int(horizon),
                "max_action_step_mm": float(max_action_step_mm),
                "correction_include_z": bool(include_z),
                "rel_live": chunk.get("rel_live"),
            }
        )
    return rows


def _markdown_row(result: dict[str, Any]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    return (
        f"| {Path(str(result['rollout_json'])).name} | "
        f"{result.get('physical_close_step')} | {result.get('reopen_step')} | "
        f"{fmt(result.get('xy_at_physical_close_mm'))} | "
        f"{fmt(result.get('best_target_xy_mm'))} @ {result.get('best_target_step', '-')} | "
        f"{fmt(result.get('target_progress_mm'))} | "
        f"{result.get('counts', {}).get('target_reached', 0)} | "
        f"{len(result.get('correction_requests', []))} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine LIBERO rollout traces for post-close transport diagnostics."
    )
    parser.add_argument("--rollout-json", nargs="+", required=True)
    parser.add_argument("--save-json")
    parser.add_argument("--save-correction-jsonl")
    parser.add_argument("--save-policy-manifest-jsonl")
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--object-target-field", default="bowl_minus_plate")
    parser.add_argument("--target-xy-tolerance-mm", type=float, default=45.0)
    parser.add_argument("--target-z-min-mm", type=float, default=-25.0)
    parser.add_argument("--target-z-max-mm", type=float, default=55.0)
    parser.add_argument("--gripper-qpos-threshold", type=float, default=0.015)
    parser.add_argument("--correction-top-k", type=int, default=3)
    parser.add_argument("--action-stats-json")
    parser.add_argument("--correction-steps", type=int, default=8)
    parser.add_argument("--max-action-step-mm", type=float, default=18.0)
    parser.add_argument("--transport-correction-include-z", action="store_true")
    args = parser.parse_args()

    action_stats = None
    if args.save_policy_manifest_jsonl:
        if not args.action_stats_json:
            raise ValueError("--save-policy-manifest-jsonl requires --action-stats-json")
        action_stats_payload = json.loads(Path(args.action_stats_json).read_text(encoding="utf-8"))
        action_stats = _extract_action_stats(action_stats_payload)

    results = []
    correction_requests: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    for path_str in args.rollout_json:
        rollout = json.loads(Path(path_str).read_text(encoding="utf-8"))
        result = mine_transport_samples(
            rollout,
            rollout_json=path_str,
            object_target_field=args.object_target_field,
            target_xy_tolerance_mm=float(args.target_xy_tolerance_mm),
            target_z_min_mm=float(args.target_z_min_mm),
            target_z_max_mm=float(args.target_z_max_mm),
            gripper_qpos_threshold=float(args.gripper_qpos_threshold),
            correction_top_k=max(0, int(args.correction_top_k)),
        )
        results.append(result)
        correction_requests.extend(result["correction_requests"])
        if action_stats is not None:
            policy_rows.extend(
                policy_manifest_rows(
                    rollout,
                    result,
                    action_stats=action_stats,
                    correction_steps=max(1, int(args.correction_steps)),
                    max_action_step_mm=float(args.max_action_step_mm),
                    include_z=bool(args.transport_correction_include_z),
                )
            )

    summary = {
        "object_target_field": args.object_target_field,
        "target_xy_tolerance_mm": float(args.target_xy_tolerance_mm),
        "target_z_min_mm": float(args.target_z_min_mm),
        "target_z_max_mm": float(args.target_z_max_mm),
        "gripper_qpos_threshold": float(args.gripper_qpos_threshold),
        "correction_top_k": max(0, int(args.correction_top_k)),
        "policy_manifest_rows": int(len(policy_rows)),
        "policy_correction_steps": max(1, int(args.correction_steps)),
        "policy_max_action_step_mm": float(args.max_action_step_mm),
        "policy_correction_include_z": bool(args.transport_correction_include_z),
        "results": results,
    }
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_correction_jsonl:
        out = Path(args.save_correction_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in correction_requests:
                f.write(json.dumps(row) + "\n")
    if args.save_policy_manifest_jsonl:
        out = Path(args.save_policy_manifest_jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for row in policy_rows:
                f.write(json.dumps(row) + "\n")

    if args.print_markdown:
        print("| rollout | physical close step | reopen step | target XY at close mm | best target XY mm | progress mm | target reached steps | correction requests |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in results:
            print(_markdown_row(result))


if __name__ == "__main__":
    main()
