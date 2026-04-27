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
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--object-target-field", default="bowl_minus_plate")
    parser.add_argument("--target-xy-tolerance-mm", type=float, default=45.0)
    parser.add_argument("--target-z-min-mm", type=float, default=-25.0)
    parser.add_argument("--target-z-max-mm", type=float, default=55.0)
    parser.add_argument("--gripper-qpos-threshold", type=float, default=0.015)
    parser.add_argument("--correction-top-k", type=int, default=3)
    args = parser.parse_args()

    results = []
    correction_requests: list[dict[str, Any]] = []
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

    summary = {
        "object_target_field": args.object_target_field,
        "target_xy_tolerance_mm": float(args.target_xy_tolerance_mm),
        "target_z_min_mm": float(args.target_z_min_mm),
        "target_z_max_mm": float(args.target_z_max_mm),
        "gripper_qpos_threshold": float(args.gripper_qpos_threshold),
        "correction_top_k": max(0, int(args.correction_top_k)),
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

    if args.print_markdown:
        print("| rollout | physical close step | reopen step | target XY at close mm | best target XY mm | progress mm | target reached steps | correction requests |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in results:
            print(_markdown_row(result))


if __name__ == "__main__":
    main()
