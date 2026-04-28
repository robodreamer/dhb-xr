#!/usr/bin/env python3
"""Mine closed-loop LIBERO rollout traces for contact-head calibration samples.

The miner is intentionally diagnostic: it does not add object-distance gates to
deployment. It converts recorded rollout geometry into reusable evidence for
what the learned close-readiness head should have done at each closed-loop
state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _vector3(value: Any, *, field: str) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))


def _sub(lhs: list[float], rhs: list[float]) -> list[float]:
    return [lhs[i] - rhs[i] for i in range(3)]


def _abs_mm(vec_m: list[float]) -> list[float]:
    return [abs(value) * 1000.0 for value in vec_m]


def _xy_norm(vec: list[float]) -> float:
    return math.hypot(vec[0], vec[1])


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty vector list")
    return [sum(vec[i] for vec in vectors) / len(vectors) for i in range(3)]


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


def _object_pair_to_eef_targets(
    trace_row: dict[str, Any] | None,
    object_target_field: str | None,
) -> tuple[list[float], list[float]]:
    zero = [0.0, 0.0, 0.0]
    if not trace_row or not object_target_field or "_minus_" not in object_target_field:
        return zero, zero
    object_token, target_token = object_target_field.split("_minus_", 1)

    def _neg_ee_minus(token: str) -> list[float]:
        value = _vector3(trace_row.get(f"ee_minus_{token}"), field=f"ee_minus_{token}")
        if value is None:
            return zero
        return [-float(value[0]), -float(value[1]), -float(value[2])]

    return _neg_ee_minus(object_token), _neg_ee_minus(target_token)


def _object_target_vector(
    trace_row: dict[str, Any] | None,
    object_target_field: str | None,
) -> list[float]:
    if not trace_row or not object_target_field:
        return [0.0, 0.0, 0.0]
    return _vector3(trace_row.get(object_target_field), field=object_target_field) or [
        0.0,
        0.0,
        0.0,
    ]


def _reference_vector(demo_geometry: dict[str, Any], event_name: str, key: str) -> list[float]:
    vectors = [
        vec
        for row in demo_geometry.get("events", [])
        if row.get("event") == event_name
        for vec in [_vector3(row.get(key), field=f"{event_name}.{key}")]
        if vec is not None
    ]
    if not vectors:
        raise ValueError(f"no vectors found for event={event_name} key={key}")
    return _mean(vectors)


def _chunk_prob_map(chunks: list[dict[str, Any]], key: str) -> dict[int, float]:
    by_step: dict[int, float] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        t_start = chunk.get("t_start")
        values = chunk.get(key)
        if not isinstance(t_start, int) or not isinstance(values, list):
            continue
        for i, value in enumerate(values):
            try:
                by_step[int(t_start) + i] = float(value)
            except (TypeError, ValueError):
                continue
    return by_step


def _classify_sample(
    *,
    is_affordance_positive: bool,
    is_near_object_hard_negative: bool,
    close_prob: float | None,
    desired_grip: float,
    effective_grip: float,
    close_step: int | None,
    step: int,
    positive_prob_margin: float,
    hard_negative_prob: float,
) -> str:
    tried_close = desired_grip > 0.0 or effective_grip > 0.0 or step == close_step
    confident_close = close_prob is not None and close_prob >= hard_negative_prob
    under_confident = close_prob is None or close_prob < positive_prob_margin

    if is_affordance_positive and under_confident and effective_grip <= 0.0:
        return "missed_positive"
    if is_affordance_positive:
        return "positive"
    if tried_close or confident_close or is_near_object_hard_negative:
        return "hard_negative"
    return "background"


def mine_contact_samples(
    rollout: dict[str, Any],
    *,
    rollout_json: str,
    close_reference_m: list[float],
    ee_object_field: str,
    positive_tolerance_mm: float,
    positive_component_tolerance_mm: list[float] | None,
    best_positive_top_k_per_rollout: int,
    best_positive_max_error_mm: float,
    best_positive_window: int,
    positive_prob_margin: float,
    hard_negative_prob: float,
    near_object_hard_negative_xy_mm: float | None,
    near_object_hard_negative_z_min_mm: float,
    near_object_hard_negative_z_max_mm: float,
) -> dict[str, Any]:
    trace = [row for row in rollout.get("step_trace", []) if isinstance(row, dict)]
    chunks = [row for row in rollout.get("chunks", []) if isinstance(row, dict)]
    close_probs = _chunk_prob_map(chunks, "close_probs")
    open_probs = _chunk_prob_map(chunks, "open_probs")
    close_step = rollout.get("close_step")
    if not isinstance(close_step, int):
        close_step = None

    candidates: list[dict[str, Any]] = []
    for row in trace:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        vector = _vector3(row.get(ee_object_field), field=ee_object_field)
        if vector is None:
            continue
        component_error_m = _sub(vector, close_reference_m)
        component_error_abs_mm = _abs_mm(component_error_m)
        vector_error_m = _norm(component_error_m)
        candidates.append(
            {
                "row": row,
                "step": step,
                "vector": vector,
                "component_error_abs_mm": component_error_abs_mm,
                "vector_error_m": vector_error_m,
            }
        )

    forced_positive_steps: set[int] = set()
    if best_positive_top_k_per_rollout > 0:
        candidate_steps = {int(candidate["step"]) for candidate in candidates}
        sorted_candidates = sorted(candidates, key=lambda item: float(item["vector_error_m"]))
        selected = [
            candidate
            for candidate in sorted_candidates
            if float(candidate["vector_error_m"]) * 1000.0 <= float(best_positive_max_error_mm)
        ][: int(best_positive_top_k_per_rollout)]
        for candidate in selected:
            step = int(candidate["step"])
            for offset in range(-int(best_positive_window), int(best_positive_window) + 1):
                forced_step = step + offset
                if forced_step in candidate_steps:
                    forced_positive_steps.add(forced_step)

    samples: list[dict[str, Any]] = []
    for candidate in candidates:
        row = candidate["row"]
        step = int(candidate["step"])
        vector = candidate["vector"]
        component_error_abs_mm = candidate["component_error_abs_mm"]
        vector_error_m = float(candidate["vector_error_m"])
        is_affordance_positive = vector_error_m * 1000.0 <= positive_tolerance_mm
        if positive_component_tolerance_mm is not None:
            is_affordance_positive = is_affordance_positive and all(
                component_error_abs_mm[i] <= positive_component_tolerance_mm[i]
                for i in range(3)
            )
        is_forced_best_positive = step in forced_positive_steps
        is_affordance_positive = is_affordance_positive or is_forced_best_positive
        is_near_object_hard_negative = False
        if near_object_hard_negative_xy_mm is not None:
            z_mm = vector[2] * 1000.0
            is_near_object_hard_negative = (
                _xy_norm(vector) * 1000.0 <= near_object_hard_negative_xy_mm
                and near_object_hard_negative_z_min_mm <= z_mm <= near_object_hard_negative_z_max_mm
                and not is_affordance_positive
            )
        close_prob = close_probs.get(step)
        desired_grip = float(row.get("desired_grip", -1.0) or -1.0)
        effective_grip = float(row.get("effective_grip", -1.0) or -1.0)
        label = _classify_sample(
            is_affordance_positive=is_affordance_positive,
            is_near_object_hard_negative=is_near_object_hard_negative,
            close_prob=close_prob,
            desired_grip=desired_grip,
            effective_grip=effective_grip,
            close_step=close_step,
            step=step,
            positive_prob_margin=positive_prob_margin,
            hard_negative_prob=hard_negative_prob,
        )
        samples.append(
            {
                "rollout_json": rollout_json,
                "checkpoint": rollout.get("checkpoint"),
                "task_id": rollout.get("task_id"),
                "episode_idx": rollout.get("episode_idx"),
                "offset_cm": rollout.get("offset_cm"),
                "step": step,
                "phase": row.get("phase"),
                "label": label,
                "ee_object_field": ee_object_field,
                "ee_minus_object_m": vector,
                "ee_object_xy_mm": _xy_norm(vector) * 1000.0,
                "ee_object_z_mm": vector[2] * 1000.0,
                "close_reference_m": close_reference_m,
                "close_vector_error_mm": vector_error_m * 1000.0,
                "close_component_error_abs_mm": component_error_abs_mm,
                "is_forced_best_positive": is_forced_best_positive,
                "is_near_object_hard_negative": is_near_object_hard_negative,
                "close_prob": close_prob,
                "open_prob": open_probs.get(step),
                "desired_grip": desired_grip,
                "effective_grip": effective_grip,
                "state_gripper": row.get("state_gripper"),
                "observed_state_gripper": row.get("observed_state_gripper"),
                "close_ready_gate": row.get("close_ready"),
            }
        )

    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample["label"]] = counts.get(sample["label"], 0) + 1

    best = min(samples, key=lambda row: row["close_vector_error_mm"], default=None)
    worst_false_close = max(
        (row for row in samples if row["label"] == "hard_negative"),
        key=lambda row: row["close_prob"] if row["close_prob"] is not None else -1.0,
        default=None,
    )
    return {
        "rollout_json": rollout_json,
        "checkpoint": rollout.get("checkpoint"),
        "task_id": rollout.get("task_id"),
        "episode_idx": rollout.get("episode_idx"),
        "offset_cm": rollout.get("offset_cm"),
        "close_step": rollout.get("close_step"),
        "reopen_step": rollout.get("reopen_step"),
        "success": bool(rollout.get("success")),
        "num_samples": len(samples),
        "counts": counts,
        "best_close_affordance": best,
        "highest_confidence_hard_negative": worst_false_close,
        "samples": samples,
    }


def policy_manifest_rows(
    rollout: dict[str, Any],
    mined_result: dict[str, Any],
    *,
    action_stats: dict[str, Any] | None = None,
    close_reference_m: list[float] | None = None,
    max_action_step_mm: float = 12.0,
    correction_steps: int = 4,
    correction_gain: float = 1.0,
    object_target_field: str | None = None,
) -> list[dict[str, Any]]:
    """Build chunk-level supervision rows for exported policy-trace NPZ files."""
    policy_trace = rollout.get("policy_trace")
    if not isinstance(policy_trace, list):
        return []

    emit_action_targets = action_stats is not None and close_reference_m is not None
    max_step_m = float(max_action_step_mm) / 1000.0
    horizon = max(1, int(correction_steps))
    close_reference = (
        np.asarray(close_reference_m, dtype=np.float32).reshape(3)
        if close_reference_m is not None
        else None
    )
    sample_by_step = {
        sample["step"]: sample
        for sample in mined_result.get("samples", [])
        if isinstance(sample, dict) and isinstance(sample.get("step"), int)
    }
    trace_by_step = {
        row["step"]: row
        for row in rollout.get("step_trace", [])
        if isinstance(row, dict) and isinstance(row.get("step"), int)
    }
    rows: list[dict[str, Any]] = []
    for chunk in policy_trace:
        if not isinstance(chunk, dict):
            continue
        t_start = chunk.get("t_start")
        n_use = chunk.get("n_use")
        npz_path = chunk.get("npz_path")
        if not isinstance(t_start, int) or not isinstance(n_use, int) or not npz_path:
            continue

        target_close: list[float] = []
        train_mask: list[float] = []
        hard_negative_mask: list[float] = []
        close_vector_error_mm: list[float | None] = []
        labels: list[str | None] = []
        target_action: list[list[float]] = []
        target_gripper_state: list[list[float]] = []
        target_object_target: list[list[float]] = []
        target_object_to_eef: list[list[float]] = []
        target_target_to_eef: list[list[float]] = []
        for step in range(t_start, t_start + n_use):
            sample = sample_by_step.get(step)
            label = sample.get("label") if sample else None
            labels.append(label)
            is_positive = label in {"positive", "missed_positive"}
            is_hard_negative = label == "hard_negative"
            supervised = is_positive or is_hard_negative
            target_close.append(1.0 if is_positive else 0.0)
            train_mask.append(1.0 if supervised else 0.0)
            hard_negative_mask.append(1.0 if is_hard_negative else 0.0)
            close_vector_error_mm.append(
                None if sample is None else sample.get("close_vector_error_mm")
            )
            if emit_action_targets:
                raw_xyz = np.zeros(3, dtype=np.float32)
                raw_gripper = 1.0 if is_positive else 0.0
                if is_hard_negative and sample is not None and close_reference is not None:
                    current = np.asarray(sample["ee_minus_object_m"], dtype=np.float32).reshape(3)
                    desired_delta = (close_reference - current) * float(correction_gain)
                    raw_xyz = np.clip(desired_delta / float(horizon), -max_step_m, max_step_m)
                raw_action = [
                    float(raw_xyz[0]),
                    float(raw_xyz[1]),
                    float(raw_xyz[2]),
                    0.0,
                    0.0,
                    0.0,
                    float(raw_gripper),
                ]
                target_action.append(_normalize_raw_action(raw_action, action_stats))
                target_gripper_state.append([1.0 if is_positive else 0.0])
                trace_row = trace_by_step.get(step)
                target_object_target.append(_object_target_vector(trace_row, object_target_field))
                object_to_eef, target_to_eef = _object_pair_to_eef_targets(
                    trace_row,
                    object_target_field,
                )
                target_object_to_eef.append(object_to_eef)
                target_target_to_eef.append(target_to_eef)

        row = {
            "rollout_json": mined_result.get("rollout_json"),
            "checkpoint": mined_result.get("checkpoint"),
            "task_id": mined_result.get("task_id"),
            "episode_idx": mined_result.get("episode_idx"),
            "offset_cm": mined_result.get("offset_cm"),
            "chunk_index": chunk.get("chunk_index"),
            "t_start": t_start,
            "n_use": n_use,
            "npz_path": str(npz_path),
            "target_close": target_close,
            "train_mask": train_mask,
            "hard_negative_mask": hard_negative_mask,
            "close_vector_error_mm": close_vector_error_mm,
            "labels": labels,
            "rel_live": chunk.get("rel_live"),
        }
        if emit_action_targets:
            row.update(
                {
                    "target_action": target_action,
                    "target_action_normalized": True,
                    "target_gripper_state": target_gripper_state,
                    "target_contact_phase": [1 if value > 0.5 else 0 for value in target_close],
                    "target_contact_release_phase": [
                        1 if value > 0.5 else 0 for value in target_close
                    ],
                    "target_object_target": target_object_target,
                    "target_object_to_eef": target_object_to_eef,
                    "target_target_to_eef": target_target_to_eef,
                    "contact_correction_steps": int(horizon),
                    "contact_correction_max_action_step_mm": float(max_action_step_mm),
                    "contact_correction_gain": float(correction_gain),
                    "contact_correction_reference_m": close_reference_m,
                    "contact_correction_object_target_field": object_target_field,
                }
            )
        rows.append(row)
    return rows


def _markdown_row(result: dict[str, Any]) -> str:
    counts = result.get("counts", {})
    best = result.get("best_close_affordance") or {}
    hard = result.get("highest_confidence_hard_negative") or {}

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    return (
        f"| {Path(str(result['rollout_json'])).name} | {result.get('close_step')} | "
        f"{counts.get('positive', 0)} | {counts.get('missed_positive', 0)} | "
        f"{counts.get('hard_negative', 0)} | "
        f"{fmt(best.get('close_vector_error_mm'))} @ {best.get('step', '-')} | "
        f"{fmt(hard.get('close_prob'))} @ {hard.get('step', '-')} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine recorded LIBERO rollout traces for contact-head calibration labels."
    )
    parser.add_argument("--demo-geometry-json", required=True)
    parser.add_argument("--rollout-json", nargs="+", required=True)
    parser.add_argument("--save-json")
    parser.add_argument("--save-policy-manifest-jsonl")
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--close-event", default="close_phys")
    parser.add_argument("--demo-close-field", default="ee_minus_object_m")
    parser.add_argument("--ee-object-field", default="ee_minus_bowl")
    parser.add_argument("--positive-tolerance-mm", type=float, default=30.0)
    parser.add_argument(
        "--positive-component-tolerance-mm",
        type=float,
        nargs=3,
        metavar=("X_MM", "Y_MM", "Z_MM"),
        help="Optional component-wise absolute error cap for positive labels.",
    )
    parser.add_argument(
        "--best-positive-top-k-per-rollout",
        type=int,
        default=0,
        help=(
            "Optionally relabel the closest rollout states as corrective "
            "positives, bounded by --best-positive-max-error-mm."
        ),
    )
    parser.add_argument("--best-positive-max-error-mm", type=float, default=math.inf)
    parser.add_argument(
        "--best-positive-window",
        type=int,
        default=0,
        help="Also relabel +/- this many neighboring steps around each selected best positive.",
    )
    parser.add_argument("--positive-prob-margin", type=float, default=0.65)
    parser.add_argument("--hard-negative-prob", type=float, default=0.55)
    parser.add_argument(
        "--near-object-hard-negative-xy-mm",
        type=float,
        default=None,
        help="If set, label near-object non-positive states as hard negatives.",
    )
    parser.add_argument("--near-object-hard-negative-z-min-mm", type=float, default=-math.inf)
    parser.add_argument("--near-object-hard-negative-z-max-mm", type=float, default=math.inf)
    parser.add_argument(
        "--emit-correction-action-targets",
        action="store_true",
        help=(
            "When saving a policy manifest, also emit normalized target_action "
            "rows that move hard negatives toward the demo contact affordance. "
            "This is offline training supervision only."
        ),
    )
    parser.add_argument("--action-stats-json")
    parser.add_argument("--contact-correction-steps", type=int, default=4)
    parser.add_argument("--contact-correction-max-action-step-mm", type=float, default=12.0)
    parser.add_argument("--contact-correction-gain", type=float, default=1.0)
    parser.add_argument(
        "--object-target-field",
        default=None,
        help=(
            "Optional generic object-minus-target field to copy into correction "
            "manifest rows, e.g. bowl_minus_plate for the current LIBERO task."
        ),
    )
    args = parser.parse_args()

    demo_geometry = json.loads(Path(args.demo_geometry_json).read_text(encoding="utf-8"))
    close_reference = _reference_vector(demo_geometry, args.close_event, args.demo_close_field)
    action_stats = None
    if args.emit_correction_action_targets:
        if not args.save_policy_manifest_jsonl:
            raise ValueError("--emit-correction-action-targets requires --save-policy-manifest-jsonl")
        if not args.action_stats_json:
            raise ValueError("--emit-correction-action-targets requires --action-stats-json")
        action_stats = _extract_action_stats(
            json.loads(Path(args.action_stats_json).read_text(encoding="utf-8"))
        )

    results = []
    manifest_rows: list[dict[str, Any]] = []
    for path_str in args.rollout_json:
        rollout = json.loads(Path(path_str).read_text(encoding="utf-8"))
        result = mine_contact_samples(
            rollout,
            rollout_json=path_str,
            close_reference_m=close_reference,
            ee_object_field=args.ee_object_field,
            positive_tolerance_mm=float(args.positive_tolerance_mm),
            positive_component_tolerance_mm=(
                [float(value) for value in args.positive_component_tolerance_mm]
                if args.positive_component_tolerance_mm is not None
                else None
            ),
            best_positive_top_k_per_rollout=max(0, int(args.best_positive_top_k_per_rollout)),
            best_positive_max_error_mm=float(args.best_positive_max_error_mm),
            best_positive_window=max(0, int(args.best_positive_window)),
            positive_prob_margin=float(args.positive_prob_margin),
            hard_negative_prob=float(args.hard_negative_prob),
            near_object_hard_negative_xy_mm=(
                None
                if args.near_object_hard_negative_xy_mm is None
                else float(args.near_object_hard_negative_xy_mm)
            ),
            near_object_hard_negative_z_min_mm=float(args.near_object_hard_negative_z_min_mm),
            near_object_hard_negative_z_max_mm=float(args.near_object_hard_negative_z_max_mm),
        )
        results.append(result)
        manifest_rows.extend(
            policy_manifest_rows(
                rollout,
                result,
                action_stats=action_stats,
                close_reference_m=close_reference if action_stats is not None else None,
                max_action_step_mm=float(args.contact_correction_max_action_step_mm),
                correction_steps=max(1, int(args.contact_correction_steps)),
                correction_gain=float(args.contact_correction_gain),
                object_target_field=args.object_target_field,
            )
        )

    summary = {
        "demo_geometry_json": args.demo_geometry_json,
        "close_event": args.close_event,
        "demo_close_field": args.demo_close_field,
        "ee_object_field": args.ee_object_field,
        "positive_tolerance_mm": float(args.positive_tolerance_mm),
        "positive_component_tolerance_mm": (
            [float(value) for value in args.positive_component_tolerance_mm]
            if args.positive_component_tolerance_mm is not None
            else None
        ),
        "best_positive_top_k_per_rollout": max(0, int(args.best_positive_top_k_per_rollout)),
        "best_positive_max_error_mm": float(args.best_positive_max_error_mm),
        "best_positive_window": max(0, int(args.best_positive_window)),
        "positive_prob_margin": float(args.positive_prob_margin),
        "hard_negative_prob": float(args.hard_negative_prob),
        "near_object_hard_negative_xy_mm": (
            None
            if args.near_object_hard_negative_xy_mm is None
            else float(args.near_object_hard_negative_xy_mm)
        ),
        "near_object_hard_negative_z_min_mm": float(args.near_object_hard_negative_z_min_mm),
        "near_object_hard_negative_z_max_mm": float(args.near_object_hard_negative_z_max_mm),
        "emit_correction_action_targets": bool(args.emit_correction_action_targets),
        "contact_correction_steps": max(1, int(args.contact_correction_steps)),
        "contact_correction_max_action_step_mm": float(args.contact_correction_max_action_step_mm),
        "contact_correction_gain": float(args.contact_correction_gain),
        "object_target_field": args.object_target_field,
        "policy_manifest_rows": int(len(manifest_rows)),
        "close_reference_m": close_reference,
        "results": results,
    }
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.save_policy_manifest_jsonl:
        manifest_path = Path(args.save_policy_manifest_jsonl)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            for row in manifest_rows:
                f.write(json.dumps(row) + "\n")

    if args.print_markdown:
        print("| rollout | close_step | positives | missed positives | hard negatives | best affordance err mm | max hard-neg prob |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for result in results:
            print(_markdown_row(result))


if __name__ == "__main__":
    main()
