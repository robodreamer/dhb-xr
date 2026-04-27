#!/usr/bin/env python3
"""Mine closed-loop LIBERO rollout traces for release-head calibration samples.

This is training-only infrastructure. It converts recorded rollout geometry
into dense gripper-state labels for policy-trace chunks:

- closed/post-grasp states far from the target should stay closed;
- closed/post-grasp states near the target should reopen.

The rollout policy still receives only observations, language, and robot state.
No object-distance gates are added to deployment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _vector3(value: Any, *, field: str) -> list[float] | None:
    if not isinstance(value, list | tuple) or len(value) < 3:
        return None
    return [float(value[0]), float(value[1]), float(value[2])]


def _xy_norm(vec: list[float]) -> float:
    return math.hypot(vec[0], vec[1])


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


def _classify_release_sample(
    *,
    after_physical_close: bool,
    release_ready: bool,
    observed_closed: bool,
    desired_open: bool,
    open_prob: float | None,
    open_prob_threshold: float,
) -> str:
    if not after_physical_close:
        return "background"
    if release_ready:
        return "release_positive"
    if observed_closed or desired_open or (open_prob is not None and open_prob >= open_prob_threshold):
        return "closed_hard_negative"
    return "closed_background"


def mine_release_samples(
    rollout: dict[str, Any],
    *,
    rollout_json: str,
    object_target_field: str,
    release_xy_tolerance_mm: float,
    release_z_min_mm: float,
    release_z_max_mm: float,
    best_release_top_k_per_rollout: int,
    best_release_max_xy_mm: float,
    best_release_window: int,
    open_prob_threshold: float,
    gripper_qpos_threshold: float,
) -> dict[str, Any]:
    trace = [row for row in rollout.get("step_trace", []) if isinstance(row, dict)]
    chunks = [row for row in rollout.get("chunks", []) if isinstance(row, dict)]
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

    open_probs = _chunk_prob_map(chunks, "open_probs")
    close_probs = _chunk_prob_map(chunks, "close_probs")
    candidates: list[dict[str, Any]] = []
    for row in trace:
        step = row.get("step")
        if not isinstance(step, int):
            continue
        vector = _vector3(row.get(object_target_field), field=object_target_field)
        if vector is None:
            continue
        candidates.append(
            {
                "row": row,
                "step": step,
                "vector": vector,
                "xy_mm": _xy_norm(vector) * 1000.0,
                "z_mm": vector[2] * 1000.0,
            }
        )

    forced_release_steps: set[int] = set()
    if best_release_top_k_per_rollout > 0:
        candidate_steps = {int(candidate["step"]) for candidate in candidates}
        sorted_candidates = sorted(
            (
                candidate
                for candidate in candidates
                if physical_close is not None and int(candidate["step"]) >= physical_close
            ),
            key=lambda item: float(item["xy_mm"]),
        )
        selected = [
            candidate
            for candidate in sorted_candidates
            if float(candidate["xy_mm"]) <= float(best_release_max_xy_mm)
        ][: int(best_release_top_k_per_rollout)]
        for candidate in selected:
            step = int(candidate["step"])
            for offset in range(-int(best_release_window), int(best_release_window) + 1):
                forced_step = step + offset
                if forced_step in candidate_steps:
                    forced_release_steps.add(forced_step)

    samples: list[dict[str, Any]] = []
    for candidate in candidates:
        row = candidate["row"]
        step = int(candidate["step"])
        vector = candidate["vector"]
        after_physical_close = physical_close is not None and step >= physical_close
        release_ready = (
            after_physical_close
            and candidate["xy_mm"] <= release_xy_tolerance_mm
            and release_z_min_mm <= candidate["z_mm"] <= release_z_max_mm
        )
        is_forced_best_release = step in forced_release_steps
        release_ready = release_ready or is_forced_best_release
        observed_value = row.get("observed_state_gripper", row.get("state_gripper"))
        observed_closed = False
        try:
            observed_closed = observed_value is not None and float(observed_value) <= gripper_qpos_threshold
        except (TypeError, ValueError):
            observed_closed = False
        desired_grip = float(row.get("desired_grip", -1.0) or -1.0)
        effective_grip = float(row.get("effective_grip", -1.0) or -1.0)
        desired_open = desired_grip < 0.0 and effective_grip < 0.0 and reopen_step is not None and step >= reopen_step
        label = _classify_release_sample(
            after_physical_close=after_physical_close,
            release_ready=release_ready,
            observed_closed=observed_closed,
            desired_open=desired_open,
            open_prob=open_probs.get(step),
            open_prob_threshold=open_prob_threshold,
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
                "object_target_field": object_target_field,
                "object_minus_target_m": vector,
                "object_target_xy_mm": candidate["xy_mm"],
                "object_target_z_mm": candidate["z_mm"],
                "target_closed": 0.0 if label == "release_positive" else 1.0,
                "train_mask": 1.0 if label in {"release_positive", "closed_hard_negative"} else 0.0,
                "is_forced_best_release": is_forced_best_release,
                "close_prob": close_probs.get(step),
                "open_prob": open_probs.get(step),
                "desired_grip": desired_grip,
                "effective_grip": effective_grip,
                "state_gripper": row.get("state_gripper"),
                "observed_state_gripper": row.get("observed_state_gripper"),
            }
        )

    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample["label"]] = counts.get(sample["label"], 0) + 1

    best = min(
        (row for row in samples if physical_close is not None and int(row["step"]) >= physical_close),
        key=lambda row: row["object_target_xy_mm"],
        default=None,
    )
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
        "num_samples": len(samples),
        "counts": counts,
        "best_release_candidate": best,
        "samples": samples,
    }


def policy_manifest_rows(
    rollout: dict[str, Any],
    mined_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build chunk-level dense gripper-state rows for policy-trace NPZ files."""
    policy_trace = rollout.get("policy_trace")
    if not isinstance(policy_trace, list):
        return []

    sample_by_step = {
        sample["step"]: sample
        for sample in mined_result.get("samples", [])
        if isinstance(sample, dict) and isinstance(sample.get("step"), int)
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

        target_closed: list[float] = []
        train_mask: list[float] = []
        object_target_xy_mm: list[float | None] = []
        labels: list[str | None] = []
        for step in range(t_start, t_start + n_use):
            sample = sample_by_step.get(step)
            label = sample.get("label") if sample else None
            labels.append(label)
            target_closed.append(float(sample.get("target_closed", 1.0)) if sample else 1.0)
            train_mask.append(float(sample.get("train_mask", 0.0)) if sample else 0.0)
            object_target_xy_mm.append(
                None if sample is None else sample.get("object_target_xy_mm")
            )

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
                "target_close": target_closed,
                "train_mask": train_mask,
                "object_target_xy_mm": object_target_xy_mm,
                "labels": labels,
                "rel_live": chunk.get("rel_live"),
            }
        )
    return rows


def _markdown_row(result: dict[str, Any]) -> str:
    counts = result.get("counts", {})
    best = result.get("best_release_candidate") or {}

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    return (
        f"| {Path(str(result['rollout_json'])).name} | {result.get('physical_close_step')} | "
        f"{result.get('reopen_step')} | {counts.get('release_positive', 0)} | "
        f"{counts.get('closed_hard_negative', 0)} | "
        f"{fmt(best.get('object_target_xy_mm'))} @ {best.get('step', '-')} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine recorded LIBERO rollout traces for release/gripper-state labels."
    )
    parser.add_argument("--rollout-json", nargs="+", required=True)
    parser.add_argument("--save-json")
    parser.add_argument("--save-policy-manifest-jsonl")
    parser.add_argument("--print-markdown", action="store_true")
    parser.add_argument("--object-target-field", default="bowl_minus_plate")
    parser.add_argument("--release-xy-tolerance-mm", type=float, default=40.0)
    parser.add_argument("--release-z-min-mm", type=float, default=-20.0)
    parser.add_argument("--release-z-max-mm", type=float, default=40.0)
    parser.add_argument("--best-release-top-k-per-rollout", type=int, default=0)
    parser.add_argument("--best-release-max-xy-mm", type=float, default=math.inf)
    parser.add_argument("--best-release-window", type=int, default=0)
    parser.add_argument("--open-prob-threshold", type=float, default=0.50)
    parser.add_argument("--gripper-qpos-threshold", type=float, default=0.015)
    args = parser.parse_args()

    results = []
    manifest_rows: list[dict[str, Any]] = []
    for path_str in args.rollout_json:
        rollout = json.loads(Path(path_str).read_text(encoding="utf-8"))
        result = mine_release_samples(
            rollout,
            rollout_json=path_str,
            object_target_field=args.object_target_field,
            release_xy_tolerance_mm=float(args.release_xy_tolerance_mm),
            release_z_min_mm=float(args.release_z_min_mm),
            release_z_max_mm=float(args.release_z_max_mm),
            best_release_top_k_per_rollout=max(0, int(args.best_release_top_k_per_rollout)),
            best_release_max_xy_mm=float(args.best_release_max_xy_mm),
            best_release_window=max(0, int(args.best_release_window)),
            open_prob_threshold=float(args.open_prob_threshold),
            gripper_qpos_threshold=float(args.gripper_qpos_threshold),
        )
        results.append(result)
        manifest_rows.extend(policy_manifest_rows(rollout, result))

    summary = {
        "object_target_field": args.object_target_field,
        "release_xy_tolerance_mm": float(args.release_xy_tolerance_mm),
        "release_z_min_mm": float(args.release_z_min_mm),
        "release_z_max_mm": float(args.release_z_max_mm),
        "best_release_top_k_per_rollout": max(0, int(args.best_release_top_k_per_rollout)),
        "best_release_max_xy_mm": float(args.best_release_max_xy_mm),
        "best_release_window": max(0, int(args.best_release_window)),
        "open_prob_threshold": float(args.open_prob_threshold),
        "gripper_qpos_threshold": float(args.gripper_qpos_threshold),
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
        print("| rollout | physical close step | reopen step | release positives | closed hard negatives | best target XY mm |")
        print("| --- | ---: | ---: | ---: | ---: | ---: |")
        for result in results:
            print(_markdown_row(result))


if __name__ == "__main__":
    main()
