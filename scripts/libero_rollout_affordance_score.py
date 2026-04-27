from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _vector3(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) < 3:
        raise ValueError(f"{field} must be a vector with at least 3 values")
    return [float(value[0]), float(value[1]), float(value[2])]


def _mean(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        raise ValueError("cannot average an empty vector list")
    return [sum(vec[i] for vec in vectors) / len(vectors) for i in range(3)]


def _std(vectors: list[list[float]], mean: list[float]) -> list[float]:
    if not vectors:
        raise ValueError("cannot compute std of an empty vector list")
    return [
        math.sqrt(sum((vec[i] - mean[i]) ** 2 for vec in vectors) / len(vectors))
        for i in range(3)
    ]


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vec))


def _xy_norm(vec: list[float]) -> float:
    return math.hypot(vec[0], vec[1])


def _sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def _mm(vec_or_value: list[float] | float) -> list[float] | float:
    if isinstance(vec_or_value, list):
        return [value * 1000.0 for value in vec_or_value]
    return vec_or_value * 1000.0


def _event_vectors(
    demo_geometry: dict[str, Any], event_name: str, key: str
) -> list[list[float]]:
    rows = [
        row
        for row in demo_geometry.get("events", [])
        if row.get("event") == event_name and key in row
    ]
    return [_vector3(row[key], field=f"{event_name}.{key}") for row in rows]


def reference_stats(
    demo_geometry: dict[str, Any], event_name: str, key: str
) -> dict[str, Any]:
    vectors = _event_vectors(demo_geometry, event_name, key)
    if not vectors:
        raise ValueError(f"no demo event vectors found for event={event_name} key={key}")
    mean = _mean(vectors)
    std = _std(vectors, mean)
    xy = [_xy_norm(vec) for vec in vectors]
    return {
        "event": event_name,
        "key": key,
        "count": len(vectors),
        "mean_m": mean,
        "std_m": std,
        "mean_mm": _mm(mean),
        "std_mm": _mm(std),
        "xy_mean_m": sum(xy) / len(xy),
        "xy_std_m": math.sqrt(sum((value - (sum(xy) / len(xy))) ** 2 for value in xy) / len(xy)),
    }


def _snapshot_for_step(
    rollout: dict[str, Any], step_key: str, snapshot_key: str
) -> dict[str, Any] | None:
    snapshot = rollout.get(snapshot_key)
    if isinstance(snapshot, dict):
        return snapshot

    step = rollout.get(step_key)
    trace = rollout.get("step_trace") or []
    if step is None or not isinstance(trace, list):
        return None
    for record in trace:
        if record.get("step") == step:
            return record
    if isinstance(step, int) and 0 <= step < len(trace):
        return trace[step]
    return None


def _field_vector(snapshot: dict[str, Any] | None, field: str) -> list[float] | None:
    if not isinstance(snapshot, dict) or field not in snapshot:
        return None
    return _vector3(snapshot[field], field=field)


def _trace_records(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    trace = rollout.get("step_trace") or []
    return [record for record in trace if isinstance(record, dict)]


def _record_for_step(records: list[dict[str, Any]], step: int | None) -> dict[str, Any] | None:
    if not isinstance(step, int):
        return None
    for record in records:
        if record.get("step") == step:
            return record
    if 0 <= step < len(records):
        return records[step]
    return None


def _physical_close_step(
    records: list[dict[str, Any]],
    *,
    command_close_step: int | None,
    gripper_qpos_threshold: float,
) -> int | None:
    """Return the first observed closed-qpos step after the close command.

    Rollout traces distinguish command timing from physical gripper closure via
    ``observed_state_gripper``. The demo close reference is a physical close
    event, so affordance scoring should use this delayed crossing when it is
    available.
    """
    if not isinstance(command_close_step, int):
        return None
    def sort_key(item: dict[str, Any]) -> int:
        step = item.get("step")
        return int(step) if isinstance(step, int) else 10**9

    for record in sorted(records, key=sort_key):
        step = record.get("step")
        if not isinstance(step, int) or step < command_close_step:
            continue
        value = record.get("observed_state_gripper")
        if value is None:
            continue
        try:
            if float(value) <= gripper_qpos_threshold:
                return step
        except (TypeError, ValueError):
            continue
    return None


def _best_vector_match(
    records: list[dict[str, Any]],
    field: str,
    reference: list[float],
    *,
    start_step: int | None = None,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for record in records:
        step = record.get("step")
        if start_step is not None and isinstance(step, int) and step < start_step:
            continue
        vector = _field_vector(record, field)
        if vector is None:
            continue
        error = _norm(_sub(vector, reference))
        if best is None or error < best["vector_error_m"]:
            best = {
                "step": step,
                "vector_m": vector,
                "vector_mm": _mm(vector),
                "vector_error_m": error,
                "vector_error_mm": _mm(error),
                "xy_m": _xy_norm(vector),
                "xy_mm": _mm(_xy_norm(vector)),
            }
    return best


def _best_xy(
    records: list[dict[str, Any]],
    field: str,
    *,
    start_step: int | None = None,
) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for record in records:
        step = record.get("step")
        if start_step is not None and isinstance(step, int) and step < start_step:
            continue
        vector = _field_vector(record, field)
        if vector is None:
            continue
        xy = _xy_norm(vector)
        if best is None or xy < best["xy_m"]:
            best = {
                "step": step,
                "vector_m": vector,
                "vector_mm": _mm(vector),
                "xy_m": xy,
                "xy_mm": _mm(xy),
            }
    return best


def _score_event_vector(
    snapshot: dict[str, Any] | None,
    field: str,
    reference: dict[str, Any],
) -> dict[str, Any] | None:
    vector = _field_vector(snapshot, field)
    if vector is None:
        return None
    ref = reference["mean_m"]
    delta = _sub(vector, ref)
    return {
        "vector_m": vector,
        "vector_mm": _mm(vector),
        "reference_mean_mm": reference["mean_mm"],
        "component_error_mm": _mm(delta),
        "vector_error_m": _norm(delta),
        "vector_error_mm": _mm(_norm(delta)),
        "xy_m": _xy_norm(vector),
        "xy_mm": _mm(_xy_norm(vector)),
        "reference_xy_mean_mm": _mm(reference["xy_mean_m"]),
        "xy_error_to_demo_mean_mm": _mm(abs(_xy_norm(vector) - reference["xy_mean_m"])),
    }


def score_rollout(
    rollout: dict[str, Any],
    demo_geometry: dict[str, Any],
    *,
    close_event: str = "close_phys",
    release_event: str = "open_phys",
    ee_object_field: str = "ee_minus_bowl",
    object_target_field: str = "bowl_minus_plate",
    close_vector_tolerance_mm: float = 30.0,
    release_xy_tolerance_mm: float = 40.0,
    gripper_qpos_threshold: float = 0.015,
) -> dict[str, Any]:
    close_ref = reference_stats(demo_geometry, close_event, "ee_minus_object_m")
    release_ref = reference_stats(demo_geometry, release_event, "object_minus_target_m")
    close_step = rollout.get("close_step")
    reopen_step = rollout.get("reopen_step")
    final_snapshot = rollout.get("final_snapshot")
    records = _trace_records(rollout)
    physical_close_step = _physical_close_step(
        records,
        command_close_step=close_step if isinstance(close_step, int) else None,
        gripper_qpos_threshold=gripper_qpos_threshold,
    )
    close_snapshot = (
        _record_for_step(records, physical_close_step)
        if physical_close_step is not None
        else _snapshot_for_step(rollout, "close_step", "close_snapshot")
    )
    close_command_snapshot = _snapshot_for_step(rollout, "close_step", "close_snapshot")
    reopen_snapshot = _snapshot_for_step(rollout, "reopen_step", "reopen_snapshot")

    close_score = _score_event_vector(close_snapshot, ee_object_field, close_ref)
    close_command_score = _score_event_vector(close_command_snapshot, ee_object_field, close_ref)
    reopen_score = _score_event_vector(reopen_snapshot, object_target_field, release_ref)
    final_release_score = _score_event_vector(final_snapshot, object_target_field, release_ref)
    best_close_match = _best_vector_match(records, ee_object_field, close_ref["mean_m"])
    best_release_after_close = _best_xy(
        records,
        object_target_field,
        start_step=close_step if isinstance(close_step, int) else None,
    )

    gates = {
        "has_close": close_step is not None,
        "has_reopen": reopen_step is not None,
        "close_affordance_pass": bool(
            close_score is not None
            and close_score["vector_error_mm"] <= close_vector_tolerance_mm
        ),
        "release_xy_pass": bool(
            (
                reopen_score is not None
                and reopen_score["xy_mm"] <= release_xy_tolerance_mm
            )
            or (
                reopen_score is None
                and final_release_score is not None
                and final_release_score["xy_mm"] <= release_xy_tolerance_mm
            )
        ),
        "success": bool(rollout.get("success")),
    }

    return {
        "task_suite_name": rollout.get("task_suite_name"),
        "task_id": rollout.get("task_id"),
        "episode_idx": rollout.get("episode_idx"),
        "offset_cm": rollout.get("offset_cm"),
        "checkpoint": rollout.get("checkpoint"),
        "close_step": close_step,
        "physical_close_step": physical_close_step,
        "reopen_step": reopen_step,
        "success": bool(rollout.get("success")),
        "ik_fail_total": int(rollout.get("ik_fail_total") or 0),
        "min_ee_object_xy_mm": _mm(float(rollout.get("min_bowl_xy_m") or 0.0)),
        "min_ee_object_xyz_mm": _mm(float(rollout.get("min_bowl_xyz_m") or 0.0)),
        "reference": {
            "close": close_ref,
            "release": release_ref,
            "close_vector_tolerance_mm": close_vector_tolerance_mm,
            "release_xy_tolerance_mm": release_xy_tolerance_mm,
        },
        "events": {
            "close": close_score,
            "close_command": close_command_score,
            "reopen": reopen_score,
            "final": final_release_score,
        },
        "best": {
            "close_affordance_match": best_close_match,
            "release_xy_after_close": best_release_after_close,
        },
        "gates": gates,
    }


def _markdown_row(path: str, score: dict[str, Any]) -> str:
    close = score["events"]["close"] or {}
    close_command = score["events"].get("close_command") or {}
    reopen = score["events"]["reopen"] or {}
    final = score["events"]["final"] or {}
    best_close = score["best"]["close_affordance_match"] or {}
    best_release = score["best"]["release_xy_after_close"] or {}

    def fmt(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    return (
        f"| {Path(path).name} | {score['close_step']} | {score.get('physical_close_step')} | "
        f"{fmt(close_command.get('vector_error_mm'))} | "
        f"{fmt(close.get('vector_error_mm'))} | {fmt(close.get('xy_mm'))} | "
        f"{fmt(best_close.get('vector_error_mm'))} @ {best_close.get('step', '-')} | "
        f"{score['reopen_step']} | {fmt(reopen.get('xy_mm'))} | "
        f"{fmt(final.get('xy_mm'))} | {fmt(best_release.get('xy_mm'))} @ "
        f"{best_release.get('step', '-')} | {score['ik_fail_total']} | "
        f"{score['gates']['close_affordance_pass']} | "
        f"{score['gates']['release_xy_pass']} | |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score LIBERO rollout traces against demo-derived grasp/release affordance geometry."
    )
    parser.add_argument("--demo-geometry-json", required=True)
    parser.add_argument("--rollout-json", nargs="+", required=True)
    parser.add_argument("--close-event", default="close_phys")
    parser.add_argument("--release-event", default="open_phys")
    parser.add_argument("--ee-object-field", default="ee_minus_bowl")
    parser.add_argument("--object-target-field", default="bowl_minus_plate")
    parser.add_argument("--close-vector-tolerance-mm", type=float, default=30.0)
    parser.add_argument("--release-xy-tolerance-mm", type=float, default=40.0)
    parser.add_argument("--gripper-qpos-threshold", type=float, default=0.015)
    parser.add_argument("--save-json", default="")
    parser.add_argument("--print-markdown", action="store_true")
    args = parser.parse_args()

    demo_geometry = json.loads(Path(args.demo_geometry_json).read_text(encoding="utf-8"))
    scores = []
    for rollout_path in args.rollout_json:
        rollout = json.loads(Path(rollout_path).read_text(encoding="utf-8"))
        score = score_rollout(
            rollout,
            demo_geometry,
            close_event=args.close_event,
            release_event=args.release_event,
            ee_object_field=args.ee_object_field,
            object_target_field=args.object_target_field,
            close_vector_tolerance_mm=args.close_vector_tolerance_mm,
            release_xy_tolerance_mm=args.release_xy_tolerance_mm,
            gripper_qpos_threshold=args.gripper_qpos_threshold,
        )
        score["rollout_json"] = rollout_path
        scores.append(score)

    payload = {
        "demo_geometry_json": args.demo_geometry_json,
        "scores": scores,
    }
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.print_markdown:
        print(
            "| Rollout | Command close step | Physical close step | "
            "Command close vector err mm | Physical close vector err mm | Close XY mm | "
            "Best close err mm | Reopen step | Reopen target XY mm | "
            "Final target XY mm | Best target XY mm | IK fails | "
            "Close pass | Release pass | Notes |"
        )
        print(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | --- | --- | --- |"
        )
        for score in scores:
            print(_markdown_row(score["rollout_json"], score))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
