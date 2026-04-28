#!/usr/bin/env python3
"""Build reset/replay plans for post-contact LIBERO correction rollouts.

This is data-generation infrastructure, not a deployment heuristic. It consumes
transport-miner correction requests and recorded rollout step traces. When the
source rollout was captured with ``--record-sim-state-trace``, each plan row
contains the exact simulator state needed to reset LIBERO to a physically closed
post-contact state before generating successful target-reaching positives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _vector3(value: Any, *, field: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{field} must be a 3-vector, got shape {tuple(arr.shape)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field} contains non-finite values: {value}")
    return arr


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def step_entry_by_step(rollout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    trace = rollout.get("step_trace")
    if not isinstance(trace, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for entry in trace:
        if not isinstance(entry, dict):
            continue
        step = entry.get("step")
        if isinstance(step, int):
            result[step] = entry
    return result


def scripted_closed_correction_actions(
    object_minus_target_m: Any,
    *,
    correction_steps: int,
    max_action_step_mm: float,
    include_z: bool,
) -> np.ndarray:
    """Return closed-gripper delta actions that move along target direction."""
    horizon = max(1, int(correction_steps))
    max_step_m = max(0.0, float(max_action_step_mm)) / 1000.0
    desired_delta = -_vector3(object_minus_target_m, field="object_minus_target_m")
    if not include_z:
        desired_delta[2] = 0.0
    per_step = desired_delta / float(horizon)
    if max_step_m > 0.0:
        per_step = np.clip(per_step, -max_step_m, max_step_m)
    action = np.zeros((horizon, 7), dtype=np.float32)
    action[:, :3] = per_step.astype(np.float32)
    action[:, 6] = 1.0
    return action


def build_plan_rows(
    rollout: dict[str, Any],
    requests: list[dict[str, Any]],
    *,
    source_rollout_json: str,
    correction_steps: int,
    max_action_step_mm: float,
    include_z: bool,
) -> list[dict[str, Any]]:
    entries = step_entry_by_step(rollout)
    rows: list[dict[str, Any]] = []
    for request in requests:
        if str(request.get("rollout_json")) != source_rollout_json:
            continue
        step = request.get("step")
        if not isinstance(step, int):
            continue
        entry = entries.get(step)
        actions = scripted_closed_correction_actions(
            request.get("object_minus_target_m"),
            correction_steps=correction_steps,
            max_action_step_mm=max_action_step_mm,
            include_z=include_z,
        )
        sim_state = None if entry is None else entry.get("sim_state")
        sim_state_dim = None
        if sim_state is not None:
            sim_state_arr = np.asarray(sim_state, dtype=np.float32).reshape(-1)
            sim_state = sim_state_arr.tolist()
            sim_state_dim = int(sim_state_arr.size)
        rows.append(
            {
                "source_rollout_json": source_rollout_json,
                "checkpoint": request.get("checkpoint") or rollout.get("checkpoint"),
                "task_suite_name": rollout.get("task_suite_name"),
                "task_id": request.get("task_id", rollout.get("task_id")),
                "episode_idx": request.get("episode_idx", rollout.get("episode_idx")),
                "offset_cm": request.get("offset_cm", rollout.get("offset_cm")),
                "step": int(step),
                "object_target_field": request.get("object_target_field"),
                "object_minus_target_m": request.get("object_minus_target_m"),
                "desired_object_delta_to_target_m": request.get(
                    "desired_object_delta_to_target_m"
                ),
                "object_target_xy_mm": request.get("object_target_xy_mm"),
                "has_step_trace_entry": entry is not None,
                "has_sim_state": sim_state is not None,
                "sim_state_dim": sim_state_dim,
                "sim_state": sim_state,
                "raw_correction_actions": actions.tolist(),
                "raw_action_convention": "[dx, dy, dz, d_rx, d_ry, d_rz, closed_gripper]",
                "correction_steps": max(1, int(correction_steps)),
                "max_action_step_mm": float(max_action_step_mm),
                "correction_include_z": bool(include_z),
                "expected_ee_delta_m": actions[:, :3].sum(axis=0).astype(np.float32).tolist(),
                "needs_sim_state_capture": sim_state is None,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build post-contact reset/replay correction plans from transport-miner "
            "requests. Source rollouts should be captured with "
            "--record-sim-state-trace before executing the reset/replay pass."
        )
    )
    parser.add_argument("--source-rollout-json", required=True)
    parser.add_argument("--correction-jsonl", required=True)
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--max-requests", type=int, default=8)
    parser.add_argument("--correction-steps", type=int, default=16)
    parser.add_argument("--max-action-step-mm", type=float, default=10.0)
    parser.add_argument("--include-z", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source_rollout_json)
    rollout = json.loads(source_path.read_text(encoding="utf-8"))
    requests = load_jsonl(args.correction_jsonl)
    rows = build_plan_rows(
        rollout,
        requests,
        source_rollout_json=str(source_path),
        correction_steps=max(1, int(args.correction_steps)),
        max_action_step_mm=float(args.max_action_step_mm),
        include_z=bool(args.include_z),
    )
    rows = rows[: max(0, int(args.max_requests))]
    payload = {
        "source_rollout_json": str(source_path),
        "correction_jsonl": str(args.correction_jsonl),
        "num_rows": int(len(rows)),
        "num_with_sim_state": int(sum(1 for row in rows if row["has_sim_state"])),
        "num_needing_sim_state_capture": int(
            sum(1 for row in rows if row["needs_sim_state_capture"])
        ),
        "rows": rows,
    }
    out = Path(args.save_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
