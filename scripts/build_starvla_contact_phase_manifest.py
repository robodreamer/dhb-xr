#!/usr/bin/env python3
"""Build a three-class gripper contact-phase manifest for StarVLA head tuning.

The policy runtime can decode gripper commands from ``dhb.contact_phase``:

* 0: no event / open approach
* 1: close or keep closed
* 2: release / open after placement

This script derives that target from existing policy-trace contact manifests
without using object-specific distance thresholds. Close/closed labels come
from target_close, target_gripper_state, or positive labels. Release labels
come from explicit release labels or existing release-phase targets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _scalar_at(values: list[Any], index: int) -> float | None:
    if index >= len(values):
        return None
    value = values[index]
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _label_at(labels: list[Any], index: int) -> str:
    if index >= len(labels):
        return ""
    return str(labels[index]).strip().lower()


def _is_release_label(label: str) -> bool:
    return "release" in label or "open_after_place" in label


def _is_positive_label(label: str) -> bool:
    if not label:
        return False
    if "negative" in label or "background" in label or "padding" in label:
        return False
    return "positive" in label or "contact" in label or "transport" in label


def infer_contact_phase(row: dict[str, Any]) -> list[int]:
    labels = _as_list(row.get("labels"))
    target_close = _as_list(row.get("target_close"))
    target_gripper_state = _as_list(row.get("target_gripper_state"))
    release_phase = _as_list(row.get("target_contact_release_phase"))
    train_mask = _as_list(row.get("train_mask"))
    n_use = int(row.get("n_use") or 0)
    n = max(
        n_use,
        len(labels),
        len(target_close),
        len(target_gripper_state),
        len(release_phase),
        len(train_mask),
    )
    phases: list[int] = []
    for index in range(n):
        label = _label_at(labels, index)
        release_value = _scalar_at(release_phase, index)
        if release_value is not None and int(round(release_value)) == 2:
            phases.append(2)
            continue
        if _is_release_label(label):
            phases.append(2)
            continue

        gripper_state = _scalar_at(target_gripper_state, index)
        if gripper_state is not None:
            phases.append(1 if gripper_state > 0.5 else 0)
            continue

        close = _scalar_at(target_close, index)
        if close is not None:
            phases.append(1 if close > 0.5 else 0)
            continue

        phases.append(1 if _is_positive_label(label) else 0)
    return phases


def _has_phase(row: dict[str, Any], phase: int) -> bool:
    phases = row.get("target_contact_phase", [])
    mask = row.get("train_mask", [])
    for index, value in enumerate(phases):
        if index < len(mask):
            try:
                if float(mask[index]) <= 0.0:
                    continue
            except (TypeError, ValueError):
                continue
        if int(value) == phase:
            return True
    return False


def build_manifest(
    input_jsonl: list[Path],
    output_jsonl: Path,
    *,
    release_repeat: int,
    closed_repeat: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    for path in input_jsonl:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row = dict(row)
            row["target_contact_phase"] = infer_contact_phase(row)
            row["target_contact_phase_semantics"] = {
                "0": "no_event_or_open_approach",
                "1": "close_or_keep_closed",
                "2": "release_or_open_after_place",
            }
            source_counts[path.name] += 1
            repeat = 1
            if _has_phase(row, 2):
                repeat = max(repeat, int(release_repeat))
            elif _has_phase(row, 1):
                repeat = max(repeat, int(closed_repeat))
            for _ in range(repeat):
                rows.append(dict(row))
            mask = row.get("train_mask", [])
            for index, phase in enumerate(row["target_contact_phase"]):
                if index < len(mask):
                    try:
                        if float(mask[index]) <= 0.0:
                            continue
                    except (TypeError, ValueError):
                        continue
                phase_counts[str(int(phase))] += repeat

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return {
        "output_jsonl": str(output_jsonl),
        "input_jsonl": [str(path) for path in input_jsonl],
        "input_row_counts": dict(source_counts),
        "rows": int(len(rows)),
        "release_repeat": int(release_repeat),
        "closed_repeat": int(closed_repeat),
        "phase_counts": dict(sorted(phase_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--release-repeat", type=int, default=8)
    parser.add_argument("--closed-repeat", type=int, default=1)
    args = parser.parse_args()

    summary = build_manifest(
        [Path(path) for path in args.input_jsonl],
        Path(args.output_jsonl),
        release_repeat=max(1, int(args.release_repeat)),
        closed_repeat=max(1, int(args.closed_repeat)),
    )
    if args.summary_json:
        out = Path(args.summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
