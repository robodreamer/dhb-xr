#!/usr/bin/env python3
"""Add chunk-local DHB invariant targets to policy-trace manifests.

This is training/data preparation infrastructure. It does not change policy
deployment. Existing attached-positive manifests can be retrofitted without
rerunning LIBERO by denormalizing their saved target actions, reading the
chunk-start state from the trace NPZ, and encoding the resulting local action
chunk into DHB targets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.libero_rollout_transport_miner import _extract_action_stats

try:
    from starVLA.dataloader.gr00t_lerobot.datasets import (
        build_chunk_local_dhb_targets_from_delta_actions,
    )
except ImportError:
    import sys

    _vendor_starvla = Path(__file__).resolve().parents[1] / "vendor" / "starVLA"
    if _vendor_starvla.exists() and str(_vendor_starvla) not in sys.path:
        sys.path.insert(0, str(_vendor_starvla))
    from starVLA.dataloader.gr00t_lerobot.datasets import (
        build_chunk_local_dhb_targets_from_delta_actions,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def _denormalize_action(action: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    low = np.asarray(action_stats["min"], dtype=np.float32).reshape(-1)
    high = np.asarray(action_stats["max"], dtype=np.float32).reshape(-1)
    mask = np.asarray(action_stats.get("mask", np.ones_like(low, dtype=bool)), dtype=bool)
    if action.shape[-1] > low.shape[0]:
        raise ValueError(
            f"Action dim {action.shape[-1]} exceeds statistics dim {low.shape[0]}"
        )
    low = low[: action.shape[-1]]
    high = high[: action.shape[-1]]
    mask = mask[: action.shape[-1]]
    denom = np.where(np.abs(high - low) < 1e-8, 1.0, high - low)
    raw = (action + 1.0) * 0.5 * denom + low
    return np.where(mask, raw, action).astype(np.float32)


def _row_current_state(row: dict[str, Any]) -> np.ndarray:
    npz_path = Path(str(row.get("npz_path", "")))
    if not npz_path.exists():
        raise FileNotFoundError(f"policy trace NPZ not found: {npz_path}")
    with np.load(npz_path) as npz:
        if "state_window" not in npz:
            raise KeyError(f"{npz_path} does not contain state_window")
        state = np.asarray(npz["state_window"], dtype=np.float32)
    if state.ndim == 2:
        return state[-1].reshape(-1)
    return state.reshape(-1)


def augment_row(row: dict[str, Any], *, action_stats: dict[str, Any]) -> dict[str, Any]:
    action = np.asarray(row.get("target_action", []), dtype=np.float32)
    if action.ndim != 2 or action.shape[0] == 0:
        raise ValueError("manifest row is missing 2-D target_action")
    if bool(row.get("target_action_normalized", False)):
        action = _denormalize_action(action, action_stats)

    current_state = _row_current_state(row)
    linear, angular = build_chunk_local_dhb_targets_from_delta_actions(current_state, action)

    out = dict(row)
    out["target_linear_invariants"] = linear.astype(np.float32).tolist()
    out["target_angular_invariants"] = angular.astype(np.float32).tolist()
    out["target_dhb_invariants_chunk_local"] = True
    return out


def augment_manifest(
    manifest_jsonl: Path,
    output_jsonl: Path,
    *,
    action_stats_json: Path,
) -> dict[str, Any]:
    rows = _read_jsonl(manifest_jsonl)
    action_stats = _extract_action_stats(json.loads(action_stats_json.read_text(encoding="utf-8")))
    augmented = [augment_row(row, action_stats=action_stats) for row in rows]
    _write_jsonl(output_jsonl, augmented)
    invariant_steps = sum(len(row.get("target_linear_invariants", [])) for row in augmented)
    return {
        "input_jsonl": str(manifest_jsonl),
        "output_jsonl": str(output_jsonl),
        "rows": int(len(augmented)),
        "invariant_steps": int(invariant_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--action-stats-json", required=True)
    parser.add_argument("--save-summary-json")
    args = parser.parse_args()

    summary = augment_manifest(
        Path(args.manifest_jsonl),
        Path(args.output_jsonl),
        action_stats_json=Path(args.action_stats_json),
    )
    if args.save_summary_json:
        out = Path(args.save_summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
