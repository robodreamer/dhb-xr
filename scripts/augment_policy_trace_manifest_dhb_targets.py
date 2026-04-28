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
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def _transport_phase_from_action(raw_action: np.ndarray, *, z_threshold_mm: float) -> str:
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    z_mm = float(action[2] * 1000.0) if action.shape[0] >= 3 else 0.0
    threshold = max(0.0, float(z_threshold_mm))
    if z_mm > threshold:
        return "lift"
    if z_mm < -threshold:
        return "lower"
    return "carry"


def _transport_phase_onehot(phase: str, *, scale: float = 1.0) -> list[float]:
    classes = ("lift", "carry", "lower")
    out = [0.0, 0.0, 0.0]
    try:
        out[classes.index(str(phase))] = float(scale)
    except ValueError:
        out[classes.index("carry")] = float(scale)
    return out


def _parse_channel_weight(value: str, *, action_dim: int) -> list[float]:
    weights = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(weights) != action_dim:
        raise ValueError(
            f"Expected {action_dim} comma-separated action-channel weights, got {len(weights)}"
        )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Action-channel weights must be non-negative")
    if sum(weights) <= 0.0:
        raise ValueError("At least one action-channel weight must be positive")
    return weights


def _phase_action_channel_weight(
    phase: str,
    *,
    lift_weight: list[float],
    carry_weight: list[float],
    lower_weight: list[float],
) -> list[float]:
    if str(phase) == "lift":
        return list(lift_weight)
    if str(phase) == "lower":
        return list(lower_weight)
    return list(carry_weight)


def _contact_release_phase_from_row(row: dict[str, Any]) -> list[int]:
    labels = list(row.get("labels", []))
    target_close = list(row.get("target_close", []))
    target_gripper_state = list(row.get("target_gripper_state", []))
    n = max(len(labels), len(target_close), len(target_gripper_state))
    phases: list[int] = []
    for idx in range(n):
        label = str(labels[idx]) if idx < len(labels) else ""
        if "release" in label:
            phases.append(2)
            continue
        if idx < len(target_close):
            close = np.asarray(target_close[idx], dtype=np.float32).reshape(-1)
            if close.size and float(close[0]) <= 0.5:
                phases.append(2)
                continue
        if idx < len(target_gripper_state):
            state = np.asarray(target_gripper_state[idx], dtype=np.float32).reshape(-1)
            if state.size and float(state[0]) <= 0.5:
                phases.append(2)
                continue
        phases.append(0)
    return phases


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


def augment_row(
    row: dict[str, Any],
    *,
    action_stats: dict[str, Any],
    phase_z_threshold_mm: float = 1.0,
    phase_onehot_scale: float = 1.0,
    emit_phase_action_channel_weights: bool = False,
    lift_action_channel_weight: list[float] | None = None,
    carry_action_channel_weight: list[float] | None = None,
    lower_action_channel_weight: list[float] | None = None,
) -> dict[str, Any]:
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
    phases = row.get("target_transport_phase")
    if phases is None:
        phases = [
            _transport_phase_from_action(raw_action, z_threshold_mm=phase_z_threshold_mm)
            for raw_action in action
        ]
        out["target_transport_phase"] = phases
    out["target_transport_phase_onehot"] = [
        _transport_phase_onehot(phase, scale=phase_onehot_scale)
        for phase in phases
    ]
    if emit_phase_action_channel_weights:
        if (
            lift_action_channel_weight is None
            or carry_action_channel_weight is None
            or lower_action_channel_weight is None
        ):
            raise ValueError("Phase action-channel weights must be provided when emission is enabled")
        out["action_loss_channel_weight"] = [
            _phase_action_channel_weight(
                phase,
                lift_weight=lift_action_channel_weight,
                carry_weight=carry_action_channel_weight,
                lower_weight=lower_action_channel_weight,
            )
            for phase in phases
        ]
    if row.get("target_contact_release_phase") is None:
        release_phase = _contact_release_phase_from_row(row)
        if release_phase:
            out["target_contact_release_phase"] = release_phase
    return out


def augment_manifest(
    manifest_jsonl: Path,
    output_jsonl: Path,
    *,
    action_stats_json: Path,
    phase_z_threshold_mm: float = 1.0,
    phase_onehot_scale: float = 1.0,
    emit_phase_action_channel_weights: bool = False,
    lift_action_channel_weight: list[float] | None = None,
    carry_action_channel_weight: list[float] | None = None,
    lower_action_channel_weight: list[float] | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(manifest_jsonl)
    action_stats = _extract_action_stats(json.loads(action_stats_json.read_text(encoding="utf-8")))
    augmented = [
        augment_row(
            row,
            action_stats=action_stats,
            phase_z_threshold_mm=phase_z_threshold_mm,
            phase_onehot_scale=phase_onehot_scale,
            emit_phase_action_channel_weights=emit_phase_action_channel_weights,
            lift_action_channel_weight=lift_action_channel_weight,
            carry_action_channel_weight=carry_action_channel_weight,
            lower_action_channel_weight=lower_action_channel_weight,
        )
        for row in rows
    ]
    _write_jsonl(output_jsonl, augmented)
    invariant_steps = sum(len(row.get("target_linear_invariants", [])) for row in augmented)
    phase_counts: dict[str, int] = {}
    for row in augmented:
        for phase in row.get("target_transport_phase", []):
            phase_counts[str(phase)] = phase_counts.get(str(phase), 0) + 1
    channel_weighted_steps = sum(
        len(row.get("action_loss_channel_weight", []))
        for row in augmented
    )
    return {
        "input_jsonl": str(manifest_jsonl),
        "output_jsonl": str(output_jsonl),
        "rows": int(len(augmented)),
        "invariant_steps": int(invariant_steps),
        "phase_counts": dict(sorted(phase_counts.items())),
        "channel_weighted_steps": int(channel_weighted_steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--action-stats-json", required=True)
    parser.add_argument("--phase-z-threshold-mm", type=float, default=1.0)
    parser.add_argument("--phase-onehot-scale", type=float, default=1.0)
    parser.add_argument(
        "--emit-phase-action-channel-weights",
        action="store_true",
        help=(
            "Emit per-step per-action-channel loss weights from transport phase. "
            "This is training-only supervision and does not affect rollout logic."
        ),
    )
    parser.add_argument(
        "--lift-action-channel-weight",
        default="2.0,2.0,8.0,0.1,0.1,0.1,0.0",
    )
    parser.add_argument(
        "--carry-action-channel-weight",
        default="6.0,7.0,1.5,0.1,0.1,0.1,0.0",
    )
    parser.add_argument(
        "--lower-action-channel-weight",
        default="2.0,2.0,8.0,0.1,0.1,0.1,0.0",
    )
    parser.add_argument("--save-summary-json")
    args = parser.parse_args()
    stats_payload = json.loads(Path(args.action_stats_json).read_text(encoding="utf-8"))
    action_dim = len(_extract_action_stats(stats_payload)["min"])
    lift_action_channel_weight = None
    carry_action_channel_weight = None
    lower_action_channel_weight = None
    if args.emit_phase_action_channel_weights:
        lift_action_channel_weight = _parse_channel_weight(
            args.lift_action_channel_weight,
            action_dim=action_dim,
        )
        carry_action_channel_weight = _parse_channel_weight(
            args.carry_action_channel_weight,
            action_dim=action_dim,
        )
        lower_action_channel_weight = _parse_channel_weight(
            args.lower_action_channel_weight,
            action_dim=action_dim,
        )

    summary = augment_manifest(
        Path(args.manifest_jsonl),
        Path(args.output_jsonl),
        action_stats_json=Path(args.action_stats_json),
        phase_z_threshold_mm=float(args.phase_z_threshold_mm),
        phase_onehot_scale=float(args.phase_onehot_scale),
        emit_phase_action_channel_weights=bool(args.emit_phase_action_channel_weights),
        lift_action_channel_weight=lift_action_channel_weight,
        carry_action_channel_weight=carry_action_channel_weight,
        lower_action_channel_weight=lower_action_channel_weight,
    )
    if args.save_summary_json:
        out = Path(args.save_summary_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
