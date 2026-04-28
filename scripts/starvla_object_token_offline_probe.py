from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.policy_trace_contact_datasets import PolicyTraceContactDataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

from starvla_dhb_predict_smoke import (
    _default_output_dir,
    _load_rollout_checkpoint,
    _uses_quantized_vlm,
)


def _as_field_list(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    return tuple(str(part) for part in value)


def _align_temporal_target(
    target: np.ndarray,
    *,
    window_size: int,
    alignment: str,
    current_prefix_steps: int,
) -> np.ndarray:
    window = target[-int(window_size) :]
    alignment = str(alignment or "future").strip().lower()
    if alignment == "future" or window.shape[0] == 0:
        return window
    if alignment == "current":
        return np.repeat(window[:1], repeats=window.shape[0], axis=0)
    if alignment == "current_prefix":
        prefix = max(1, int(current_prefix_steps or 0))
        aligned = window.copy()
        aligned[: min(prefix, len(aligned))] = aligned[:1]
        return aligned
    raise ValueError(
        "object_target_target_alignment must be one of 'future', 'current', "
        f"or 'current_prefix', got {alignment!r}"
    )


def _stack_target_fields(sample: dict[str, Any], fields: tuple[str, ...]) -> np.ndarray | None:
    parts: list[np.ndarray] = []
    for field in fields:
        if field not in sample:
            return None
        arr = np.asarray(sample[field], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            return None
        parts.append(arr)
    if not parts:
        return None
    n = min(part.shape[0] for part in parts)
    return np.concatenate([part[:n] for part in parts], axis=-1)


def _masked_rows(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = min(values.shape[0], mask.shape[0])
    return values[:n][mask[:n]]


def _checkpoint_label(path: str) -> str:
    stem = Path(path).stem
    if stem.endswith("_pytorch_model"):
        stem = stem[: -len("_pytorch_model")]
    return stem


def _evaluate_checkpoint(
    *,
    cfg: Any,
    checkpoint: str,
    limit: int,
    transport_only: bool,
    include_rows: int,
) -> dict[str, Any]:
    model = build_framework(cfg)
    _load_rollout_checkpoint(model, checkpoint)
    model.eval()
    if torch.cuda.is_available() and not _uses_quantized_vlm(model):
        model = model.to(torch.device("cuda"))

    dataset = PolicyTraceContactDataset(cfg.datasets.vla_data)
    action_cfg = getattr(cfg.framework, "action_model", {})
    fields = _as_field_list(
        getattr(action_cfg, "object_target_fields", None),
        ("dhb.object_minus_target_pos",),
    )
    window_size = int(getattr(action_cfg, "future_action_window_size", 7) or 7) + 1
    alignment = str(getattr(action_cfg, "object_target_target_alignment", "future"))
    current_prefix_steps = int(getattr(action_cfg, "object_target_current_prefix_steps", 0) or 0)

    per_dim_abs: list[np.ndarray] = []
    object_minus_target_abs: list[np.ndarray] = []
    target_norms: list[float] = []
    pred_norms: list[float] = []
    rows: list[dict[str, Any]] = []
    phase_counts: dict[str, int] = {}
    evaluated_rows = 0

    for index in range(len(dataset)):
        sample = dataset[index]
        phase = np.asarray(
            sample.get("dhb.transport_phase_onehot", sample.get("dhb.contact_phase", [])),
            dtype=np.float32,
        )
        if transport_only and phase.ndim == 2 and not np.any(phase[:, 1:] > 0.5):
            continue

        target = _stack_target_fields(sample, fields)
        if target is None:
            continue
        target = _align_temporal_target(
            target,
            window_size=window_size,
            alignment=alignment,
            current_prefix_steps=current_prefix_steps,
        )

        mask = np.asarray(
            sample.get("dhb.action_valid_mask", np.ones(target.shape[0])),
            dtype=np.float32,
        ).reshape(-1)
        mask = mask[-window_size:] > 0.5

        with torch.inference_mode():
            traj = model.predict_trajectory(sample)
        if "pred_object_target" not in traj:
            raise RuntimeError("model output does not contain pred_object_target")
        pred = np.asarray(traj["pred_object_target"], dtype=np.float32)[0]

        n = min(pred.shape[0], target.shape[0], mask.shape[0])
        pred = pred[:n, : target.shape[-1]]
        target = target[:n]
        active = mask[:n]
        if not np.any(active):
            continue

        evaluated_rows += 1
        pred_active = pred[active]
        target_active = target[active]
        abs_err = np.abs(pred_active - target_active)
        per_dim_abs.append(abs_err)
        target_norms.extend(np.linalg.norm(target_active, axis=-1).astype(float).tolist())
        pred_norms.extend(np.linalg.norm(pred_active, axis=-1).astype(float).tolist())

        if target.shape[-1] >= 6:
            pred_delta = pred_active[:, :3] - pred_active[:, 3:6]
            target_delta = target_active[:, :3] - target_active[:, 3:6]
            object_minus_target_abs.append(np.abs(pred_delta - target_delta))

        if phase.ndim == 2:
            aligned_phase = phase[-window_size:][:n]
            if aligned_phase.shape[0] == n:
                for phase_id in aligned_phase[active].argmax(axis=-1).tolist():
                    key = str(int(phase_id))
                    phase_counts[key] = phase_counts.get(key, 0) + 1

        if include_rows <= 0 or len(rows) < include_rows:
            rows.append(
                {
                    "index": int(index),
                    "npz_path": sample.get("policy_trace.npz_path"),
                    "labels": list(sample.get("policy_trace.labels", [])),
                    "target_mean": target_active.mean(axis=0).astype(float).tolist(),
                    "pred_mean": pred_active.mean(axis=0).astype(float).tolist(),
                    "mae": abs_err.mean(axis=0).astype(float).tolist(),
                }
            )

        if limit > 0 and evaluated_rows >= limit:
            break

    abs_all = np.concatenate(per_dim_abs, axis=0) if per_dim_abs else np.zeros((0, 0), dtype=np.float32)
    delta_all = (
        np.concatenate(object_minus_target_abs, axis=0)
        if object_minus_target_abs
        else np.zeros((0, 3), dtype=np.float32)
    )
    return {
        "checkpoint": checkpoint,
        "label": _checkpoint_label(checkpoint),
        "fields": list(fields),
        "alignment": alignment,
        "rows_evaluated": int(evaluated_rows),
        "steps_evaluated": int(abs_all.shape[0]),
        "phase_counts": dict(sorted(phase_counts.items())),
        "object_token_mae": abs_all.mean(axis=0).astype(float).tolist() if abs_all.size else [],
        "object_token_p95_abs": (
            np.percentile(abs_all, 95, axis=0).astype(float).tolist() if abs_all.size else []
        ),
        "object_minus_target_mae_m": (
            delta_all.mean(axis=0).astype(float).tolist() if delta_all.size else []
        ),
        "object_minus_target_mae_xy_m": (
            float(np.linalg.norm(delta_all[:, :2], axis=-1).mean()) if delta_all.size else None
        ),
        "target_norm_mean": float(np.mean(target_norms)) if target_norms else None,
        "pred_norm_mean": float(np.mean(pred_norms)) if pred_norms else None,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--include-rows", type=int, default=20)
    parser.add_argument("--transport-only", action="store_true")
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)
    cfg.output_dir = str(_default_output_dir(cfg))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StarVLA object-token offline probing.")

    results = [
        _evaluate_checkpoint(
            cfg=cfg,
            checkpoint=checkpoint,
            limit=args.limit,
            transport_only=args.transport_only,
            include_rows=args.include_rows,
        )
        for checkpoint in args.checkpoint
    ]
    payload = {
        "config_yaml": args.config_yaml,
        "overrides": dotlist,
        "limit": int(args.limit),
        "transport_only": bool(args.transport_only),
        "results": results,
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "results": [{k: v for k, v in row.items() if k != "rows"} for row in results]}, indent=2))


if __name__ == "__main__":
    main()
