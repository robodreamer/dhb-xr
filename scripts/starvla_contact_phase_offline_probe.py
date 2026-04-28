from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def _align_phase_target(target: np.ndarray, *, alignment: str, current_prefix_steps: int) -> np.ndarray:
    alignment = str(alignment or "future").strip().lower()
    if alignment == "future" or target.size == 0:
        return target
    if alignment == "current":
        return np.full_like(target, target[0])
    if alignment == "current_prefix":
        out = target.copy()
        prefix = max(1, int(current_prefix_steps or 0))
        out[: min(prefix, out.size)] = out[0]
        return out
    raise ValueError(
        "contact_phase_target_alignment must be one of 'future', 'current', "
        f"or 'current_prefix', got {alignment!r}"
    )


def _phase_metrics(
    sample: dict,
    traj: dict,
    index: int,
    *,
    target_alignment: str,
    current_prefix_steps: int,
) -> dict:
    target = np.asarray(sample.get("dhb.contact_phase", []), dtype=np.int64).reshape(-1)
    mask = np.asarray(sample.get("dhb.action_valid_mask", []), dtype=np.float32).reshape(-1)
    probs = np.asarray(traj["pred_contact_phase_probs"][0], dtype=np.float32)

    n = min(len(target), len(mask), probs.shape[0])
    target = _align_phase_target(
        target[:n],
        alignment=target_alignment,
        current_prefix_steps=current_prefix_steps,
    )
    mask = mask[:n] > 0.5
    probs = probs[:n]
    pred = probs.argmax(axis=-1)
    release = mask & (target == 2)
    supervised = mask

    open_probs = probs[:, 2] if probs.shape[-1] >= 3 else probs[:, -1]
    close_probs = probs[:, 1] if probs.shape[-1] >= 3 else probs[:, 0]

    return {
        "index": int(index),
        "npz_path": sample.get("policy_trace.npz_path"),
        "labels": list(sample.get("policy_trace.labels", [])),
        "target_phase": target.tolist(),
        "pred_phase": pred.tolist(),
        "valid_mask": mask.astype(int).tolist(),
        "release_steps": int(release.sum()),
        "supervised_steps": int(supervised.sum()),
        "max_open_prob": float(open_probs[supervised].max()) if supervised.any() else 0.0,
        "max_close_prob": float(close_probs[supervised].max()) if supervised.any() else 0.0,
        "mean_open_prob": float(open_probs[supervised].mean()) if supervised.any() else 0.0,
        "target_release_open_prob_mean": float(open_probs[release].mean()) if release.any() else None,
        "target_release_open_prob_max": float(open_probs[release].max()) if release.any() else None,
        "target_release_pred_accuracy": float((pred[release] == 2).mean()) if release.any() else None,
        "phase_probs": probs.tolist(),
    }


def _aggregate(rows: list[dict]) -> dict:
    supervised = sum(row["supervised_steps"] for row in rows)
    release = sum(row["release_steps"] for row in rows)
    release_rows = [row for row in rows if row["release_steps"] > 0]
    noevent_open_probs: list[float] = []
    noevent_open_argmax = 0
    noevent_steps = 0
    release_open_probs: list[float] = []
    release_open_argmax = 0
    release_means = [
        row["target_release_open_prob_mean"]
        for row in release_rows
        if row["target_release_open_prob_mean"] is not None
    ]
    release_maxes = [
        row["target_release_open_prob_max"]
        for row in release_rows
        if row["target_release_open_prob_max"] is not None
    ]
    release_accs = [
        row["target_release_pred_accuracy"]
        for row in release_rows
        if row["target_release_pred_accuracy"] is not None
    ]
    for row in rows:
        target = np.asarray(row["target_phase"], dtype=np.int64)
        pred = np.asarray(row["pred_phase"], dtype=np.int64)
        valid = np.asarray(row["valid_mask"], dtype=bool)
        probs = np.asarray(row["phase_probs"], dtype=np.float32)
        open_probs = probs[:, 2] if probs.shape[-1] >= 3 else probs[:, -1]
        release_mask = valid & (target == 2)
        noevent_mask = valid & (target != 2)
        if release_mask.any():
            release_open_probs.extend(open_probs[release_mask].astype(float).tolist())
            release_open_argmax += int(np.sum(pred[release_mask] == 2))
        if noevent_mask.any():
            noevent_open_probs.extend(open_probs[noevent_mask].astype(float).tolist())
            noevent_open_argmax += int(np.sum(pred[noevent_mask] == 2))
            noevent_steps += int(noevent_mask.sum())

    class_count: dict[str, int] = {}
    pred_count: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    total = 0
    for row in rows:
        target = np.asarray(row["target_phase"], dtype=np.int64)
        pred = np.asarray(row["pred_phase"], dtype=np.int64)
        valid = np.asarray(row["valid_mask"], dtype=bool)
        n = min(len(target), len(pred), len(valid))
        for target_phase, pred_phase, is_valid in zip(target[:n], pred[:n], valid[:n]):
            if not bool(is_valid):
                continue
            target_key = str(int(target_phase))
            pred_key = str(int(pred_phase))
            class_count[target_key] = class_count.get(target_key, 0) + 1
            pred_count[pred_key] = pred_count.get(pred_key, 0) + 1
            confusion.setdefault(target_key, {})
            confusion[target_key][pred_key] = confusion[target_key].get(pred_key, 0) + 1
            correct += int(int(target_phase) == int(pred_phase))
            total += 1

    class_recall: dict[str, float] = {}
    for target_key, pred_by_class in confusion.items():
        denom = max(1, sum(pred_by_class.values()))
        class_recall[target_key] = float(pred_by_class.get(target_key, 0) / denom)

    release_open_arr = np.asarray(release_open_probs, dtype=np.float32)
    noevent_open_arr = np.asarray(noevent_open_probs, dtype=np.float32)
    return {
        "rows_evaluated": int(len(rows)),
        "release_rows_evaluated": int(len(release_rows)),
        "supervised_steps": int(supervised),
        "release_steps": int(release),
        "noevent_steps": int(noevent_steps),
        "phase_accuracy": float(correct / max(1, total)) if total else None,
        "phase_target_counts": dict(sorted(class_count.items())),
        "phase_pred_counts": dict(sorted(pred_count.items())),
        "phase_confusion": {
            key: dict(sorted(value.items())) for key, value in sorted(confusion.items())
        },
        "phase_recall": dict(sorted(class_recall.items())),
        "max_open_prob": float(max((row["max_open_prob"] for row in rows), default=0.0)),
        "max_close_prob": float(max((row["max_close_prob"] for row in rows), default=0.0)),
        "mean_open_prob": float(np.mean([row["mean_open_prob"] for row in rows])) if rows else 0.0,
        "target_release_open_prob_mean": float(np.mean(release_means)) if release_means else None,
        "target_release_open_prob_max": float(max(release_maxes)) if release_maxes else None,
        "target_release_pred_accuracy": float(np.mean(release_accs)) if release_accs else None,
        "release_open_ge_0.5_rate": (
            float(np.mean(release_open_arr >= 0.5)) if release_open_arr.size else None
        ),
        "release_open_argmax_rate": (
            float(release_open_argmax / max(1, len(release_open_probs)))
            if release_open_probs
            else None
        ),
        "noevent_open_prob_mean": (
            float(noevent_open_arr.mean()) if noevent_open_arr.size else None
        ),
        "noevent_open_prob_p95": (
            float(np.percentile(noevent_open_arr, 95)) if noevent_open_arr.size else None
        ),
        "noevent_open_prob_max": (
            float(noevent_open_arr.max()) if noevent_open_arr.size else None
        ),
        "noevent_open_ge_0.5_rate": (
            float(np.mean(noevent_open_arr >= 0.5)) if noevent_open_arr.size else None
        ),
        "noevent_open_ge_0.7_rate": (
            float(np.mean(noevent_open_arr >= 0.7)) if noevent_open_arr.size else None
        ),
        "noevent_open_argmax_rate": (
            float(noevent_open_argmax / max(1, noevent_steps)) if noevent_steps else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--release-only", action="store_true")
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)
    cfg.output_dir = str(_default_output_dir(cfg))

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StarVLA contact-phase offline probing.")

    model = build_framework(cfg)
    _load_rollout_checkpoint(model, args.checkpoint)
    model.eval()
    if not _uses_quantized_vlm(model):
        model = model.to(torch.device("cuda"))

    dataset = PolicyTraceContactDataset(cfg.datasets.vla_data)
    action_cfg = getattr(cfg.framework, "action_model", {})
    target_alignment = str(getattr(action_cfg, "contact_phase_target_alignment", "future"))
    current_prefix_steps = int(getattr(action_cfg, "contact_phase_current_prefix_steps", 0) or 0)
    rows: list[dict] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        phase = np.asarray(sample.get("dhb.contact_phase", []), dtype=np.int64).reshape(-1)
        mask = np.asarray(sample.get("dhb.action_valid_mask", []), dtype=np.float32).reshape(-1) > 0.5
        n = min(len(phase), len(mask))
        has_release = bool(n and np.any((phase[:n] == 2) & mask[:n]))
        if args.release_only and not has_release:
            continue
        with torch.inference_mode():
            traj = model.predict_trajectory(sample)
        rows.append(
            _phase_metrics(
                sample,
                traj,
                index,
                target_alignment=target_alignment,
                current_prefix_steps=current_prefix_steps,
            )
        )
        if args.limit > 0 and len(rows) >= args.limit:
            break

    payload = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "overrides": dotlist,
        "release_only": bool(args.release_only),
        "target_alignment": target_alignment,
        "current_prefix_steps": current_prefix_steps,
        "summary": _aggregate(rows),
        "rows": rows,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
