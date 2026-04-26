from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PATH = REPO_ROOT / "vendor" / "starVLA"
SRC_PATH = REPO_ROOT / "src"
for path in (VENDOR_PATH, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _default_output_dir(cfg) -> Path:
    run_root_dir = Path(getattr(cfg, "run_root_dir", "_workspace/starvla_runs"))
    run_id = getattr(cfg, "run_id", "dhb_discrete_offline_sweep")
    return run_root_dir / run_id


def _uses_quantized_vlm(model) -> bool:
    qwen = getattr(model, "qwen_vl_interface", None)
    if qwen is None:
        return False
    backend = getattr(qwen, "model", None)
    return bool(
        getattr(backend, "is_loaded_in_8bit", False) or getattr(backend, "is_loaded_in_4bit", False)
    )


def _load_rollout_checkpoint(model, checkpoint_path: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if _uses_quantized_vlm(model):
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("qwen_vl_interface.")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [
            k
            for k in missing
            if not k.startswith("qwen_vl_interface.")
            and "committed_prefix_" not in k
            and "oat_token_context_proj" not in k
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Quantized discrete sweep checkpoint load mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return

    model.load_state_dict(state_dict, strict=True)


def _parse_token_ks(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("Expected at least one token prefix length")
    return values


def _iter_holdout_samples(dataset, *, holdout_mod: int, holdout_fold: int, max_samples: int) -> Iterable[tuple[dict, dict]]:
    subsets = getattr(dataset, "datasets", [dataset])
    emitted = 0
    global_idx = 0
    for subset_index, subset in enumerate(subsets):
        all_steps = getattr(subset, "all_steps", None)
        if not all_steps:
            continue
        for row_index, (traj_id, base_index) in enumerate(all_steps):
            if global_idx % holdout_mod != holdout_fold:
                global_idx += 1
                continue

            raw = subset.get_step_data(traj_id, base_index)
            sample = subset._pack_sample(subset.transforms(raw))
            meta = {
                "subset_index": int(subset_index),
                "subset_name": getattr(subset, "repo_id", None) or getattr(subset, "repo", None),
                "traj_id": str(traj_id),
                "base_index": int(base_index),
                "global_index": int(global_idx),
            }
            yield meta, sample
            emitted += 1
            global_idx += 1
            if emitted >= max_samples:
                return
        else:
            continue


def _stack_gt_invariants(example: dict) -> np.ndarray:
    linear = np.asarray(example["dhb.linear_invariants"], dtype=np.float32)
    angular = np.asarray(example["dhb.angular_invariants"], dtype=np.float32)
    return np.concatenate([linear, angular], axis=-1)


def _derive_gt_oat_tokens(model, gt_inv: np.ndarray) -> np.ndarray | None:
    action_model = getattr(model, "action_model", None)
    backend = getattr(action_model, "backend", None)
    oat = getattr(backend, "oat", None)
    if oat is None:
        return None
    try:
        param = next(backend.parameters())
        device = param.device
        dtype = param.dtype if param.dtype.is_floating_point else torch.float32
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

    was_training = oat.training
    oat.eval()
    try:
        with torch.no_grad():
            inv = torch.as_tensor(gt_inv, device=device, dtype=dtype).unsqueeze(0)
            if hasattr(backend, "_tokenize_oat_invariants"):
                tokens = backend._tokenize_oat_invariants(inv)
            else:
                tokens = oat.tokenize(inv)
            return tokens[0].detach().cpu().numpy().astype(np.int64)
    finally:
        if was_training:
            oat.train()


def _decode_oat_tokens(model, tokens: np.ndarray) -> np.ndarray | None:
    action_model = getattr(model, "action_model", None)
    backend = getattr(action_model, "backend", None)
    oat = getattr(backend, "oat", None)
    if oat is None or backend is None:
        return None
    try:
        param = next(backend.parameters())
        device = param.device
    except StopIteration:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        token_tensor = torch.as_tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)
        decoded, _ = backend._decode_oat_tokens(token_tensor)
        return decoded[0].detach().to(dtype=torch.float32).cpu().numpy().astype(np.float32)


def _initial_pose_from_sample(sample: dict) -> dict[str, np.ndarray] | None:
    state = sample.get("state", sample.get("observation.state"))
    if state is None:
        return None
    state_arr = np.asarray(state, dtype=np.float32)
    if state_arr.ndim == 2:
        state_arr = state_arr[-1]
    elif state_arr.ndim > 2:
        state_arr = state_arr.reshape(-1, state_arr.shape[-1])[-1]
    if state_arr.shape[-1] < 6:
        return None
    quat_xyzw = Rotation.from_rotvec(state_arr[3:6]).as_quat().astype(np.float32)
    return {
        "position": state_arr[:3].astype(np.float32),
        "quaternion": quat_xyzw[[3, 0, 1, 2]].astype(np.float32),
    }


def _decode_positions_from_invariants(invariants: np.ndarray, initial_pose: dict[str, np.ndarray]) -> np.ndarray:
    from dhb_xr.core.types import EncodingMethod
    from dhb_xr.decoder.dhb_dr import decode_dhb_dr

    decoded = decode_dhb_dr(
        invariants[:, :4],
        invariants[:, 4:],
        initial_pose=initial_pose,
        method=EncodingMethod.POSITION,
    )
    return np.asarray(decoded["positions"], dtype=np.float32)


def _mean_abs(arr: np.ndarray) -> float:
    return float(np.abs(arr).mean())


def _endpoint_l2(positions_a: np.ndarray, positions_b: np.ndarray) -> float:
    return float(np.linalg.norm(positions_a[-1] - positions_b[-1]))


def _trajectory_path_length(positions: np.ndarray) -> float:
    if positions.shape[0] <= 1:
        return 0.0
    deltas = np.diff(positions, axis=0)
    return float(np.linalg.norm(deltas, axis=1).sum())


def _monotonic_nonincreasing(values: list[float], *, tol: float = 1e-6) -> bool:
    return all(values[i] <= values[i - 1] + tol for i in range(1, len(values)))


def _round_or_none(value: float | None, digits: int = 6) -> float | None:
    return None if value is None or math.isnan(value) else round(float(value), digits)


def _token_histogram(values: list[int]) -> dict[str, int]:
    if not values:
        return {}
    tokens, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    return {
        str(token): int(count)
        for token, count in zip(tokens, counts)
    }


def _token_position_unique_counts(position_values: list[list[int]]) -> list[int]:
    return [
        int(np.unique(np.asarray(values, dtype=np.int64)).size)
        for values in position_values
    ]


def _unique_sequence_count(sequences: list[list[int]]) -> int:
    if not sequences:
        return 0
    return len({tuple(int(v) for v in sequence) for sequence in sequences})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--holdout_mod", type=int, default=10)
    parser.add_argument("--holdout_fold", type=int, default=0)
    parser.add_argument("--token_prefix_ks", default="1,2,4,8")
    parser.add_argument("--include_full", action="store_true")
    parser.add_argument(
        "--tokenize_continuous_head",
        action="store_true",
        help=(
            "Evaluate codec mode by deriving OAT tokens from the continuous invariant "
            "head instead of using a learned token-head policy."
        ),
    )
    parser.add_argument(
        "--evaluate_gt_codec",
        action="store_true",
        help="Also measure teacher-forced OAT reconstruction of ground-truth DHB invariant chunks.",
    )
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)

    cfg.output_dir = str(_default_output_dir(cfg))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DHB discrete offline sweep.")

    token_ks = sorted(set(_parse_token_ks(args.token_prefix_ks)))
    device = torch.device("cuda")

    model = build_framework(cfg)
    _load_rollout_checkpoint(model, args.checkpoint)
    model.eval()
    if not _uses_quantized_vlm(model):
        model = model.to(device)

    dataset = get_vla_dataset(cfg.datasets.vla_data, mode="val")

    per_case_rows: list[dict] = []
    metric_sums = defaultdict(float)
    metric_counts = defaultdict(int)
    prefix_metrics = {
        str(k): defaultdict(float) for k in token_ks
    }
    prefix_counts = defaultdict(int)
    full_metrics = defaultdict(float)
    full_count = 0
    gt_codec_metrics = defaultdict(float)
    gt_codec_count = 0
    all_pred_tokens: list[int] = []
    token_position_values: list[list[int]] = []
    pred_token_sequences: list[list[int]] = []
    all_gt_tokens: list[int] = []
    gt_token_position_values: list[list[int]] = []
    gt_token_sequences: list[list[int]] = []

    samples = list(
        _iter_holdout_samples(
            dataset,
            holdout_mod=max(1, int(args.holdout_mod)),
            holdout_fold=max(0, int(args.holdout_fold)),
            max_samples=max(1, int(args.max_samples)),
        )
    )
    if not samples:
        raise RuntimeError("No holdout samples selected; adjust holdout_mod/holdout_fold/max_samples.")

    for meta, sample in samples:
        gt_inv = _stack_gt_invariants(sample)
        continuous = model.predict_trajectory(
            [sample],
            emit_oat_tokens=True,
            tokenize_continuous_head=bool(args.tokenize_continuous_head),
        )
        cont_inv = np.asarray(
            continuous.get("pred_invariants_execute", continuous["pred_invariants"]),
            dtype=np.float32,
        )[0]
        cont_pos = np.asarray(continuous["positions"], dtype=np.float32)[0]
        cont_gt_inv_mae = _mean_abs(cont_inv - gt_inv)
        cont_path_length_m = _trajectory_path_length(cont_pos)
        tokens = np.asarray(continuous.get("pred_oat_tokens", np.empty((1, 0))), dtype=np.int64)
        tokens_first = tokens[0] if tokens.size else np.empty((0,), dtype=np.int64)
        unique_tokens = int(np.unique(tokens_first).size) if tokens_first.size else 0
        gt_tokens = _derive_gt_oat_tokens(model, gt_inv)
        token_accuracy = None
        if gt_tokens is not None and tokens_first.shape == gt_tokens.shape and tokens_first.size:
            token_accuracy = float(np.mean(tokens_first == gt_tokens))
            metric_sums["token_accuracy"] += token_accuracy
            metric_counts["token_accuracy"] += 1
            gt_unique_tokens = int(np.unique(gt_tokens).size)
            metric_sums["gt_token_unique_count"] += float(gt_unique_tokens)
            metric_counts["gt_tokens"] += 1
            if gt_unique_tokens <= 1:
                metric_sums["gt_collapsed_samples"] += 1.0
            gt_token_sequences.append([int(v) for v in gt_tokens.tolist()])
            all_gt_tokens.extend(int(v) for v in gt_tokens.tolist())
            if not gt_token_position_values:
                gt_token_position_values = [[] for _ in range(gt_tokens.size)]
            for idx, value in enumerate(gt_tokens.tolist()):
                gt_token_position_values[idx].append(int(value))

            if args.evaluate_gt_codec:
                gt_codec_inv = _decode_oat_tokens(model, gt_tokens)
                initial_pose = _initial_pose_from_sample(sample)
                if gt_codec_inv is not None and initial_pose is not None:
                    gt_pos = _decode_positions_from_invariants(gt_inv, initial_pose)
                    gt_codec_pos = _decode_positions_from_invariants(gt_codec_inv, initial_pose)
                    gt_codec_row = {
                        "gt_codec_inv_mae": _mean_abs(gt_codec_inv - gt_inv),
                        "gt_codec_pos_mae_m": _mean_abs(gt_codec_pos - gt_pos),
                        "gt_codec_endpoint_drift_m": _endpoint_l2(gt_codec_pos, gt_pos),
                        "gt_codec_path_length_m": _trajectory_path_length(gt_codec_pos),
                    }
                    gt_codec_count += 1
                    for key, value in gt_codec_row.items():
                        gt_codec_metrics[key] += value
                else:
                    gt_codec_row = None
            else:
                gt_codec_row = None
        else:
            gt_codec_row = None
        if tokens_first.size:
            pred_token_sequences.append([int(v) for v in tokens_first.tolist()])
            all_pred_tokens.extend(int(v) for v in tokens_first.tolist())
            if not token_position_values:
                token_position_values = [[] for _ in range(tokens_first.size)]
            for idx, value in enumerate(tokens_first.tolist()):
                token_position_values[idx].append(int(value))

        base_row = {
            **meta,
            "token_ids": tokens_first.tolist(),
            "gt_token_ids": None if gt_tokens is None else gt_tokens.tolist(),
            "token_accuracy": None if token_accuracy is None else round(token_accuracy, 6),
            "token_unique_count": unique_tokens,
            "continuous_gt_inv_mae": cont_gt_inv_mae,
            "continuous_path_length_m": cont_path_length_m,
            "gt_codec_result": gt_codec_row,
        }
        per_k_rows = []

        for k in token_ks:
            discrete = model.predict_trajectory(
                [sample],
                use_discrete_tokens=True,
                emit_oat_tokens=True,
                token_prefix_keep_k=k,
                tokenize_continuous_head=bool(args.tokenize_continuous_head),
            )
            disc_inv = np.asarray(discrete["pred_invariants_execute"], dtype=np.float32)[0]
            disc_pos = np.asarray(discrete["positions"], dtype=np.float32)[0]
            row = {
                "k": int(k),
                "gt_inv_mae": _mean_abs(disc_inv - gt_inv),
                "cont_disc_inv_mae": _mean_abs(disc_inv - cont_inv),
                "cont_disc_pos_mae_m": _mean_abs(disc_pos - cont_pos),
                "endpoint_drift_m": _endpoint_l2(disc_pos, cont_pos),
                "path_length_m": _trajectory_path_length(disc_pos),
            }
            per_k_rows.append(row)
            prefix_counts[str(k)] += 1
            for key in ("gt_inv_mae", "cont_disc_inv_mae", "cont_disc_pos_mae_m", "endpoint_drift_m", "path_length_m"):
                prefix_metrics[str(k)][key] += row[key]

        if args.include_full:
            discrete_full = model.predict_trajectory(
                [sample],
                use_discrete_tokens=True,
                emit_oat_tokens=True,
                tokenize_continuous_head=bool(args.tokenize_continuous_head),
            )
            full_inv = np.asarray(discrete_full["pred_invariants_execute"], dtype=np.float32)[0]
            full_pos = np.asarray(discrete_full["positions"], dtype=np.float32)[0]
            full_row = {
                "gt_inv_mae": _mean_abs(full_inv - gt_inv),
                "cont_disc_inv_mae": _mean_abs(full_inv - cont_inv),
                "cont_disc_pos_mae_m": _mean_abs(full_pos - cont_pos),
                "endpoint_drift_m": _endpoint_l2(full_pos, cont_pos),
                "path_length_m": _trajectory_path_length(full_pos),
            }
            full_count += 1
            for key, value in full_row.items():
                full_metrics[key] += value
        else:
            full_row = None

        per_case_rows.append(
            {
                **base_row,
                "prefix_results": per_k_rows,
                "full_result": full_row,
            }
        )
        metric_sums["continuous_gt_inv_mae"] += cont_gt_inv_mae
        metric_sums["token_unique_count"] += float(unique_tokens)
        metric_counts["samples"] += 1
        if unique_tokens <= 1:
            metric_sums["collapsed_samples"] += 1.0

    aggregates_by_k = {}
    for k in token_ks:
        count = max(prefix_counts[str(k)], 1)
        aggregates_by_k[str(k)] = {
            key: _round_or_none(value / count)
            for key, value in prefix_metrics[str(k)].items()
        }

    full_aggregate = None
    if args.include_full and full_count > 0:
        full_aggregate = {
            key: _round_or_none(value / full_count)
            for key, value in full_metrics.items()
        }
    gt_codec_aggregate = None
    if args.evaluate_gt_codec and gt_codec_count > 0:
        gt_codec_aggregate = {
            key: _round_or_none(value / gt_codec_count)
            for key, value in gt_codec_metrics.items()
        }

    ordered_endpoint = [aggregates_by_k[str(k)]["endpoint_drift_m"] for k in token_ks]
    ordered_pos_mae = [aggregates_by_k[str(k)]["cont_disc_pos_mae_m"] for k in token_ks]
    ordered_inv_mae = [aggregates_by_k[str(k)]["cont_disc_inv_mae"] for k in token_ks]
    token_hist = _token_histogram(all_pred_tokens)
    gt_token_hist = _token_histogram(all_gt_tokens)
    token_position_unique_counts = _token_position_unique_counts(token_position_values)
    gt_token_position_unique_counts = _token_position_unique_counts(gt_token_position_values)
    unique_pred_sequences = _unique_sequence_count(pred_token_sequences)
    unique_gt_sequences = _unique_sequence_count(gt_token_sequences)
    gt_majority_baseline = None
    if gt_token_hist:
        total_gt_tokens = sum(gt_token_hist.values())
        gt_majority_baseline = max(gt_token_hist.values()) / max(total_gt_tokens, 1)

    payload = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "output_dir": str(output_dir),
        "overrides": dotlist,
        "holdout_mod": int(args.holdout_mod),
        "holdout_fold": int(args.holdout_fold),
        "sample_count": int(metric_counts["samples"]),
        "token_prefix_ks": token_ks,
        "tokenize_continuous_head": bool(args.tokenize_continuous_head),
        "evaluate_gt_codec": bool(args.evaluate_gt_codec),
        "continuous": {
            "avg_gt_inv_mae": _round_or_none(metric_sums["continuous_gt_inv_mae"] / max(metric_counts["samples"], 1)),
            "avg_token_unique_count": _round_or_none(metric_sums["token_unique_count"] / max(metric_counts["samples"], 1)),
            "global_token_unique_count": int(len(token_hist)),
            "unique_token_sequence_count": int(unique_pred_sequences),
            "repeated_token_sequence_fraction": _round_or_none(
                1.0 - (unique_pred_sequences / max(len(pred_token_sequences), 1))
            )
            if pred_token_sequences
            else None,
            "token_position_unique_counts": token_position_unique_counts,
            "avg_token_accuracy": _round_or_none(metric_sums["token_accuracy"] / metric_counts["token_accuracy"])
            if metric_counts["token_accuracy"] > 0
            else None,
            "gt_majority_token_accuracy_baseline": _round_or_none(gt_majority_baseline),
            "collapsed_sample_fraction": _round_or_none(metric_sums["collapsed_samples"] / max(metric_counts["samples"], 1)),
            "token_histogram": token_hist,
            "gt_avg_token_unique_count": _round_or_none(
                metric_sums["gt_token_unique_count"] / metric_counts["gt_tokens"]
            )
            if metric_counts["gt_tokens"] > 0
            else None,
            "gt_global_token_unique_count": int(len(gt_token_hist)),
            "gt_unique_token_sequence_count": int(unique_gt_sequences),
            "gt_repeated_token_sequence_fraction": _round_or_none(
                1.0 - (unique_gt_sequences / max(len(gt_token_sequences), 1))
            )
            if gt_token_sequences
            else None,
            "gt_token_position_unique_counts": gt_token_position_unique_counts,
            "gt_collapsed_sample_fraction": _round_or_none(
                metric_sums["gt_collapsed_samples"] / metric_counts["gt_tokens"]
            )
            if metric_counts["gt_tokens"] > 0
            else None,
            "gt_token_histogram": gt_token_hist,
        },
        "prefix_aggregates": aggregates_by_k,
        "full_aggregate": full_aggregate,
        "gt_codec_aggregate": gt_codec_aggregate,
        "monotonicity": {
            "endpoint_drift_nonincreasing_with_k": _monotonic_nonincreasing(ordered_endpoint),
            "position_mae_nonincreasing_with_k": _monotonic_nonincreasing(ordered_pos_mae),
            "invariant_mae_nonincreasing_with_k": _monotonic_nonincreasing(ordered_inv_mae),
        },
        "rows": per_case_rows,
    }

    save_json = Path(args.save_json) if args.save_json else output_dir / "discrete_offline_sweep.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
