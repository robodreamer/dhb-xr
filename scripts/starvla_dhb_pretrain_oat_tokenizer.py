#!/usr/bin/env python3
"""Standalone OAT tokenizer pretrain/eval gate for StarVLA DHB chunks.

This script intentionally avoids building the VLA backbone. It loads DHB
invariant chunks from the StarVLA LeRobot dataset described by a config YAML,
trains ``dhb_xr.tokenization.oat_tokenizer.OATTokenizer`` directly, and reports
reconstruction plus token-collapse diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

import numpy as np


T = TypeVar("T")
REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PATH = REPO_ROOT / "vendor" / "starVLA"
SRC_PATH = REPO_ROOT / "src"
for _path in (VENDOR_PATH, SRC_PATH):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def stack_dhb_invariants(
    sample: dict,
    *,
    linear_key: str = "dhb.linear_invariants",
    angular_key: str = "dhb.angular_invariants",
) -> np.ndarray:
    """Return one ``(T, D_linear + D_angular)`` invariant chunk."""
    if linear_key not in sample or angular_key not in sample:
        missing = [key for key in (linear_key, angular_key) if key not in sample]
        raise KeyError(f"Missing DHB invariant field(s): {missing}")

    linear = np.asarray(sample[linear_key], dtype=np.float32)
    angular = np.asarray(sample[angular_key], dtype=np.float32)
    if linear.ndim != 2 or angular.ndim != 2:
        raise ValueError(
            f"Expected rank-2 invariant arrays, got linear={linear.shape}, angular={angular.shape}"
        )
    if linear.shape[0] != angular.shape[0]:
        raise ValueError(
            f"Linear/angular horizons differ: {linear.shape[0]} != {angular.shape[0]}"
        )
    if linear.shape[0] == 0:
        raise ValueError("Invariant chunk is empty")
    return np.concatenate([linear, angular], axis=-1).astype(np.float32, copy=False)


def split_by_holdout(
    chunks: Sequence[np.ndarray],
    *,
    holdout_mod: int = 10,
    holdout_fold: int = 0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Deterministically split chunks by global index modulo."""
    if holdout_mod <= 1:
        raise ValueError("holdout_mod must be > 1")
    if holdout_fold < 0 or holdout_fold >= holdout_mod:
        raise ValueError("holdout_fold must be in [0, holdout_mod)")

    train, eval_ = [], []
    for idx, chunk in enumerate(chunks):
        if idx % holdout_mod == holdout_fold:
            eval_.append(chunk)
        else:
            train.append(chunk)
    return train, eval_


def split_items_by_holdout(
    items: Sequence[T],
    *,
    holdout_mod: int = 10,
    holdout_fold: int = 0,
) -> tuple[list[T], list[T]]:
    """Deterministically split arbitrary metadata with the same rule as chunks."""
    if holdout_mod <= 1:
        raise ValueError("holdout_mod must be > 1")
    if holdout_fold < 0 or holdout_fold >= holdout_mod:
        raise ValueError("holdout_fold must be in [0, holdout_mod)")

    train, eval_ = [], []
    for idx, item in enumerate(items):
        if idx % holdout_mod == holdout_fold:
            eval_.append(item)
        else:
            train.append(item)
    return train, eval_


def pad_or_trim_chunks(chunks: Sequence[np.ndarray], *, horizon: int | None = None) -> np.ndarray:
    """Stack chunks into ``(N, T, D)``, padding short chunks with their final row."""
    if not chunks:
        raise ValueError("No chunks provided")
    if horizon is None:
        horizon = int(chunks[0].shape[0])
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    dim = int(chunks[0].shape[1])
    out = np.zeros((len(chunks), horizon, dim), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        arr = np.asarray(chunk, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != dim:
            raise ValueError(f"Chunk {i} has incompatible shape {arr.shape}; expected (*, {dim})")
        n = min(horizon, arr.shape[0])
        out[i, :n] = arr[:n]
        if n < horizon:
            out[i, n:] = arr[n - 1]
    return out


def reconstruction_metrics(target: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    """Compute scale-free and absolute reconstruction diagnostics."""
    target = np.asarray(target, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    if target.shape != recon.shape:
        raise ValueError(f"Shape mismatch: target={target.shape}, recon={recon.shape}")

    err = recon - target
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(math.sqrt(mse))
    max_abs = float(np.max(np.abs(err))) if err.size else 0.0
    denom = float(np.mean((target - np.mean(target)) ** 2))
    r2 = 1.0 - (mse / denom) if denom > 0.0 else (1.0 if mse == 0.0 else 0.0)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs": max_abs,
        "r2": float(r2),
    }


def token_usage_metrics(
    tokens: np.ndarray,
    *,
    codebook_size: int | None = None,
    collapsed_unique_threshold: int = 1,
) -> dict:
    """Compute token diversity and collapse diagnostics for ``(N, K)`` tokens."""
    tokens = np.asarray(tokens, dtype=np.int64)
    if tokens.ndim != 2:
        raise ValueError(f"Expected tokens with shape (N, K), got {tokens.shape}")
    if tokens.size == 0:
        return {
            "num_samples": int(tokens.shape[0]),
            "num_positions": int(tokens.shape[1]) if tokens.ndim == 2 else 0,
            "unique_token_count": 0,
            "unique_token_fraction": 0.0,
            "token_entropy": 0.0,
            "effective_num_tokens": 0.0,
            "per_position_unique_counts": [],
            "per_sample_unique_mean": 0.0,
            "collapsed_sample_fraction": 0.0,
            "token_histogram": {},
        }

    flat = tokens.reshape(-1)
    counts = Counter(int(v) for v in flat.tolist())
    total = float(flat.size)
    probs = np.asarray([count / total for count in counts.values()], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0
    per_pos = [int(np.unique(tokens[:, i]).size) for i in range(tokens.shape[1])]
    per_sample_unique = np.asarray([np.unique(row).size for row in tokens], dtype=np.float64)
    denom = int(codebook_size) if codebook_size and codebook_size > 0 else int(max(counts) + 1)

    return {
        "num_samples": int(tokens.shape[0]),
        "num_positions": int(tokens.shape[1]),
        "unique_token_count": int(len(counts)),
        "unique_token_fraction": float(len(counts) / denom) if denom > 0 else 0.0,
        "token_entropy": entropy,
        "effective_num_tokens": float(2.0**entropy),
        "per_position_unique_counts": per_pos,
        "per_sample_unique_mean": float(per_sample_unique.mean()) if per_sample_unique.size else 0.0,
        "collapsed_sample_fraction": float(np.mean(per_sample_unique <= collapsed_unique_threshold)),
        "token_histogram": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def gate_thresholds(
    *,
    max_collapsed_fraction: float,
    min_unique_tokens: int,
    min_effective_tokens: float,
    min_per_position_unique: int,
    min_reconstruction_r2: float,
) -> dict[str, float | int]:
    return {
        "max_collapsed_fraction": float(max_collapsed_fraction),
        "min_unique_tokens": int(min_unique_tokens),
        "min_effective_tokens": float(min_effective_tokens),
        "min_per_position_unique": int(min_per_position_unique),
        "min_reconstruction_r2": float(min_reconstruction_r2),
    }


def evaluate_gate(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    """Return strict gate status and human-readable failure reasons."""
    reasons = []
    if not finite_metrics(metrics):
        reasons.append("non_finite_metrics")

    token_usage = metrics.get("token_usage", {})
    reconstruction = metrics.get("reconstruction", {})
    collapsed_fraction = float(token_usage.get("collapsed_sample_fraction", 1.0))
    unique_tokens = int(token_usage.get("unique_token_count", 0))
    effective_tokens = float(token_usage.get("effective_num_tokens", 0.0))
    per_position_unique = token_usage.get("per_position_unique_counts", [])
    reconstruction_r2 = float(reconstruction.get("r2", float("-inf")))
    if collapsed_fraction > float(thresholds["max_collapsed_fraction"]):
        reasons.append(
            "collapsed_sample_fraction "
            f"{collapsed_fraction:.6g} > {float(thresholds['max_collapsed_fraction']):.6g}"
        )
    if unique_tokens < int(thresholds["min_unique_tokens"]):
        reasons.append(f"unique_token_count {unique_tokens} < {int(thresholds['min_unique_tokens'])}")
    if effective_tokens < float(thresholds["min_effective_tokens"]):
        reasons.append(
            "effective_num_tokens "
            f"{effective_tokens:.6g} < {float(thresholds['min_effective_tokens']):.6g}"
        )
    if per_position_unique and min(per_position_unique) < int(thresholds["min_per_position_unique"]):
        reasons.append(
            "min_per_position_unique "
            f"{min(per_position_unique)} < {int(thresholds['min_per_position_unique'])}"
        )
    if reconstruction_r2 < float(thresholds["min_reconstruction_r2"]):
        reasons.append(
            "reconstruction_r2 "
            f"{reconstruction_r2:.6g} < {float(thresholds['min_reconstruction_r2']):.6g}"
        )
    return not reasons, reasons


def parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def maybe_normalize(
    train: np.ndarray,
    eval_: np.ndarray,
    *,
    enabled: bool,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict | None]:
    if not enabled:
        return train, eval_, None
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, eps)
    stats = {
        "mean": mean.reshape(-1).astype(float).tolist(),
        "std": std.reshape(-1).astype(float).tolist(),
    }
    return (train - mean) / std, (eval_ - mean) / std, stats


def unnormalize(arr: np.ndarray, stats: dict | None) -> np.ndarray:
    if stats is None:
        return arr
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(1, 1, -1)
    return arr * std + mean


def iter_dhb_chunks_from_starvla_dataset(dataset, *, max_chunks: int) -> Iterable[tuple[dict, np.ndarray]]:
    """Yield deterministic DHB chunks from a StarVLA LeRobot dataset or mixture."""
    emitted = 0
    global_index = 0
    subsets = getattr(dataset, "datasets", None)
    if subsets is None:
        subsets = [dataset]

    for subset_index, subset in enumerate(subsets):
        all_steps = getattr(subset, "all_steps", None)
        if all_steps is None:
            for row_index in range(len(subset)):
                sample = subset[row_index]
                meta = {
                    "global_index": int(global_index),
                    "subset_index": int(subset_index),
                    "row_index": int(row_index),
                }
                yield meta, stack_dhb_invariants(sample)
                emitted += 1
                global_index += 1
                if emitted >= max_chunks:
                    return
            continue

        for row_index, (traj_id, base_index) in enumerate(all_steps):
            raw = subset.get_step_data(traj_id, base_index)
            sample = subset._pack_sample(subset.transforms(raw))
            meta = {
                "global_index": int(global_index),
                "subset_index": int(subset_index),
                "subset_name": getattr(subset, "repo_id", None)
                or getattr(subset, "repo", None)
                or getattr(subset, "dataset_name", None),
                "row_index": int(row_index),
                "traj_id": str(traj_id),
                "base_index": int(base_index),
            }
            yield meta, stack_dhb_invariants(sample)
            emitted += 1
            global_index += 1
            if emitted >= max_chunks:
                return


def load_starvla_config(config_yaml: str, dotlist_args: Sequence[str]):
    from omegaconf import OmegaConf
    from starVLA.training.trainer_utils.config_tracker import wrap_config
    from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

    dotlist = normalize_dotlist_args(list(dotlist_args))
    cfg = OmegaConf.load(config_yaml)
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return wrap_config(cfg, cli_overrides=dotlist), dotlist


def load_dhb_chunks_from_config(
    config_yaml: str,
    *,
    max_chunks: int,
    dataset_mode: str,
    dotlist_args: Sequence[str],
) -> tuple[list[np.ndarray], list[dict], list[str]]:
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset

    cfg, dotlist = load_starvla_config(config_yaml, dotlist_args)
    dataset = get_vla_dataset(cfg.datasets.vla_data, mode=dataset_mode)
    chunks, metas = [], []
    for meta, chunk in iter_dhb_chunks_from_starvla_dataset(dataset, max_chunks=max_chunks):
        chunks.append(chunk)
        metas.append(meta)
    if not chunks:
        raise RuntimeError("No DHB invariant chunks were loaded from the StarVLA dataset")
    return chunks, metas, dotlist


def infer_num_registers(cfg, default: int) -> int:
    try:
        value = cfg.framework.action_model.num_oat_tokens
        return int(value)
    except Exception:
        return int(default)


def infer_fsq_levels(cfg, default: Sequence[int]) -> list[int]:
    try:
        levels = cfg.framework.action_model.oat_fsq_levels
        return [int(v) for v in levels]
    except Exception:
        return [int(v) for v in default]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_tokenizer(
    *,
    input_dim: int,
    input_horizon: int,
    emb_dim: int,
    latent_dim: int,
    num_registers: int,
    fsq_levels: Sequence[int],
    encoder_depth: int,
    decoder_depth: int,
    num_heads: int,
    dropout: float,
):
    from dhb_xr.tokenization.oat_tokenizer import OATTokenizer

    return OATTokenizer(
        input_dim=input_dim,
        input_horizon=input_horizon,
        emb_dim=emb_dim,
        latent_dim=latent_dim,
        num_registers=num_registers,
        fsq_levels=list(fsq_levels),
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        num_heads=num_heads,
        dropout=dropout,
    )


def compute_register_diversity_loss(model, batch, *, weight: float):
    """Return weighted register-diversity loss if the tokenizer supports it."""
    import torch

    zero = batch.new_zeros(())
    if weight <= 0.0:
        return zero, zero, False
    loss_fn = getattr(model, "register_diversity_loss", None)
    if loss_fn is None:
        return zero, zero, False
    raw = loss_fn(batch)
    if raw is None:
        raw = zero
    if not torch.is_tensor(raw):
        raw = torch.as_tensor(raw, dtype=batch.dtype, device=batch.device)
    return raw * float(weight), raw, True


def train_tokenizer(model, train_array: np.ndarray, args) -> list[dict[str, float]]:
    import torch

    device = torch.device(args.device)
    x_train = torch.as_tensor(train_array, dtype=torch.float32, device=device)
    optimizer = model.get_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )
    model.train()
    history = []
    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, x_train.shape[0], (min(args.batch_size, x_train.shape[0]),), device=device)
        batch = x_train[idx]
        reconstruction_loss = model(batch)
        diversity_loss, raw_diversity_loss, diversity_available = compute_register_diversity_loss(
            model,
            batch,
            weight=float(args.lambda_register_diversity),
        )
        loss = reconstruction_loss + diversity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step == args.train_steps or step % args.log_every == 0:
            row = {
                "step": int(step),
                "loss": float(loss.detach().cpu().item()),
                "reconstruction_loss": float(reconstruction_loss.detach().cpu().item()),
                "register_diversity_loss": float(raw_diversity_loss.detach().cpu().item()),
                "weighted_register_diversity_loss": float(diversity_loss.detach().cpu().item()),
                "register_diversity_available": bool(diversity_available),
                "lambda_register_diversity": float(args.lambda_register_diversity),
            }
            history.append(row)
            print(json.dumps({"train": row}))
    return history


def evaluate_tokenizer(model, eval_array: np.ndarray, *, batch_size: int, prefix_ks: Sequence[int], norm_stats: dict | None) -> dict:
    import torch

    device = next(model.parameters()).device
    model.eval()
    recon_batches = []
    token_batches = []
    prefix_batches = {int(k): [] for k in prefix_ks}
    with torch.inference_mode():
        for start in range(0, eval_array.shape[0], batch_size):
            batch_np = eval_array[start : start + batch_size]
            batch = torch.as_tensor(batch_np, dtype=torch.float32, device=device)
            latents, tokens = model.encode(batch)
            recon = model.decode(latents)
            recon_batches.append(recon.detach().cpu().numpy())
            token_batches.append(tokens.detach().cpu().numpy())
            for k in prefix_batches:
                if k <= model.num_registers:
                    prefix = model.detokenize(tokens[:, :k])
                    prefix_batches[k].append(prefix.detach().cpu().numpy())

    eval_raw = unnormalize(eval_array, norm_stats)
    recon_raw = unnormalize(np.concatenate(recon_batches, axis=0), norm_stats)
    tokens_np = np.concatenate(token_batches, axis=0)
    prefix_metrics = {}
    for k, batches in prefix_batches.items():
        if batches:
            prefix_raw = unnormalize(np.concatenate(batches, axis=0), norm_stats)
            prefix_metrics[str(k)] = reconstruction_metrics(eval_raw, prefix_raw)
        else:
            prefix_metrics[str(k)] = None

    return {
        "reconstruction": reconstruction_metrics(eval_raw, recon_raw),
        "token_usage": token_usage_metrics(tokens_np, codebook_size=int(model.codebook_size)),
        "prefix_reconstruction": prefix_metrics,
    }


def finite_metrics(metrics: dict) -> bool:
    for value in metrics.values():
        if isinstance(value, dict):
            if not finite_metrics(value):
                return False
        elif isinstance(value, list):
            continue
        elif value is not None and isinstance(value, (int, float)) and not math.isfinite(float(value)):
            return False
    return True


def default_output_dir(config_yaml: str) -> Path:
    return REPO_ROOT / "_workspace" / "starvla_runs" / (Path(config_yaml).stem + "_oat_tokenizer")


def chunks_to_load(max_train_chunks: int, max_eval_chunks: int, holdout_mod: int) -> int:
    """Estimate enough chunks to satisfy both sides of a modulo holdout split."""
    train_needed = int(math.ceil(max_train_chunks * holdout_mod / max(1, holdout_mod - 1)))
    eval_needed = int(max_eval_chunks * holdout_mod)
    return max(max_train_chunks + max_eval_chunks, train_needed, eval_needed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--checkpoint_out", default=None)
    parser.add_argument("--dataset_mode", default="val", choices=["train", "val"])
    parser.add_argument("--max_train_chunks", type=int, default=1024)
    parser.add_argument("--max_eval_chunks", type=int, default=256)
    parser.add_argument("--holdout_mod", type=int, default=10)
    parser.add_argument("--holdout_fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--train_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_register_diversity", type=float, default=0.0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=4)
    parser.add_argument("--num_registers", type=int, default=None)
    parser.add_argument("--fsq_levels", default="8,5,5,5")
    parser.add_argument("--encoder_depth", type=int, default=2)
    parser.add_argument("--decoder_depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--prefix_ks", default="1,2,4,8")
    parser.add_argument("--max_collapsed_fraction", type=float, default=0.25)
    parser.add_argument("--min_unique_tokens", type=int, default=8)
    parser.add_argument("--min_effective_tokens", type=float, default=4.0)
    parser.add_argument("--min_per_position_unique", type=int, default=2)
    parser.add_argument("--min_reconstruction_r2", type=float, default=0.0)
    args, unknown = parser.parse_known_args()

    import torch

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    total_needed = chunks_to_load(args.max_train_chunks, args.max_eval_chunks, args.holdout_mod)
    chunks, metas, dotlist = load_dhb_chunks_from_config(
        args.config_yaml,
        max_chunks=total_needed,
        dataset_mode=args.dataset_mode,
        dotlist_args=unknown,
    )
    train_chunks, eval_chunks = split_by_holdout(
        chunks,
        holdout_mod=args.holdout_mod,
        holdout_fold=args.holdout_fold,
    )
    _train_metas, eval_metas = split_items_by_holdout(
        metas,
        holdout_mod=args.holdout_mod,
        holdout_fold=args.holdout_fold,
    )
    train_chunks = train_chunks[: args.max_train_chunks]
    eval_chunks = eval_chunks[: args.max_eval_chunks]
    eval_metas = eval_metas[: args.max_eval_chunks]
    if not train_chunks or not eval_chunks:
        raise RuntimeError(
            f"Need non-empty train/eval splits, got train={len(train_chunks)}, eval={len(eval_chunks)}"
        )

    horizon = int(train_chunks[0].shape[0])
    train_array_raw = pad_or_trim_chunks(train_chunks, horizon=horizon)
    eval_array_raw = pad_or_trim_chunks(eval_chunks, horizon=horizon)
    train_array, eval_array, norm_stats = maybe_normalize(train_array_raw, eval_array_raw, enabled=args.normalize)

    cfg, _ = load_starvla_config(args.config_yaml, dotlist)
    fsq_levels = parse_int_list(args.fsq_levels)
    num_registers = args.num_registers or infer_num_registers(cfg, 8)
    fsq_levels = infer_fsq_levels(cfg, fsq_levels)
    if len(fsq_levels) != args.latent_dim:
        raise ValueError(f"Expected {args.latent_dim} fsq levels, got {fsq_levels}")

    model = build_tokenizer(
        input_dim=int(train_array.shape[-1]),
        input_horizon=int(train_array.shape[1]),
        emb_dim=args.emb_dim,
        latent_dim=args.latent_dim,
        num_registers=int(num_registers),
        fsq_levels=fsq_levels,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(torch.device(args.device))

    train_history = train_tokenizer(model, train_array, args)
    metrics = evaluate_tokenizer(
        model,
        eval_array,
        batch_size=args.batch_size,
        prefix_ks=parse_int_list(args.prefix_ks),
        norm_stats=norm_stats,
    )
    thresholds = gate_thresholds(
        max_collapsed_fraction=args.max_collapsed_fraction,
        min_unique_tokens=args.min_unique_tokens,
        min_effective_tokens=args.min_effective_tokens,
        min_per_position_unique=args.min_per_position_unique,
        min_reconstruction_r2=args.min_reconstruction_r2,
    )
    gate_passed, gate_failures = evaluate_gate(metrics, thresholds)

    summary = {
        "config_yaml": args.config_yaml,
        "overrides": dotlist,
        "dataset_mode": args.dataset_mode,
        "num_loaded_chunks": len(chunks),
        "num_train_chunks": int(train_array.shape[0]),
        "num_eval_chunks": int(eval_array.shape[0]),
        "chunk_shape": list(train_array.shape[1:]),
        "normalization": norm_stats,
        "tokenizer": {
            "input_dim": int(train_array.shape[-1]),
            "input_horizon": int(train_array.shape[1]),
            "emb_dim": int(args.emb_dim),
            "latent_dim": int(args.latent_dim),
            "num_registers": int(num_registers),
            "fsq_levels": fsq_levels,
            "codebook_size": int(model.codebook_size),
            "encoder_depth": int(args.encoder_depth),
            "decoder_depth": int(args.decoder_depth),
            "num_heads": int(args.num_heads),
            "total_params": int(sum(p.numel() for p in model.parameters())),
        },
        "train_history": train_history,
        "metrics": metrics,
        "gate_thresholds": thresholds,
        "gate_failures": gate_failures,
        "first_eval_meta": eval_metas[: min(3, len(eval_metas))],
        "gate_passed": bool(gate_passed),
    }

    if args.checkpoint_out:
        checkpoint_out = Path(args.checkpoint_out)
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "tokenizer": summary["tokenizer"],
                "normalization": norm_stats,
                "metrics": metrics,
            },
            checkpoint_out,
        )
        summary["checkpoint_out"] = str(checkpoint_out)

    output_dir = default_output_dir(args.config_yaml)
    save_json = Path(args.save_json) if args.save_json else output_dir / "oat_tokenizer_pretrain_eval.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
