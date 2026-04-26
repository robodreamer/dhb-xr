from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_PATH = REPO_ROOT / "vendor" / "starVLA"
SRC_PATH = REPO_ROOT / "src"
for path in (VENDOR_PATH, SRC_PATH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _default_output_dir(cfg) -> Path:
    run_root_dir = Path(getattr(cfg, "run_root_dir", "_workspace/starvla_runs"))
    run_id = getattr(cfg, "run_id", "dhb_token_overfit_probe")
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
                "Quantized token-overfit checkpoint load mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return

    model.load_state_dict(state_dict, strict=True)


def _stack_gt_invariants(examples: list[dict]) -> torch.Tensor:
    chunks = []
    for example in examples:
        if "dhb_invariants" in example:
            inv = np.asarray(example["dhb_invariants"], dtype=np.float32)
        else:
            linear = np.asarray(example["dhb.linear_invariants"], dtype=np.float32)
            angular = np.asarray(example["dhb.angular_invariants"], dtype=np.float32)
            inv = np.concatenate([linear, angular], axis=-1)
        chunks.append(inv)
    return torch.as_tensor(np.stack(chunks), dtype=torch.float32)


def _iter_examples(dataloader) -> Iterable[dict]:
    while True:
        for batch in dataloader:
            for example in batch:
                yield example


def _token_branch_param(name: str) -> bool:
    return any(
        part in name
        for part in (
            "oat_token_query",
            "oat_token_decoder",
            "oat_token_head",
            "oat_token_context_proj",
        )
    )


def _action_head_except_oat_param(name: str) -> bool:
    if name.startswith("oat."):
        return False
    return True


def _select_trainable_params(backend: torch.nn.Module, scope: str) -> list[torch.nn.Parameter]:
    selectors = {
        "token_branch": _token_branch_param,
        "action_head_except_oat": _action_head_except_oat_param,
    }
    selector = selectors[scope]
    params = []
    for name, param in backend.named_parameters():
        trainable = selector(name)
        param.requires_grad_(trainable)
        if trainable:
            params.append(param)
    return params


def _token_counts(tokens: list[list[int]]) -> dict:
    if not tokens:
        return {
            "global_unique_count": 0,
            "unique_sequence_count": 0,
            "per_position_unique_counts": [],
            "histogram": {},
        }
    arr = np.asarray(tokens, dtype=np.int64)
    flat = arr.reshape(-1)
    hist = Counter(int(v) for v in flat.tolist())
    return {
        "global_unique_count": int(len(hist)),
        "unique_sequence_count": int(len({tuple(row) for row in arr.tolist()})),
        "per_position_unique_counts": [
            int(np.unique(arr[:, idx]).size)
            for idx in range(arr.shape[1])
        ],
        "histogram": {str(k): int(v) for k, v in sorted(hist.items())},
    }


def _majority_baseline(target_tokens: list[list[int]]) -> float | None:
    if not target_tokens:
        return None
    hist = Counter(int(v) for row in target_tokens for v in row)
    total = sum(hist.values())
    return max(hist.values()) / max(total, 1)


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _evaluate_backend(backend, cached: list[dict], device: torch.device) -> dict:
    backend.eval()
    losses = []
    accuracies = []
    sequence_matches = []
    recon_maes = []
    pred_sequences: list[list[int]] = []
    target_sequences: list[list[int]] = []

    with torch.no_grad():
        for item in cached:
            hidden = item["hidden"].to(device=device, dtype=torch.float32)
            state = item["state"]
            state = None if state is None else state.to(device=device, dtype=torch.float32)
            target_inv = item["target_invariants"].to(device=device, dtype=torch.float32)

            result = backend.forward(
                hidden,
                state=state,
                target_invariants=target_inv,
            )
            target_tokens = backend._tokenize_oat_invariants(target_inv)
            logits = result["pred_oat_token_logits"]
            pred_tokens = logits.argmax(dim=-1)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_tokens.reshape(-1))
            losses.append(float(loss.detach().cpu().item()))
            matches = pred_tokens == target_tokens
            accuracies.append(float(matches.to(dtype=torch.float32).mean().detach().cpu().item()))
            sequence_matches.append(float(matches.all(dim=-1).to(dtype=torch.float32).mean().detach().cpu().item()))

            recon = result.get("pred_invariants_token_recon")
            if recon is not None:
                recon_maes.append(float((recon - target_inv).abs().mean().detach().cpu().item()))

            pred_sequences.extend(pred_tokens.detach().cpu().numpy().astype(np.int64).tolist())
            target_sequences.extend(target_tokens.detach().cpu().numpy().astype(np.int64).tolist())

    return {
        "loss_oat_token": _round(float(np.mean(losses)) if losses else None),
        "token_accuracy": _round(float(np.mean(accuracies)) if accuracies else None),
        "sequence_accuracy": _round(float(np.mean(sequence_matches)) if sequence_matches else None),
        "token_recon_inv_mae": _round(float(np.mean(recon_maes)) if recon_maes else None),
        "majority_token_baseline": _round(_majority_baseline(target_sequences)),
        "pred_tokens": _token_counts(pred_sequences),
        "target_tokens": _token_counts(target_sequences),
    }


def _cache_examples(model, dataloader, num_examples: int, horizon: int, device: torch.device) -> list[dict]:
    model.eval()
    cached = []
    source_examples = []
    for example in _iter_examples(dataloader):
        source_examples.append(example)
        if len(source_examples) >= num_examples:
            break

    for idx, example in enumerate(source_examples):
        with torch.no_grad():
            hidden, state = model._encode_examples([example])
        target_inv = _stack_gt_invariants([example])[:, -horizon:, :]
        cached.append(
            {
                "index": idx,
                "hidden": hidden.detach().to(device="cpu", dtype=torch.float32),
                "state": None if state is None else state.detach().to(device="cpu", dtype=torch.float32),
                "target_invariants": target_inv.detach().to(device="cpu", dtype=torch.float32),
            }
        )
        torch.cuda.empty_cache()
    if not cached:
        raise RuntimeError("No examples were cached for the token overfit probe")
    return cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument(
        "--examples_per_step",
        type=int,
        default=0,
        help="Number of cached examples per optimizer step; 0 uses the full cached set.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument(
        "--train_scope",
        choices=("token_branch", "action_head_except_oat"),
        default="token_branch",
        help="Parameter subset to train during the memorization probe.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args, unknown = parser.parse_known_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        raise RuntimeError("CUDA is required for Qwen+DHB token overfit probe.")

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)

    cfg.output_dir = str(_default_output_dir(cfg))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_framework(cfg)
    if args.checkpoint:
        _load_rollout_checkpoint(model, args.checkpoint)

    action_model = getattr(model, "action_model", None)
    backend = getattr(action_model, "backend", None)
    if backend is None or getattr(backend, "oat_token_head", None) is None:
        raise RuntimeError("Token overfit probe requires a DHB action head with supervised OAT token branch.")

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    cached = _cache_examples(
        model,
        dataloader,
        num_examples=max(1, int(args.num_examples)),
        horizon=int(getattr(backend, "action_horizon")),
        device=device,
    )

    if _uses_quantized_vlm(model):
        # Keep the quantized VLM loaded for compatibility, but do all probe
        # optimization through cached hidden states only.
        pass
    backend = backend.to(device=device, dtype=torch.float32)
    params = _select_trainable_params(backend, args.train_scope)
    if not params:
        raise RuntimeError(f"No trainable parameters selected for scope {args.train_scope!r}")

    optimizer = torch.optim.AdamW(
        params,
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )

    initial_metrics = _evaluate_backend(backend, cached, device)
    history = []
    backend.eval()
    examples_per_step = int(args.examples_per_step)
    for step in range(1, int(args.train_steps) + 1):
        if examples_per_step <= 0 or examples_per_step >= len(cached):
            step_items = cached
        else:
            start = (step - 1) * examples_per_step
            step_items = [
                cached[(start + offset) % len(cached)]
                for offset in range(examples_per_step)
            ]
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for item in step_items:
            hidden = item["hidden"].to(device=device, dtype=torch.float32)
            state = item["state"]
            state = None if state is None else state.to(device=device, dtype=torch.float32)
            target_inv = item["target_invariants"].to(device=device, dtype=torch.float32)
            result = backend.forward(
                hidden,
                state=state,
                target_invariants=target_inv,
            )
            losses.append(result["loss_oat_token"])
        loss = torch.stack(losses).mean()
        loss.backward()
        if args.max_grad_norm is not None and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(params, float(args.max_grad_norm))
        optimizer.step()

        if step == 1 or step % max(1, int(args.log_interval)) == 0 or step == int(args.train_steps):
            metrics = _evaluate_backend(backend, cached, device)
            history.append({"step": int(step), **metrics})

    final_metrics = _evaluate_backend(backend, cached, device)
    gate_passed = bool(
        final_metrics["token_accuracy"] is not None
        and final_metrics["sequence_accuracy"] is not None
        and final_metrics["token_accuracy"] >= 0.90
        and final_metrics["sequence_accuracy"] >= 0.75
    )

    payload = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "output_dir": str(output_dir),
        "overrides": dotlist,
        "train_scope": args.train_scope,
        "num_examples": len(cached),
        "train_steps": int(args.train_steps),
        "examples_per_step": int(args.examples_per_step),
        "learning_rate": float(args.learning_rate),
        "gate": {
            "passed": gate_passed,
            "min_token_accuracy": 0.90,
            "min_sequence_accuracy": 0.75,
        },
        "initial": initial_metrics,
        "final": final_metrics,
        "history": history,
    }

    save_json = Path(args.save_json) if args.save_json else output_dir / "token_overfit_probe.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
