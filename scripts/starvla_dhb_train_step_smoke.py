from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _default_output_dir(cfg) -> Path:
    run_root_dir = Path(getattr(cfg, "run_root_dir", "_workspace/starvla_runs"))
    run_id = getattr(cfg, "run_id", "dhb_train_step_smoke")
    return run_root_dir / run_id


def _json_ready(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return value


def _freeze_qwen(model) -> int:
    frozen = 0
    qwen = getattr(model, "qwen_vl_interface", None)
    if qwen is None:
        return frozen
    for param in qwen.parameters():
        param.requires_grad = False
        frozen += param.numel()
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--save_json", default=None)
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)

    cfg.output_dir = str(_default_output_dir(cfg))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for WM4A_DHB train-step smoke.")

    device = torch.device("cuda")

    model = build_framework(cfg)
    frozen_params = _freeze_qwen(model)
    model.train()

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    batch = next(iter(dataloader))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)

    optimizer.zero_grad(set_to_none=True)
    t0.record()
    output = model.forward(batch)
    loss = output["action_loss"]
    loss.backward()
    optimizer.step()
    t1.record()
    torch.cuda.synchronize(device)

    summary = {
        "config_yaml": args.config_yaml,
        "overrides": dotlist,
        "output_dir": str(output_dir),
        "batch_len": len(batch),
        "frozen_qwen_params": frozen_params,
        "trainable_params": sum(p.numel() for p in trainable_params),
        "step_time_ms": round(t0.elapsed_time(t1), 2),
        "metrics": {key: _json_ready(value) for key, value in output.items()},
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
    }

    save_json = Path(args.save_json) if args.save_json else output_dir / "train_step_smoke_summary.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
