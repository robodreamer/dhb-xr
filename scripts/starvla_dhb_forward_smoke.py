from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _default_output_dir(cfg) -> Path:
    run_root_dir = Path(getattr(cfg, "run_root_dir", "_workspace/starvla_runs"))
    run_id = getattr(cfg, "run_id", "dhb_forward_smoke")
    return run_root_dir / run_id


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return value


def _uses_quantized_vlm(model) -> bool:
    qwen = getattr(model, "qwen_vl_interface", None)
    if qwen is None:
        return False
    backend = getattr(qwen, "model", None)
    return bool(
        getattr(backend, "is_loaded_in_8bit", False) or getattr(backend, "is_loaded_in_4bit", False)
    )


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
        raise RuntimeError("CUDA is required for WM4A_DHB forward smoke.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build_framework(cfg)
    model.eval()
    if not _uses_quantized_vlm(model):
        model = model.to(device)

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    t0 = time.perf_counter()
    batch = next(iter(dataloader))
    t1 = time.perf_counter()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    t2 = time.perf_counter()
    with torch.inference_mode():
        output = model.forward(batch)
    torch.cuda.synchronize(device)
    t3 = time.perf_counter()

    summary = {
        "config_yaml": args.config_yaml,
        "overrides": dotlist,
        "output_dir": str(output_dir),
        "batch_len": len(batch),
        "data_time_sec": round(t1 - t0, 4),
        "model_time_sec": round(t3 - t2, 4),
        "loss_keys": sorted(output.keys()),
        "metrics": {key: _json_ready(value) for key, value in output.items()},
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
    }

    save_json = Path(args.save_json) if args.save_json else output_dir / "forward_smoke_summary.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
