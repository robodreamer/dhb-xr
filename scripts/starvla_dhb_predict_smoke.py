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
    run_id = getattr(cfg, "run_id", "dhb_predict_smoke")
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
        missing = [k for k in missing if not k.startswith("qwen_vl_interface.")]
        if missing or unexpected:
            raise RuntimeError(
                "Quantized predict checkpoint load mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return

    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", default=None)
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
        raise RuntimeError("CUDA is required for Qwen+DHB predict smoke.")

    device = torch.device("cuda")

    model = build_framework(cfg)
    if args.checkpoint:
        _load_rollout_checkpoint(model, args.checkpoint)
    model.eval()
    if not _uses_quantized_vlm(model):
        model = model.to(device)

    dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
    batch = next(iter(dataloader))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        pred = model.predict_action(batch)
        traj = model.predict_trajectory(batch) if hasattr(model, "predict_trajectory") else None
    end.record()
    torch.cuda.synchronize(device)

    actions = pred["normalized_actions"]
    target = None
    if batch and "action" in batch[0]:
        target = batch[0]["action"][-actions.shape[1] :]
        target = target.tolist() if hasattr(target, "tolist") else list(target)

    summary = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "overrides": dotlist,
        "output_dir": str(output_dir),
        "batch_len": len(batch),
        "predict_time_ms": round(start.elapsed_time(end), 2),
        "gpu_name": torch.cuda.get_device_name(device),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
        "pred_shape": list(actions.shape),
        "pred_first_action": actions[0, 0].tolist(),
        "pred_last_action": actions[0, -1].tolist(),
    }
    if traj is not None and "pred_contact_probs" in traj:
        contact_probs = traj["pred_contact_probs"]
        if contact_probs.shape[-1] >= 3:
            summary["pred_hold_probs_head"] = contact_probs[0, :, 0].tolist()
            summary["pred_close_probs_head"] = contact_probs[0, :, 1].tolist()
            summary["pred_open_probs_head"] = contact_probs[0, :, 2].tolist()
        else:
            summary["pred_close_probs_head"] = contact_probs[0, :, 0].tolist()
            if contact_probs.shape[-1] > 1:
                summary["pred_open_probs_head"] = contact_probs[0, :, 1].tolist()
    if target is not None:
        import numpy as np

        target_arr = np.asarray(target, dtype=np.float32)
        pred_arr = np.asarray(actions[0], dtype=np.float32)
        summary["target_first_action"] = target_arr[0].tolist()
        summary["target_last_action"] = target_arr[-1].tolist()
        summary["chunk_mae"] = float(np.abs(pred_arr - target_arr).mean())
        if batch and "dhb.close_contact" in batch[0]:
            summary["target_close_contact"] = np.asarray(batch[0]["dhb.close_contact"], dtype=np.float32)[
                -actions.shape[1] :
            ].tolist()
        if batch and "dhb.open_contact" in batch[0]:
            summary["target_open_contact"] = np.asarray(batch[0]["dhb.open_contact"], dtype=np.float32)[
                -actions.shape[1] :
            ].tolist()
        if batch and "dhb.gripper_event" in batch[0]:
            summary["target_gripper_event"] = np.asarray(batch[0]["dhb.gripper_event"], dtype=np.int64)[
                -actions.shape[1] :
            ].tolist()

    save_json = Path(args.save_json) if args.save_json else output_dir / "predict_smoke_summary.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
