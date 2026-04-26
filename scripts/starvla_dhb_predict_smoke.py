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
        missing = [
            k
            for k in missing
            if not k.startswith("qwen_vl_interface.")
            and "committed_prefix_" not in k
        ]
        if missing or unexpected:
            raise RuntimeError(
                "Quantized predict checkpoint load mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return

    model.load_state_dict(state_dict, strict=True)


def _extract_target_invariants(example: dict, horizon: int):
    linear = example.get("dhb.linear_invariants")
    angular = example.get("dhb.angular_invariants")
    if linear is None or angular is None:
        direct = example.get("dhb_invariants") or example.get("invariants") or example.get("gt_invariants")
        if direct is None:
            return None
        target = torch.as_tensor(direct, dtype=torch.float32)
    else:
        target = torch.cat(
            [
                torch.as_tensor(linear, dtype=torch.float32),
                torch.as_tensor(angular, dtype=torch.float32),
            ],
            dim=-1,
        )
    return target[-horizon:]


def _finite_stats(arr):
    import numpy as np

    values = np.asarray(arr, dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _temporal_l2_mean(arr):
    import numpy as np

    values = np.asarray(arr, dtype=np.float32)
    if values.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(values, axis=0), axis=-1).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--use-discrete-tokens", action="store_true")
    parser.add_argument("--emit-oat-tokens", action="store_true")
    parser.add_argument("--token-prefix-keep-k", type=int, default=None)
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
        traj = (
            model.predict_trajectory(
                batch,
                use_discrete_tokens=args.use_discrete_tokens,
                emit_oat_tokens=args.emit_oat_tokens or args.use_discrete_tokens,
                token_prefix_keep_k=args.token_prefix_keep_k,
            )
            if hasattr(model, "predict_trajectory")
            else None
        )
        forward_metrics = model.forward(batch)
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
    if traj is not None and "pred_gripper_state_probs" in traj:
        state_probs = traj["pred_gripper_state_probs"]
        summary["pred_open_probs_head"] = state_probs[0, :, 0].tolist()
        if state_probs.shape[-1] > 1:
            summary["pred_close_probs_head"] = state_probs[0, :, 1].tolist()
    elif traj is not None and "pred_contact_probs" in traj:
        contact_probs = traj["pred_contact_probs"]
        contact_mode = str(traj.get("contact_mode", "event"))
        if contact_mode == "gripper_state":
            summary["pred_open_probs_head"] = contact_probs[0, :, 0].tolist()
            if contact_probs.shape[-1] > 1:
                summary["pred_close_probs_head"] = contact_probs[0, :, 1].tolist()
        elif contact_probs.shape[-1] >= 3:
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
        pred_action_delta_l2 = _temporal_l2_mean(pred_arr)
        target_action_delta_l2 = _temporal_l2_mean(target_arr)
        summary["pred_action_step_delta_l2_mean"] = pred_action_delta_l2
        summary["target_action_step_delta_l2_mean"] = target_action_delta_l2
        summary["action_temporal_l2_ratio"] = float(
            pred_action_delta_l2 / max(target_action_delta_l2, 1e-8)
        )
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
        if batch and "dhb.gripper_state" in batch[0]:
            summary["target_gripper_state"] = np.asarray(batch[0]["dhb.gripper_state"], dtype=np.int64)[
                -actions.shape[1] :
            ].tolist()
        if batch and "dhb.action_valid_mask" in batch[0]:
            valid_mask = np.asarray(batch[0]["dhb.action_valid_mask"], dtype=np.float32)[-actions.shape[1] :]
            summary["target_action_valid_mask"] = valid_mask.tolist()
            summary["valid_action_fraction"] = float(valid_mask.mean())
    if traj is not None and "pred_invariants" in traj:
        import numpy as np

        pred_inv = np.asarray(traj["pred_invariants"][0], dtype=np.float32)
        summary["pred_invariants_shape"] = list(traj["pred_invariants"].shape)
        summary["pred_inv_first"] = pred_inv[0].tolist()
        summary["pred_inv_last"] = pred_inv[-1].tolist()
        summary["pred_inv_stats"] = _finite_stats(pred_inv)
        summary["pred_inv_step_delta_l2_mean"] = _temporal_l2_mean(pred_inv)
        summary["pred_inv_temporal_std_mean"] = float(pred_inv.std(axis=0).mean())
        summary["pred_inv_first_last_l2"] = float(np.linalg.norm(pred_inv[-1] - pred_inv[0]))
        target_inv = _extract_target_invariants(batch[0], pred_inv.shape[0])
        if target_inv is not None:
            target_inv_arr = target_inv.numpy().astype(np.float32)
            inv_abs = np.abs(pred_inv - target_inv_arr)
            inv_l2 = np.linalg.norm(pred_inv - target_inv_arr, axis=-1)
            summary["target_inv_first"] = target_inv_arr[0].tolist()
            summary["target_inv_last"] = target_inv_arr[-1].tolist()
            summary["target_inv_stats"] = _finite_stats(target_inv_arr)
            target_inv_delta_l2 = _temporal_l2_mean(target_inv_arr)
            summary["target_inv_step_delta_l2_mean"] = target_inv_delta_l2
            summary["target_inv_temporal_std_mean"] = float(target_inv_arr.std(axis=0).mean())
            summary["target_inv_first_last_l2"] = float(np.linalg.norm(target_inv_arr[-1] - target_inv_arr[0]))
            summary["inv_temporal_l2_ratio"] = float(
                summary["pred_inv_step_delta_l2_mean"] / max(target_inv_delta_l2, 1e-8)
            )
            summary["inv_temporal_std_ratio"] = float(
                summary["pred_inv_temporal_std_mean"]
                / max(summary["target_inv_temporal_std_mean"], 1e-8)
            )
            summary["inv_mae"] = float(inv_abs.mean())
            summary["inv_max_abs"] = float(inv_abs.max())
            summary["inv_l2_per_step_mean"] = float(inv_l2.mean())
            summary["inv_l2_per_step_max"] = float(inv_l2.max())
            if batch and "dhb.action_valid_mask" in batch[0]:
                valid_mask = np.asarray(batch[0]["dhb.action_valid_mask"], dtype=np.float32)[-pred_inv.shape[0] :]
                valid = valid_mask > 0.5
                if bool(valid.any()):
                    summary["valid_inv_mae"] = float(inv_abs[valid].mean())
                    summary["valid_inv_l2_per_step_mean"] = float(inv_l2[valid].mean())
                    summary["valid_target_inv_stats"] = _finite_stats(target_inv_arr[valid])
                    summary["valid_pred_inv_stats"] = _finite_stats(pred_inv[valid])
                    if int(valid.sum()) > 1:
                        valid_pred_delta_l2 = _temporal_l2_mean(pred_inv[valid])
                        valid_target_delta_l2 = _temporal_l2_mean(target_inv_arr[valid])
                        summary["valid_pred_inv_step_delta_l2_mean"] = valid_pred_delta_l2
                        summary["valid_target_inv_step_delta_l2_mean"] = valid_target_delta_l2
                        summary["valid_inv_temporal_l2_ratio"] = float(
                            valid_pred_delta_l2 / max(valid_target_delta_l2, 1e-8)
                        )
    for key in (
        "action_loss",
        "loss_inv",
        "loss_inv_raw",
        "loss_inv_delta",
        "loss_act",
        "loss_contact",
        "loss_contact_phase",
        "loss_gripper_state",
        "oat_loss",
        "loss_oat_token",
    ):
        if key in forward_metrics:
            value = forward_metrics[key]
            if value is not None:
                summary[f"forward_{key}"] = float(value.detach().float().cpu().item())
    if traj is not None and "pred_oat_tokens" in traj:
        summary["pred_oat_tokens_shape"] = list(traj["pred_oat_tokens"].shape)
        summary["pred_oat_tokens_first"] = traj["pred_oat_tokens"][0].tolist()
    if traj is not None and "pred_invariants_token_recon" in traj:
        summary["pred_invariants_token_recon_shape"] = list(traj["pred_invariants_token_recon"].shape)
        summary["pred_invariants_execute_shape"] = list(traj["pred_invariants_execute"].shape)
        summary["token_prefix_keep_k"] = (
            traj["token_prefix_keep_k"].tolist() if "token_prefix_keep_k" in traj else None
        )

    save_json = Path(args.save_json) if args.save_json else output_dir / "predict_smoke_summary.json"
    save_json.parent.mkdir(parents=True, exist_ok=True)
    save_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
