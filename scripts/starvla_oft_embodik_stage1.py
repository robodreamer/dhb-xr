from __future__ import annotations

import argparse
import functools
import json
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch
from libero.libero import benchmark
from omegaconf import OmegaConf

from dhb_xr.integration.vla.chunk_rollout import ChunkRolloutConfig, TrajectoryChunkRunner
from dhb_xr.integration.vla.libero import (
    build_libero_state_vector,
    build_state_history_window,
    integrate_delta_action_chunk,
    get_state_history_window,
    load_run_action_stats,
    unnormalize_actions,
)
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


LIBERO_DATA_ROOT = Path("/home/andypark/Projects/data/libero")
LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD = 0.015


@contextmanager
def _libero_torch_load_compat():
    original_torch_load = torch.load

    @functools.wraps(original_torch_load)
    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def _default_output_dir(cfg) -> Path:
    run_root_dir = Path(getattr(cfg, "run_root_dir", "_workspace/starvla_runs"))
    run_id = getattr(cfg, "run_id", "qwenoft_embodik_stage1")
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
                "Quantized checkpoint load mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return
    model.load_state_dict(state_dict, strict=True)


def _demo_hdf5_path(task_suite_name: str, task_name: str) -> Path:
    return LIBERO_DATA_ROOT / task_suite_name / f"{task_name}_demo.hdf5"


def _load_demo_episode(task_suite_name: str, task_name: str, episode_idx: int) -> dict:
    hdf5_path = _demo_hdf5_path(task_suite_name, task_name)
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][f"demo_{episode_idx}"]
        robot_states = np.asarray(demo["robot_states"], dtype=np.float32)
        agentview = np.asarray(demo["obs"]["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(demo["obs"]["eye_in_hand_rgb"], dtype=np.uint8)
        init_state = np.asarray(demo["states"][0], dtype=np.float32)

    ee_pos = robot_states[:, 2:5].astype(np.float32)
    ee_quat_wxyz = robot_states[:, 5:9].astype(np.float32)
    ee_quat_wxyz = ee_quat_wxyz / np.linalg.norm(ee_quat_wxyz, axis=1, keepdims=True)
    grip_qpos = robot_states[:, 0].astype(np.float32)
    gripper_cmd = np.where(grip_qpos > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD, -1.0, 1.0).astype(np.float32)

    return {
        "agentview_rgb": agentview,
        "wrist_rgb": wrist,
        "ee_pos": ee_pos,
        "ee_quat_wxyz": ee_quat_wxyz,
        "gripper_qpos": grip_qpos,
        "gripper_cmd": gripper_cmd,
        "init_state": init_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--replan_every", type=int, default=4)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gripper-source", choices=("demo", "model"), default="demo")
    parser.add_argument("--max-first-offset-mm", type=float, default=0.0)
    parser.add_argument("--max-step-mm", type=float, default=0.0)
    parser.add_argument("--smooth-window", type=int, default=0)
    parser.add_argument("--blend-overlap-alpha", type=float, default=0.0)
    parser.add_argument("--blend-mode", choices=("constant", "ramp"), default="constant")
    parser.add_argument("--commit-prefix-steps", type=int, default=0)
    parser.add_argument("--save_npz", default=None)
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.6)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.4)
    parser.add_argument("--gripper-transition-debounce-steps", type=int, default=1)
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)

    output_dir = _default_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_norm_stats = load_run_action_stats(output_dir, checkpoint=args.checkpoint)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    demo = _load_demo_episode(args.task_suite_name, task.name, args.episode_idx)
    num_demo_steps = len(demo["ee_pos"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StarVLA OFT embodik stage-1 smoke.")

    model = build_framework(cfg)
    _load_rollout_checkpoint(model, args.checkpoint)
    model.eval()

    state_history_window = get_state_history_window(cfg)
    demo_states = np.stack(
        [
            build_libero_state_vector(
                demo["ee_pos"][i],
                demo["ee_quat_wxyz"][i],
                float(demo["gripper_qpos"][i]),
            )
            for i in range(num_demo_steps)
        ],
        axis=0,
    ).astype(np.float32)

    all_positions = []
    all_quaternions = []
    all_gripper = []
    debug_chunks = []
    chunk_cfg = ChunkRolloutConfig(
        max_first_offset_m=max(0.0, args.max_first_offset_mm) / 1000.0,
        max_step_m=max(0.0, args.max_step_mm) / 1000.0,
        smooth_window=max(0, args.smooth_window),
        blend_overlap_alpha=float(args.blend_overlap_alpha),
        blend_mode=args.blend_mode,
        commit_prefix_steps=max(0, args.commit_prefix_steps),
        gripper_source=args.gripper_source,
        gripper_threshold=float(args.gripper_threshold),
        gripper_close_threshold=float(args.gripper_close_threshold),
        gripper_open_threshold=float(args.gripper_open_threshold),
        gripper_transition_debounce_steps=max(1, int(args.gripper_transition_debounce_steps)),
    )
    chunk_runner = TrajectoryChunkRunner(chunk_cfg)
    t = 0
    last_gripper_cmd = float(demo["gripper_cmd"][0]) if len(demo["gripper_cmd"]) else -1.0
    while t < num_demo_steps:
        state_window = build_state_history_window(
            demo_states[max(0, t - state_history_window + 1) : t + 1],
            history_window=state_history_window,
        )
        example = {
            "image": [
                np.ascontiguousarray(demo["agentview_rgb"][t]),
                np.ascontiguousarray(demo["wrist_rgb"][t]),
            ],
            "lang": task.language,
            "state": state_window,
        }

        with torch.inference_mode():
            pred = model.predict_action([example])

        normalized_chunk = np.asarray(pred["normalized_actions"], dtype=np.float32)
        raw_actions = unnormalize_actions(normalized_chunk[0], action_norm_stats)
        chunk_pos, chunk_quat = integrate_delta_action_chunk(
            demo["ee_pos"][t],
            demo["ee_quat_wxyz"][t],
            raw_actions,
        )
        result = {
            "positions": chunk_pos[None, ...],
            "quaternions": chunk_quat[None, ...],
            "normalized_actions": normalized_chunk,
        }

        n_use = min(args.replan_every, len(chunk_pos), num_demo_steps - t)
        prepared = chunk_runner.prepare_chunk(
            result,
            anchor_pos=demo["ee_pos"][t],
            current_start=t,
            n_use=n_use,
            initial_gripper_cmd=last_gripper_cmd,
            demo_gripper_chunk=demo["gripper_cmd"][t : t + n_use],
        )

        all_positions.append(prepared["positions"])
        all_quaternions.append(prepared["quaternions"])
        all_gripper.append(prepared["gripper_cmds"])
        if len(prepared["gripper_cmds"]):
            last_gripper_cmd = float(prepared["gripper_cmds"][-1])

        gt_chunk = demo["ee_pos"][t : t + n_use]
        pos_err_mm = np.linalg.norm(prepared["positions"] - gt_chunk, axis=1) * 1000.0
        debug_chunks.append(
            {
                "t_start": int(t),
                "n_use": int(n_use),
                "chunk_mean_err_mm": float(np.mean(pos_err_mm)),
                "chunk_max_err_mm": float(np.max(pos_err_mm)),
                "first_pred_pos": prepared["first_pred_pos"],
                "first_stabilized_pos": prepared["first_stabilized_pos"],
                "first_gt_pos": gt_chunk[0].tolist(),
                "first_raw_action": raw_actions[0].tolist() if len(raw_actions) else None,
                "phase_mode": "default",
                **prepared["debug"],
            }
        )
        t += n_use

    positions = np.concatenate(all_positions, axis=0)
    quaternions = np.concatenate(all_quaternions, axis=0)
    gripper = np.concatenate(all_gripper, axis=0)

    save_npz = Path(args.save_npz) if args.save_npz else output_dir / "embodik_stage1_oft_chunk.npz"
    np.savez(
        save_npz,
        positions=positions,
        quaternions=quaternions,
        gripper=gripper,
        init_state=demo["init_state"],
        task_language=task.language,
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        episode_idx=args.episode_idx,
    )

    summary = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "overrides": dotlist,
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "episode_idx": args.episode_idx,
        "task_language": task.language,
        "replan_every": args.replan_every,
        "gripper_source": args.gripper_source,
        "max_first_offset_mm": float(args.max_first_offset_mm),
        "max_step_mm": float(args.max_step_mm),
        "smooth_window": int(args.smooth_window),
        "blend_overlap_alpha": float(args.blend_overlap_alpha),
        "blend_mode": args.blend_mode,
        "commit_prefix_steps": int(args.commit_prefix_steps),
        "gripper_threshold": float(args.gripper_threshold),
        "gripper_close_threshold": float(args.gripper_close_threshold),
        "gripper_open_threshold": float(args.gripper_open_threshold),
        "gripper_transition_debounce_steps": int(args.gripper_transition_debounce_steps),
        "positions_shape": list(positions.shape),
        "quaternions_shape": list(quaternions.shape),
        "gripper_shape": list(gripper.shape),
        "save_npz": str(save_npz),
        "chunks": debug_chunks,
    }

    if args.save_json:
        save_json = Path(args.save_json)
        save_json.parent.mkdir(parents=True, exist_ok=True)
        save_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
