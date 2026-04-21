from __future__ import annotations

import argparse
import functools
import json
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from dhb_xr.integration.vla.chunk_rollout import ChunkRolloutConfig, PhaseAwareChunkRunner
from dhb_xr.integration.vla.libero import (
    build_libero_state_vector,
    build_state_history_window,
    get_state_history_window,
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
    run_id = getattr(cfg, "run_id", "dhb_starvla_embodik_stage1")
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


def _load_demo_gripper(task_suite_name: str, task_name: str, episode_idx: int) -> np.ndarray:
    hdf5_path = _demo_hdf5_path(task_suite_name, task_name)
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][f"demo_{episode_idx}"]
        robot_states = np.asarray(demo["robot_states"], dtype=np.float32)
    grip_qpos = robot_states[:, 0]
    return np.where(grip_qpos > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD, -1.0, 1.0).astype(np.float32)


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
    parser.add_argument("--replan_every", type=int, default=6)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gripper-source", choices=("demo", "model"), default="demo")
    parser.add_argument("--max-first-offset-mm", type=float, default=0.0)
    parser.add_argument("--max-step-mm", type=float, default=0.0)
    parser.add_argument("--smooth-window", type=int, default=0)
    parser.add_argument("--blend-overlap-alpha", type=float, default=0.0)
    parser.add_argument("--blend-mode", choices=("constant", "ramp"), default="constant")
    parser.add_argument("--commit-prefix-steps", type=int, default=0)
    parser.add_argument("--approach-blend-mode", choices=("constant", "ramp"), default=None)
    parser.add_argument("--approach-commit-prefix-steps", type=int, default=-1)
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

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    demo = _load_demo_episode(args.task_suite_name, task.name, args.episode_idx)
    num_demo_steps = len(demo["ee_pos"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StarVLA DHB embodik stage-1 smoke.")

    model = build_framework(cfg)
    if args.checkpoint:
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
    approach_cfg = None
    if args.approach_blend_mode is not None or args.approach_commit_prefix_steps >= 0:
        approach_cfg = ChunkRolloutConfig(
            max_first_offset_m=chunk_cfg.max_first_offset_m,
            max_step_m=chunk_cfg.max_step_m,
            smooth_window=chunk_cfg.smooth_window,
            blend_overlap_alpha=chunk_cfg.blend_overlap_alpha,
            blend_mode=args.approach_blend_mode or chunk_cfg.blend_mode,
            commit_prefix_steps=max(
                0,
                args.approach_commit_prefix_steps
                if args.approach_commit_prefix_steps >= 0
                else chunk_cfg.commit_prefix_steps,
            ),
            gripper_source=chunk_cfg.gripper_source,
            gripper_threshold=chunk_cfg.gripper_threshold,
            gripper_close_threshold=chunk_cfg.gripper_close_threshold,
            gripper_open_threshold=chunk_cfg.gripper_open_threshold,
            gripper_transition_debounce_steps=chunk_cfg.gripper_transition_debounce_steps,
        )
    phase_runner = PhaseAwareChunkRunner(
        chunk_cfg,
        phase_configs={} if approach_cfg is None else {"approach": approach_cfg},
    )
    first_close_step = None
    if len(demo["gripper_cmd"]) > 1:
        close_edges = np.nonzero(
            (demo["gripper_cmd"][1:] > 0.0) & (demo["gripper_cmd"][:-1] <= 0.0)
        )[0]
        if len(close_edges):
            first_close_step = int(close_edges[0] + 1)
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
            result = model.predict_trajectory([example])

        n_use = min(args.replan_every, len(result["positions"][0]), num_demo_steps - t)
        active_phase = (
            "approach"
            if (approach_cfg is not None and first_close_step is not None and t < first_close_step)
            else "default"
        )
        prepared = phase_runner.prepare_chunk(
            result,
            anchor_pos=demo["ee_pos"][t],
            current_start=t,
            n_use=n_use,
            initial_gripper_cmd=last_gripper_cmd,
            demo_gripper_chunk=demo["gripper_cmd"][t : t + n_use],
            phase_name=active_phase,
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
                **prepared["debug"],
            }
        )
        t += n_use

    positions = np.concatenate(all_positions, axis=0)
    quaternions = np.concatenate(all_quaternions, axis=0)
    gripper = np.concatenate(all_gripper, axis=0)
    if args.gripper_source == "demo":
        assert len(gripper) == len(positions)

    save_npz = Path(args.save_npz) if args.save_npz else output_dir / "embodik_stage1_chunk.npz"
    np.savez(
        save_npz,
        positions=positions,
        quaternions=quaternions,
        gripper=gripper if gripper is not None else np.full(len(positions), -1.0, dtype=np.float32),
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
        "max_first_offset_mm": float(args.max_first_offset_mm),
        "max_step_mm": float(args.max_step_mm),
        "smooth_window": int(args.smooth_window),
        "blend_overlap_alpha": float(args.blend_overlap_alpha),
        "blend_mode": args.blend_mode,
        "commit_prefix_steps": int(args.commit_prefix_steps),
        "approach_blend_mode": args.approach_blend_mode,
        "approach_commit_prefix_steps": int(args.approach_commit_prefix_steps),
        "first_close_step": None if first_close_step is None else int(first_close_step),
        "num_demo_steps": num_demo_steps,
        "num_chunks": len(debug_chunks),
        "gripper_source": args.gripper_source,
        "gripper_threshold": float(args.gripper_threshold),
        "gripper_close_threshold": float(args.gripper_close_threshold),
        "gripper_open_threshold": float(args.gripper_open_threshold),
        "gripper_transition_debounce_steps": int(args.gripper_transition_debounce_steps),
        "state_history_window": int(state_history_window),
        "positions_shape": list(positions.shape),
        "quaternions_shape": list(quaternions.shape),
        "gripper_shape": list(gripper.shape) if gripper is not None else None,
        "gripper_preview": gripper[: min(len(gripper), 12)].tolist() if gripper is not None else None,
        "first_position": positions[0].tolist(),
        "first_quaternion_wxyz": quaternions[0].tolist(),
        "last_position": positions[-1].tolist(),
        "mean_step_pos_mm": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).mean() * 1000.0) if len(positions) > 1 else 0.0,
        "chunks": debug_chunks,
        "save_npz": str(save_npz),
    }
    save_json = Path(args.save_json) if args.save_json else output_dir / "embodik_stage1_summary.json"
    save_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
