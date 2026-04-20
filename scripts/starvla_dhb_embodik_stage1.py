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

from dhb_xr.integration.vla.chunk_rollout import ChunkRolloutConfig, TrajectoryChunkRunner
from dhb_xr.integration.vla.libero import (
    libero_obs_quat_to_axis_angle,
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
        missing = [k for k in missing if not k.startswith("qwen_vl_interface.")]
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


def _demo_state_vector(ee_pos: np.ndarray, ee_quat_wxyz: np.ndarray, grip_qpos: float) -> np.ndarray:
    axis_angle = libero_obs_quat_to_axis_angle(ee_quat_wxyz)
    return np.concatenate(
        (
            np.asarray(ee_pos, dtype=np.float32),
            axis_angle.astype(np.float32),
            np.array([0.0], dtype=np.float32),
            np.array([grip_qpos], dtype=np.float32),
        )
    ).astype(np.float32)


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
    parser.add_argument("--save_npz", default=None)
    parser.add_argument("--save_json", default=None)
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

    all_positions = []
    all_quaternions = []
    all_gripper = []
    debug_chunks = []
    chunk_cfg = ChunkRolloutConfig(
        max_first_offset_m=max(0.0, args.max_first_offset_mm) / 1000.0,
        max_step_m=max(0.0, args.max_step_mm) / 1000.0,
        smooth_window=max(0, args.smooth_window),
        blend_overlap_alpha=float(args.blend_overlap_alpha),
        gripper_source=args.gripper_source,
    )
    chunk_runner = TrajectoryChunkRunner(chunk_cfg)
    t = 0
    while t < num_demo_steps:
        state = _demo_state_vector(
            demo["ee_pos"][t],
            demo["ee_quat_wxyz"][t],
            float(demo["gripper_qpos"][t]),
        )
        example = {
            "image": [
                np.ascontiguousarray(demo["agentview_rgb"][t]),
                np.ascontiguousarray(demo["wrist_rgb"][t]),
            ],
            "lang": task.language,
            "state": np.expand_dims(state, axis=0),
        }

        with torch.inference_mode():
            result = model.predict_trajectory([example])

        n_use = min(args.replan_every, len(result["positions"][0]), num_demo_steps - t)
        prepared = chunk_runner.prepare_chunk(
            result,
            anchor_pos=demo["ee_pos"][t],
            current_start=t,
            n_use=n_use,
            initial_gripper_cmd=float(demo["gripper_cmd"][max(t - 1, 0)]),
            demo_gripper_chunk=demo["gripper_cmd"][t : t + n_use],
        )

        all_positions.append(prepared["positions"])
        all_quaternions.append(prepared["quaternions"])
        all_gripper.append(prepared["gripper_cmds"])

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
        "num_demo_steps": num_demo_steps,
        "num_chunks": len(debug_chunks),
        "gripper_source": args.gripper_source,
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
