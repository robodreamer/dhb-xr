from __future__ import annotations

import argparse
import functools
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import h5py
import imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from dhb_xr.integration.vla.libero import libero_obs_quat_to_axis_angle
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@contextmanager
def _libero_torch_load_compat():
    # LIBERO init-state files were serialized with numpy objects.
    # PyTorch >=2.6 defaults torch.load(..., weights_only=True), which rejects them.
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
    run_id = getattr(cfg, "run_id", "dhb_libero_rollout_smoke")
    return run_root_dir / run_id


def _checkpoint_tag(checkpoint: str | None) -> str:
    if not checkpoint:
        return "current"
    ckpt = Path(checkpoint)
    tag = ckpt.stem.replace("_pytorch_model", "")
    return tag.replace(" ", "_")


def _load_action_norm_stats(cfg, output_dir: Path) -> dict[str, Any]:
    stats_path = output_dir / "dataset_statistics.json"
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Missing dataset statistics at {stats_path}. Run a StarVLA dataloader smoke first to materialize stats."
        )
    payload = json.loads(stats_path.read_text())
    if not payload:
        raise ValueError(f"Dataset statistics file is empty: {stats_path}")
    first_key = next(iter(payload.keys()))
    action_stats = payload[first_key]["action"]
    required = {"min", "max"}
    missing = required - set(action_stats.keys())
    if missing:
        raise KeyError(f"Dataset action stats missing required keys {sorted(missing)} in {stats_path}")
    return action_stats


def _unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: dict[str, Any]) -> np.ndarray:
    normalized_actions = np.asarray(normalized_actions, dtype=np.float32).copy()
    mask = np.asarray(action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool)), dtype=bool)
    action_high = np.asarray(action_norm_stats["max"], dtype=np.float32)
    action_low = np.asarray(action_norm_stats["min"], dtype=np.float32)
    normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
    normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0.0, 1.0)
    return np.where(
        mask,
        0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low,
        normalized_actions,
    )


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

    # Quantized Qwen backbones carry bitsandbytes-specific buffers (e.g. SCB)
    # that do not round-trip cleanly through a plain state_dict reload here.
    # During these laptop smokes the Qwen backbone is frozen, so we only need
    # to restore the trained DHB-side modules.
    if _uses_quantized_vlm(model):
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("qwen_vl_interface.")}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        missing = [k for k in missing if not k.startswith("qwen_vl_interface.")]
        if missing or unexpected:
            raise RuntimeError(
                "Quantized rollout checkpoint load mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return

    model.load_state_dict(state_dict, strict=True)


def _binarize_gripper(open_val: float) -> float:
    return float(1.0 - 2.0 * (open_val > 0.5))


def _euler_to_axisangle(euler_xyz: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", np.asarray(euler_xyz, dtype=np.float32), degrees=False).as_rotvec().astype(
        np.float32
    )


def _max_steps_for_suite(name: str) -> int:
    return {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }[name]


def _load_libero_demo_absolute_actions(hdf5_path: str, demo_id: int) -> np.ndarray:
    demo_key = f"demo_{demo_id}"
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        robot_states = np.asarray(demo["robot_states"], dtype=np.float64)
        ee_pos = robot_states[:, 2:5].astype(np.float32)
        ee_quat_wxyz = robot_states[:, 5:9].astype(np.float64)
        ee_quat_xyzw = ee_quat_wxyz[:, [1, 2, 3, 0]]
        ee_quat_xyzw = ee_quat_xyzw / np.linalg.norm(ee_quat_xyzw, axis=1, keepdims=True)
        ee_euler = Rotation.from_quat(ee_quat_xyzw).as_euler("xyz", degrees=False).astype(np.float32)
        gripper = robot_states[:, 0:1].astype(np.float32)
    return np.concatenate([ee_pos, ee_euler, gripper], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--gt_hdf5", default=None)
    parser.add_argument("--gt_demo_id", type=int, default=0)
    parser.add_argument("--save_json", default=None)
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)

    cfg.output_dir = str(_default_output_dir(cfg))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_norm_stats = None if args.gt_hdf5 else _load_action_norm_stats(cfg, output_dir)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    with _libero_torch_load_compat():
        initial_states = task_suite.get_task_init_states(args.task_id)

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(cfg.seed)

    model = None
    device = None
    raw_actions_gt = _load_libero_demo_absolute_actions(args.gt_hdf5, args.gt_demo_id) if args.gt_hdf5 else None
    if raw_actions_gt is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for StarVLA LIBERO rollout smoke.")
        device = torch.device("cuda")
        model = build_framework(cfg)
        if args.checkpoint:
            _load_rollout_checkpoint(model, args.checkpoint)
        model.eval()
        if not _uses_quantized_vlm(model):
            model = model.to(device)

    obs = env.reset()
    obs = env.set_init_state(initial_states[args.episode_idx])
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    for robot in env.robots:
        robot.controller.use_delta = False

    max_steps = _max_steps_for_suite(args.task_suite_name)
    replay_images = []
    actions = []
    raw_actions_chunk = None
    done = False

    if device is not None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    for _step in range(max_steps):
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        replay_images.append(img)

        state = np.concatenate(
            (
                obs["robot0_eef_pos"],
                libero_obs_quat_to_axis_angle(obs["robot0_eef_quat"]),
                np.array([0.0], dtype=np.float32),  # pad channel to match 8D training state
                obs["robot0_gripper_qpos"][:1],
            )
        ).astype(np.float32)

        if raw_actions_gt is None:
            example = {
                "image": [img, wrist_img],
                "lang": task.language,
                "state": np.expand_dims(state, axis=0),
            }

            if raw_actions_chunk is None or _step % len(raw_actions_chunk) == 0:
                with torch.inference_mode():
                    pred = model.predict_action([example])
                normalized_chunk = pred["normalized_actions"][0]
                raw_actions_chunk = _unnormalize_actions(normalized_chunk, action_norm_stats=action_norm_stats)
            raw_action = raw_actions_chunk[_step % len(raw_actions_chunk)]
        else:
            if _step >= len(raw_actions_gt):
                break
            raw_action = raw_actions_gt[_step]

        delta_action = np.concatenate(
            [
                np.asarray(raw_action[:3], dtype=np.float32),
                _euler_to_axisangle(np.asarray(raw_action[3:6], dtype=np.float32)),
                np.asarray([_binarize_gripper(float(raw_action[6]))], dtype=np.float32),
            ]
        )
        actions.append(delta_action.tolist())

        obs, _, done, _info = env.step(delta_action.tolist())
        if done:
            break

    if device is not None:
        end.record()
        torch.cuda.synchronize(device)

    task_segment = task.language.replace(" ", "_")
    checkpoint_tag = f"gt_demo{args.gt_demo_id}" if args.gt_hdf5 else _checkpoint_tag(args.checkpoint)
    suffix = "success" if done else "failure"
    video_path = output_dir / f"rollout_{checkpoint_tag}_{task_segment}_episode{args.episode_idx}_{suffix}.mp4"
    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

    summary = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "gt_hdf5": args.gt_hdf5,
        "gt_demo_id": args.gt_demo_id if args.gt_hdf5 else None,
        "overrides": dotlist,
        "output_dir": str(output_dir),
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "episode_idx": args.episode_idx,
        "task_language": task.language,
        "success": bool(done),
        "num_steps": len(actions),
        "rollout_time_ms": round(start.elapsed_time(end), 2) if device is not None else None,
        "gpu_name": torch.cuda.get_device_name(device) if device is not None else None,
        "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2) if device is not None else None,
        "video_path": str(video_path),
        "first_action": actions[0] if actions else None,
        "last_action": actions[-1] if actions else None,
    }

    save_json = Path(args.save_json) if args.save_json else output_dir / "libero_rollout_smoke_summary.json"
    save_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    env.close()


if __name__ == "__main__":
    main()
