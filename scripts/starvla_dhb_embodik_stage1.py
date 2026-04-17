from __future__ import annotations

import argparse
import functools
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import OmegaConf

from dhb_xr.integration.vla.libero import libero_obs_quat_to_axis_angle
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


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


def _binarize_gripper(open_val: float) -> float:
    return float(1.0 - 2.0 * (open_val > 0.5))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--checkpoint", default=None)
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
    with _libero_torch_load_compat():
        initial_states = task_suite.get_task_init_states(args.task_id)

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(cfg.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StarVLA DHB embodik stage-1 smoke.")

    model = build_framework(cfg)
    if args.checkpoint:
        _load_rollout_checkpoint(model, args.checkpoint)
    model.eval()

    obs = env.reset()
    obs = env.set_init_state(initial_states[args.episode_idx])
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            libero_obs_quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.array([0.0], dtype=np.float32),
            obs["robot0_gripper_qpos"][:1],
        )
    ).astype(np.float32)
    example = {
        "image": [
            np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        ],
        "lang": task.language,
        "state": np.expand_dims(state, axis=0),
    }

    with torch.inference_mode():
        result = model.predict_trajectory([example])

    positions = np.asarray(result["positions"][0], dtype=np.float32)
    quaternions = np.asarray(result["quaternions"][0], dtype=np.float32)  # wxyz
    normalized_actions = np.asarray(result.get("normalized_actions", []), dtype=np.float32)
    gripper = None
    if normalized_actions.size:
        gripper = np.array([_binarize_gripper(v) for v in normalized_actions[0, :, 6]], dtype=np.float32)

    save_npz = Path(args.save_npz) if args.save_npz else output_dir / "embodik_stage1_chunk.npz"
    np.savez(
        save_npz,
        positions=positions,
        quaternions=quaternions,
        gripper=gripper if gripper is not None else np.full(len(positions), -1.0, dtype=np.float32),
        init_state=initial_states[args.episode_idx],
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
        "positions_shape": list(positions.shape),
        "quaternions_shape": list(quaternions.shape),
        "first_position": positions[0].tolist(),
        "first_quaternion_wxyz": quaternions[0].tolist(),
        "save_npz": str(save_npz),
    }
    save_json = Path(args.save_json) if args.save_json else output_dir / "embodik_stage1_summary.json"
    save_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
