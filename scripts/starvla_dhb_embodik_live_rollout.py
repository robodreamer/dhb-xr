from __future__ import annotations

import argparse
import functools
import json
from contextlib import contextmanager
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import OmegaConf

from dhb_xr.integration.vla.chunk_rollout import ChunkRolloutConfig, TrajectoryChunkRunner
from dhb_xr.integration.vla.libero import (
    libero_obs_quat_to_axis_angle,
)
from starVLA.model.framework.base_framework import build_framework
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args

try:
    from robot_descriptions import panda_description
except ImportError:
    import sys

    _fallback_site = (
        Path("/home/andypark/Projects/repos/embodik_test/ik_venv/lib/python3.10/site-packages")
    )
    if _fallback_site.exists() and str(_fallback_site) not in sys.path:
        sys.path.insert(0, str(_fallback_site))
    from robot_descriptions import panda_description

try:
    import embodik
except ImportError:
    import sys

    _embodik_python = Path("/home/andypark/Projects/repos/embodik/python")
    if _embodik_python.exists() and str(_embodik_python) not in sys.path:
        sys.path.insert(0, str(_embodik_python))
    import embodik


LIBERO_DATA_ROOT = Path("/home/andypark/Projects/data/libero")
LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD = 0.015
LIBERO_ENV_RESOLUTION = 256


class _NullWriter:
    def append_data(self, frame):
        return None

    def close(self):
        return None


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
    run_id = getattr(cfg, "run_id", "dhb_starvla_embodik_live")
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


def _load_demo_episode(task_suite_name: str, task_name: str, episode_idx: int) -> dict:
    hdf5_path = _demo_hdf5_path(task_suite_name, task_name)
    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][f"demo_{episode_idx}"]
        robot_states = np.asarray(demo["robot_states"], dtype=np.float32)
        agentview = np.asarray(demo["obs"]["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(demo["obs"]["eye_in_hand_rgb"], dtype=np.uint8)
        init_state = np.asarray(demo["states"][0], dtype=np.float32)

    ee_pos = robot_states[:, 2:5].astype(np.float32)
    grip_qpos = robot_states[:, 0].astype(np.float32)
    gripper_cmd = np.where(grip_qpos > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD, -1.0, 1.0).astype(np.float32)

    return {
        "agentview_rgb": agentview,
        "wrist_rgb": wrist,
        "ee_pos": ee_pos,
        "gripper_qpos": grip_qpos,
        "gripper_cmd": gripper_cmd,
        "init_state": init_state,
    }


def _state_vector_from_obs(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            libero_obs_quat_to_axis_angle(obs["robot0_eef_quat"]).astype(np.float32),
            np.array([0.0], dtype=np.float32),
            np.asarray(obs["robot0_gripper_qpos"][:1], dtype=np.float32),
        )
    ).astype(np.float32)


def _make_video_path(video_dir: Path, label: str, suffix: str = "mp4") -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label).strip("_")
    return video_dir / f"{safe}.{suffix}"


def _capture_scene_refs(env):
    sim = env.env.sim

    def _matches(name, tokens):
        name = name.lower()
        return all(tok in name for tok in tokens)

    def _find(body_tokens=(), site_tokens=(), geom_tokens=()):
        model = sim.model
        data = sim.data
        for body_id in range(model.nbody):
            body_name = model.body_id2name(body_id)
            if body_name and _matches(body_name, body_tokens):
                return np.array(data.body_xpos[body_id]), f"body:{body_name}"
        for site_id in range(model.nsite):
            site_name = model.site_id2name(site_id)
            if site_name and _matches(site_name, site_tokens):
                return np.array(data.site_xpos[site_id]), f"site:{site_name}"
        for geom_id in range(model.ngeom):
            geom_name = model.geom_id2name(geom_id)
            if geom_name and _matches(geom_name, geom_tokens):
                return np.array(data.geom_xpos[geom_id]), f"geom:{geom_name}"
        return None, None

    bowl_pos, bowl_ref = _find(body_tokens=("bowl",), site_tokens=("bowl",), geom_tokens=("bowl",))
    plate_pos, plate_ref = _find(body_tokens=("plate",), site_tokens=("plate",), geom_tokens=("plate",))
    return {
        "bowl_pos": bowl_pos,
        "bowl_ref": bowl_ref,
        "plate_pos": plate_pos,
        "plate_ref": plate_ref,
    }


def _relative_snapshot(ee_pos, refs):
    snap = {"ee_pos": np.asarray(ee_pos, dtype=np.float32).tolist()}
    bowl_pos = refs.get("bowl_pos")
    plate_pos = refs.get("plate_pos")
    if bowl_pos is not None:
        snap["ee_minus_bowl"] = (np.asarray(ee_pos) - bowl_pos).tolist()
    if plate_pos is not None:
        snap["ee_minus_plate"] = (np.asarray(ee_pos) - plate_pos).tolist()
    return snap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--offset-cm", type=float, default=0.0)
    parser.add_argument("--replan_every", type=int, default=4)
    parser.add_argument("--max-first-offset-mm", type=float, default=35.0)
    parser.add_argument("--max-step-mm", type=float, default=20.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--blend-overlap-alpha", type=float, default=0.6)
    parser.add_argument("--gripper-source", choices=("demo", "model"), default="demo")
    parser.add_argument("--init-state-source", choices=("demo", "benchmark"), default="demo")
    parser.add_argument("--close-gate-xy-m", type=float, default=0.05)
    parser.add_argument("--close-gate-z-m", type=float, default=0.05)
    parser.add_argument("--release-gate-xy-m", type=float, default=0.04)
    parser.add_argument("--release-gate-z-m", type=float, default=0.06)
    parser.add_argument("--disable-gripper-gates", action="store_true")
    parser.add_argument("--output-max", type=float, default=0.5)
    parser.add_argument("--kp", type=float, default=150.0)
    parser.add_argument("--substeps", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--save_video", default=None)
    parser.add_argument("--no-video", action="store_true")
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)

    output_dir = _default_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "live_rollout_videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StarVLA live embodik rollout.")

    model = build_framework(cfg)
    _load_rollout_checkpoint(model, args.checkpoint)
    model.eval()

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    with _libero_torch_load_compat():
        initial_states = task_suite.get_task_init_states(args.task_id)

    demo = _load_demo_episode(args.task_suite_name, task.name, args.episode_idx)
    num_demo_steps = len(demo["ee_pos"])

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        controller="JOINT_POSITION",
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(cfg.seed)
    obs = env.reset()
    benchmark_init_state = np.asarray(initial_states[args.episode_idx], dtype=np.float32).copy()
    demo_init_state = np.asarray(demo["init_state"], dtype=np.float32).copy()
    if args.init_state_source == "demo":
        init_state = demo_init_state.copy()
    else:
        init_state = benchmark_init_state.copy()
    if args.offset_cm != 0:
        init_state[11] += args.offset_cm / 100.0
    env.set_init_state(init_state)
    obs = env.env._get_observations()

    robot = env.env.robots[0]
    ctrl = robot.controller
    ctrl.output_max = ctrl.nums2array(args.output_max, len(ctrl.qpos_index))
    ctrl.output_min = ctrl.nums2array(-args.output_max, len(ctrl.qpos_index))
    ctrl.kp = ctrl.nums2array(args.kp, len(ctrl.qpos_index))
    ctrl.kd = 2 * np.sqrt(ctrl.kp) * 1.0
    ctrl.action_scale = None

    scene_refs = _capture_scene_refs(env)

    import embodik

    robot_model = embodik.RobotModel(urdf_path=panda_description.URDF_PATH)
    solver = embodik.KinematicsSolver(robot_model)
    ik_opts = embodik.PositionIKOptions()
    ik_opts.position_tolerance = 0.0005
    ik_opts.orientation_tolerance = 100.0
    ik_opts.orientation_gain = 0.1
    ik_opts.max_iterations = 200

    def append_frame(writer, observation):
        frame = observation.get("agentview_image")
        if frame is None:
            return
        writer.append_data(np.flipud(np.asarray(frame, dtype=np.uint8)))

    label = (
        f"live_offset_{args.offset_cm:.1f}cm_ep_{args.episode_idx}"
        f"_replan{args.replan_every}_clamp{int(args.max_first_offset_mm)}"
    )
    video_path = None if args.no_video else (
        Path(args.save_video) if args.save_video else _make_video_path(video_dir, label)
    )
    writer = _NullWriter() if args.no_video else imageio.get_writer(video_path, fps=10)
    append_frame(writer, obs)

    q_current = np.zeros(robot_model.nq, dtype=np.float64)
    q_current[:7] = np.asarray(robot._joint_positions, dtype=np.float64)
    robot_model.update_configuration(q_current)
    base_offset = np.asarray(obs["robot0_eef_pos"], dtype=np.float64) - np.asarray(
        robot_model.get_frame_pose("panda_hand_tcp").translation, dtype=np.float64
    )

    chunk_cfg = ChunkRolloutConfig(
        max_first_offset_m=max(0.0, args.max_first_offset_mm) / 1000.0,
        max_step_m=max(0.0, args.max_step_mm) / 1000.0,
        smooth_window=max(0, args.smooth_window),
        blend_overlap_alpha=float(args.blend_overlap_alpha),
        gripper_source=args.gripper_source,
    )
    chunk_runner = TrajectoryChunkRunner(chunk_cfg)
    debug_chunks = []
    close_step = None
    reopen_step = None
    close_snapshot = None
    reopen_snapshot = None
    last_effective_grip = float(demo["gripper_cmd"][0]) if len(demo["gripper_cmd"]) else -1.0
    reward = 0.0
    success_at = None
    ik_fail_total = 0
    first_ik_fail_chunk = None
    min_bowl_xy_m = float("inf")
    min_bowl_xyz_m = float("inf")
    min_plate_xy_m = float("inf")
    min_plate_xyz_m = float("inf")

    def update_min_dists(ee_pos_xyz: np.ndarray) -> None:
        nonlocal min_bowl_xy_m, min_bowl_xyz_m, min_plate_xy_m, min_plate_xyz_m
        rel = _relative_snapshot(ee_pos_xyz, scene_refs)
        if "ee_minus_bowl" in rel:
            ee_minus_bowl = np.asarray(rel["ee_minus_bowl"], dtype=np.float32)
            min_bowl_xy_m = min(min_bowl_xy_m, float(np.linalg.norm(ee_minus_bowl[:2])))
            min_bowl_xyz_m = min(min_bowl_xyz_m, float(np.linalg.norm(ee_minus_bowl)))
        if "ee_minus_plate" in rel:
            ee_minus_plate = np.asarray(rel["ee_minus_plate"], dtype=np.float32)
            min_plate_xy_m = min(min_plate_xy_m, float(np.linalg.norm(ee_minus_plate[:2])))
            min_plate_xyz_m = min(min_plate_xyz_m, float(np.linalg.norm(ee_minus_plate)))

    update_min_dists(np.asarray(robot.controller.ee_pos, dtype=np.float32))

    t = 0
    while t < num_demo_steps:
        ee_pos_live = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        state = _state_vector_from_obs(obs)
        agentview = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        example = {
            "image": [agentview, wrist],
            "lang": task.language,
            "state": np.expand_dims(state, axis=0),
        }

        with torch.inference_mode():
            result = model.predict_trajectory([example])

        n_use = min(args.replan_every, len(result["positions"][0]), num_demo_steps - t)
        prepared = chunk_runner.prepare_chunk(
            result,
            anchor_pos=ee_pos_live,
            current_start=t,
            n_use=n_use,
            initial_gripper_cmd=last_effective_grip,
            demo_gripper_chunk=demo["gripper_cmd"][t : t + n_use],
        )
        chunk_pos_stable = prepared["positions_full"]
        q_chunk = np.zeros((n_use, robot_model.nq), dtype=np.float64)
        ik_failures = 0
        q_seed = np.zeros(robot_model.nq, dtype=np.float64)
        q_seed[:7] = np.asarray(robot._joint_positions, dtype=np.float64)

        for i in range(n_use):
            robot_model.update_configuration(q_seed)
            current_rot = np.asarray(robot_model.get_frame_pose("panda_hand_tcp").rotation, dtype=np.float64)
            target_T = np.eye(4, order="F")
            target_T[:3, :3] = current_rot
            target_T[:3, 3] = chunk_pos_stable[i].astype(np.float64) - base_offset

            result_ik = solver.solve_position(q_seed, target_T, "panda_hand_tcp", ik_opts)
            if result_ik.status == embodik.SolverStatus.SUCCESS:
                q_seed = np.asarray(result_ik.q_solution, dtype=np.float64)
            else:
                ik_failures += 1
            q_chunk[i] = q_seed.copy()

        gripper_cmds = prepared["gripper_cmds"]
        gripper_debug = prepared["debug"]

        gt_chunk = demo["ee_pos"][t : t + n_use]
        pos_err_mm = np.linalg.norm(chunk_pos_stable[:n_use] - gt_chunk, axis=1) * 1000.0
        debug_chunks.append(
            {
                "t_start": int(t),
                "n_use": int(n_use),
                "ik_failures": int(ik_failures),
                "chunk_mean_err_mm": float(np.mean(pos_err_mm)),
                "chunk_max_err_mm": float(np.max(pos_err_mm)),
                "first_live_ee_pos": ee_pos_live.tolist(),
                "first_pred_pos": prepared["first_pred_pos"],
                "first_stabilized_pos": prepared["first_stabilized_pos"],
                "first_gt_pos": gt_chunk[0].tolist(),
                **gripper_debug,
            }
        )
        ik_fail_total += int(ik_failures)
        if ik_failures > 0 and first_ik_fail_chunk is None:
            first_ik_fail_chunk = len(debug_chunks) - 1

        for i in range(n_use):
            desired_grip = float(gripper_cmds[i])
            ee_pos_pre = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            effective_grip = desired_grip
            rel_pre = _relative_snapshot(ee_pos_pre, scene_refs)
            ee_minus_bowl = np.asarray(rel_pre.get("ee_minus_bowl", [np.inf, np.inf, np.inf]), dtype=np.float32)
            ee_minus_plate = np.asarray(rel_pre.get("ee_minus_plate", [np.inf, np.inf, np.inf]), dtype=np.float32)
            close_ready = (
                abs(float(ee_minus_bowl[0])) <= args.close_gate_xy_m
                and abs(float(ee_minus_bowl[1])) <= args.close_gate_xy_m
                and float(ee_minus_bowl[2]) <= args.close_gate_z_m
            )
            release_ready = (
                abs(float(ee_minus_plate[0])) <= args.release_gate_xy_m
                and abs(float(ee_minus_plate[1])) <= args.release_gate_xy_m
                and float(ee_minus_plate[2]) <= args.release_gate_z_m
            )

            # Geometry vetoes are useful for replayed demo timing, but they
            # should be disable-able once the gripper token is retrained from
            # geometry-conditioned labels and is meant to act directly from
            # perception, state, and the current DHB plan.
            if not args.disable_gripper_gates:
                if desired_grip > 0 and last_effective_grip <= 0:
                    if not close_ready:
                        effective_grip = -1.0
                if desired_grip <= 0 and last_effective_grip > 0:
                    if not release_ready:
                        effective_grip = 1.0

            for _ in range(args.substeps):
                q_target = q_chunk[i, :7]
                q_now = np.asarray(robot._joint_positions, dtype=np.float64)
                delta = q_target - q_now
                action = np.clip(delta / args.output_max, -1.0, 1.0)
                full_action = np.concatenate([action, [effective_grip]])
                obs, r, done, info = env.step(full_action)
                if r > reward:
                    reward = float(r)
                append_frame(writer, obs)
                update_min_dists(np.asarray(robot.controller.ee_pos, dtype=np.float32))

            prev_grip = last_effective_grip
            last_effective_grip = float(effective_grip)
            ee_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            if close_step is None and effective_grip > 0 and prev_grip <= 0:
                close_step = t + i
                close_snapshot = _relative_snapshot(ee_pos, scene_refs)
            if reopen_step is None and effective_grip <= 0 and prev_grip > 0:
                reopen_step = t + i
                reopen_snapshot = _relative_snapshot(ee_pos, scene_refs)

            success = env.check_success()
            if success and success_at is None:
                success_at = t + i

        t += n_use

    settle_grip = last_effective_grip
    for i in range(args.settle_steps):
        action = np.zeros(7, dtype=np.float64)
        full_action = np.concatenate([action, [settle_grip]])
        obs, r, done, info = env.step(full_action)
        if r > reward:
            reward = float(r)
        success = env.check_success()
        if success and success_at is None:
            success_at = num_demo_steps + i
        append_frame(writer, obs)

    final_success = env.check_success()
    ee_final = np.asarray(robot.controller.ee_pos, dtype=np.float32)
    writer.close()
    env.close()

    summary = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "task_suite_name": args.task_suite_name,
        "task_id": int(args.task_id),
        "episode_idx": int(args.episode_idx),
        "offset_cm": float(args.offset_cm),
        "task_language": task.language,
        "replan_every": int(args.replan_every),
        "max_first_offset_mm": float(args.max_first_offset_mm),
        "max_step_mm": float(args.max_step_mm),
        "smooth_window": int(args.smooth_window),
        "blend_overlap_alpha": float(args.blend_overlap_alpha),
        "gripper_source": args.gripper_source,
        "init_state_source": args.init_state_source,
        "init_state_l2_diff": float(np.linalg.norm(demo_init_state - benchmark_init_state)),
        "close_gate_xy_m": float(args.close_gate_xy_m),
        "close_gate_z_m": float(args.close_gate_z_m),
        "disable_gripper_gates": bool(args.disable_gripper_gates),
        "success": bool(final_success),
        "success_at": None if success_at is None else int(success_at),
        "reward": float(reward),
        "ik_fail_total": int(ik_fail_total),
        "first_ik_fail_chunk": None if first_ik_fail_chunk is None else int(first_ik_fail_chunk),
        "num_chunks": int(len(debug_chunks)),
        "close_step": None if close_step is None else int(close_step),
        "reopen_step": None if reopen_step is None else int(reopen_step),
        "close_snapshot": close_snapshot,
        "reopen_snapshot": reopen_snapshot,
        "final_snapshot": _relative_snapshot(ee_final, scene_refs),
        "min_bowl_xy_m": None if not np.isfinite(min_bowl_xy_m) else float(min_bowl_xy_m),
        "min_bowl_xyz_m": None if not np.isfinite(min_bowl_xyz_m) else float(min_bowl_xyz_m),
        "min_plate_xy_m": None if not np.isfinite(min_plate_xy_m) else float(min_plate_xy_m),
        "min_plate_xyz_m": None if not np.isfinite(min_plate_xyz_m) else float(min_plate_xyz_m),
        "scene_refs": {
            "bowl_ref": scene_refs.get("bowl_ref"),
            "plate_ref": scene_refs.get("plate_ref"),
        },
        "chunks": debug_chunks,
        "video_path": None if video_path is None else str(video_path),
        "overrides": dotlist,
    }
    save_json = Path(args.save_json) if args.save_json else output_dir / f"{label}_summary.json"
    save_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
