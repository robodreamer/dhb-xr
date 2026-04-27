from __future__ import annotations

import argparse
from collections import deque
import functools
import json
import random
from contextlib import contextmanager
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from omegaconf import OmegaConf

from dhb_xr.integration.vla.chunk_rollout import (
    ChunkRolloutConfig,
    PhaseAwareChunkRunner,
    compute_motion_metrics,
)
from dhb_xr.integration.vla.libero import (
    build_libero_state_vector_from_obs,
    build_state_history_window,
    get_state_history_window,
    integrate_delta_action_chunk,
    load_run_action_stats,
    unnormalize_actions,
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
LIBERO_COMMAND_OPEN_QPOS = 0.04
LIBERO_COMMAND_CLOSED_QPOS = 0.0
LIBERO_ENV_RESOLUTION = 256


class _NullWriter:
    def append_data(self, frame):
        return None

    def close(self):
        return None


def _seed_rollout(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _policy_state_with_gripper_source(
    obs: dict,
    *,
    gripper_cmd: float,
    gripper_source: str,
) -> np.ndarray:
    state = build_libero_state_vector_from_obs(obs)
    if gripper_source == "commanded":
        state = state.copy()
        state[-1] = (
            LIBERO_COMMAND_CLOSED_QPOS
            if float(gripper_cmd) > 0.0
            else LIBERO_COMMAND_OPEN_QPOS
        )
    return state


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


def _resolve_scene_ref_pos(env, ref, fallback):
    if env is None or not ref:
        return fallback
    try:
        kind, name = ref.split(":", 1)
        sim = env.env.sim
        model = sim.model
        data = sim.data
        if kind == "body":
            return np.array(data.body_xpos[model.body_name2id(name)])
        if kind == "site":
            return np.array(data.site_xpos[model.site_name2id(name)])
        if kind == "geom":
            return np.array(data.geom_xpos[model.geom_name2id(name)])
    except Exception:
        return fallback
    return fallback


def _relative_snapshot(ee_pos, refs, env=None):
    snap = {"ee_pos": np.asarray(ee_pos, dtype=np.float32).tolist()}
    bowl_pos = _resolve_scene_ref_pos(env, refs.get("bowl_ref"), refs.get("bowl_pos"))
    plate_pos = _resolve_scene_ref_pos(env, refs.get("plate_ref"), refs.get("plate_pos"))
    if bowl_pos is not None:
        snap["bowl_pos"] = np.asarray(bowl_pos, dtype=np.float32).tolist()
        snap["ee_minus_bowl"] = (np.asarray(ee_pos) - bowl_pos).tolist()
    if plate_pos is not None:
        snap["plate_pos"] = np.asarray(plate_pos, dtype=np.float32).tolist()
        snap["ee_minus_plate"] = (np.asarray(ee_pos) - plate_pos).tolist()
    if bowl_pos is not None and plate_pos is not None:
        snap["bowl_minus_plate"] = (np.asarray(bowl_pos) - np.asarray(plate_pos)).tolist()
    return snap


def _preprocess_libero_image(image: np.ndarray, orientation: str) -> np.ndarray:
    image = np.asarray(image, dtype=np.uint8)
    if orientation == "raw":
        return np.ascontiguousarray(image)
    if orientation == "flipudlr":
        return np.ascontiguousarray(image[::-1, ::-1])
    raise ValueError(f"Unsupported image orientation: {orientation}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--episode_idx", type=int, default=0)
    parser.add_argument("--offset-cm", type=float, default=0.0)
    parser.add_argument(
        "--rollout-seed",
        type=int,
        default=None,
        help=(
            "Evaluation-only seed for Python, NumPy, Torch, CUDA, and LIBERO. "
            "Defaults to the config seed when omitted."
        ),
    )
    parser.add_argument("--replan_every", type=int, default=4)
    parser.add_argument("--max-first-offset-mm", type=float, default=35.0)
    parser.add_argument("--max-step-mm", type=float, default=20.0)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--blend-overlap-alpha", type=float, default=0.6)
    parser.add_argument("--blend-mode", choices=("constant", "ramp"), default="constant")
    parser.add_argument("--commit-prefix-steps", type=int, default=0)
    parser.add_argument("--adaptive-commit-threshold-mm", type=float, default=0.0)
    parser.add_argument("--adaptive-commit-max-steps", type=int, default=0)
    parser.add_argument("--delta-continuation-alpha", type=float, default=0.0)
    parser.add_argument("--delta-continuation-steps", type=int, default=0)
    parser.add_argument("--approach-blend-mode", choices=("constant", "ramp"), default=None)
    parser.add_argument("--approach-commit-prefix-steps", type=int, default=-1)
    parser.add_argument("--approach-bowl-xy-m", type=float, default=0.10)
    parser.add_argument("--approach-bowl-z-m", type=float, default=0.10)
    parser.add_argument("--gripper-source", choices=("demo", "model"), default="demo")
    parser.add_argument(
        "--model-gripper-decode-source",
        choices=("auto", "gripper_state", "contact_phase", "contact", "contact_gripper_state", "normalized_action"),
        default="auto",
        help=(
            "Model-gripper decoder used when --gripper-source=model. "
            "auto preserves the existing preference order."
        ),
    )
    parser.add_argument("--init-state-source", choices=("demo", "benchmark"), default="demo")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.6)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.4)
    parser.add_argument("--gripper-transition-debounce-steps", type=int, default=1)
    parser.add_argument(
        "--image-orientation",
        choices=("raw", "flipudlr"),
        default="raw",
        help=(
            "Preprocessing for live LIBERO images before policy inference. "
            "Use raw for the local LeRobot export; flipudlr matches upstream "
            "StarVLA LIBERO examples trained with 180-degree-rotated images."
        ),
    )
    parser.add_argument(
        "--trajectory-source",
        choices=("dhb", "direct_action"),
        default="dhb",
        help=(
            "Use DHB-decoded absolute waypoints, or integrate the auxiliary "
            "normalized action head into EE waypoints for an execution-control ablation."
        ),
    )
    parser.add_argument(
        "--direct-action-xyz-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help=(
            "Global calibration applied to direct-action XYZ deltas before integration. "
            "This is task-agnostic and does not use object positions."
        ),
    )
    parser.add_argument(
        "--direct-action-xyz-bias-mm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("BX", "BY", "BZ"),
        help=(
            "Task-agnostic per-step XYZ delta bias in millimeters applied after "
            "--direct-action-xyz-scale."
        ),
    )
    parser.add_argument(
        "--closed-direct-action-xyz-scale",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("SX", "SY", "SZ"),
        help=(
            "Additional task-agnostic XYZ delta scale applied only while the "
            "executed gripper command is closed."
        ),
    )
    parser.add_argument(
        "--closed-direct-action-xyz-bias-mm",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("BX", "BY", "BZ"),
        help=(
            "Additional per-step XYZ delta bias in millimeters applied only "
            "while the executed gripper command is closed."
        ),
    )
    parser.add_argument("--close-gate-xy-m", type=float, default=0.05)
    parser.add_argument("--close-gate-z-m", type=float, default=0.05)
    parser.add_argument("--release-gate-xy-m", type=float, default=0.04)
    parser.add_argument("--release-gate-z-m", type=float, default=0.06)
    parser.add_argument("--disable-gripper-gates", action="store_true")
    parser.add_argument("--reach-only", action="store_true")
    parser.add_argument("--reach-threshold-xy-m", type=float, default=0.05)
    parser.add_argument("--reach-threshold-z-m", type=float, default=0.08)
    parser.add_argument("--reach-stop-on-success", action="store_true")
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=0,
        help=(
            "Closed-loop policy execution cap. The default 0 preserves the old "
            "demo-length cap; set this above the demo length to let offset-adapted "
            "rollouts continue after delayed reach/close events."
        ),
    )
    parser.add_argument("--output-max", type=float, default=0.5)
    parser.add_argument("--kp", type=float, default=150.0)
    parser.add_argument("--substeps", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument(
        "--record-step-trace",
        action="store_true",
        help=(
            "Store per-policy-step dynamic EE/object snapshots in the JSON summary. "
            "This is diagnostic instrumentation only and does not alter policy actions."
        ),
    )
    parser.add_argument(
        "--state-history-update-rate",
        choices=("policy", "substep"),
        default="policy",
        help=(
            "Timing contract for the short state history passed to the policy. "
            "'policy' appends one post-action observation per VLA action step; "
            "'substep' appends every low-level controller observation, matching "
            "a streaming deployment with faster proprioception than VLA replans."
        ),
    )
    parser.add_argument(
        "--state-gripper-source",
        choices=("observed", "commanded"),
        default="observed",
        help=(
            "Gripper component placed in the policy state token. 'observed' uses "
            "the simulator qpos; 'commanded' uses the executed gripper command so "
            "closed-phase heads are not delayed by gripper actuator settling."
        ),
    )
    parser.add_argument("--save_json", default=None)
    parser.add_argument("--save_video", default=None)
    parser.add_argument("--no-video", action="store_true")
    args, unknown = parser.parse_known_args()

    dotlist = normalize_dotlist_args(unknown)
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    cfg = wrap_config(cfg, cli_overrides=dotlist)
    rollout_seed = int(cfg.seed if args.rollout_seed is None else args.rollout_seed)
    _seed_rollout(rollout_seed)

    output_dir = _default_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "live_rollout_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    action_norm_stats = None
    if args.trajectory_source == "direct_action":
        action_norm_stats = load_run_action_stats(output_dir, checkpoint=args.checkpoint)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for StarVLA live embodik rollout.")

    model = build_framework(cfg)
    _load_rollout_checkpoint(model, args.checkpoint)
    model.eval()
    state_history_window = get_state_history_window(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    with _libero_torch_load_compat():
        initial_states = task_suite.get_task_init_states(args.task_id)

    demo = _load_demo_episode(args.task_suite_name, task.name, args.episode_idx)
    num_demo_steps = len(demo["ee_pos"])
    rollout_steps = int(args.max_policy_steps) if int(args.max_policy_steps) > 0 else num_demo_steps
    rollout_steps = max(1, rollout_steps)
    last_effective_grip = float(demo["gripper_cmd"][0]) if len(demo["gripper_cmd"]) else -1.0

    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        controller="JOINT_POSITION",
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(rollout_seed)
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
    state_history: deque[np.ndarray] = deque(
        [
            _policy_state_with_gripper_source(
                obs,
                gripper_cmd=last_effective_grip,
                gripper_source=args.state_gripper_source,
            )
        ],
        maxlen=state_history_window,
    )

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
        blend_mode=args.blend_mode,
        commit_prefix_steps=max(0, args.commit_prefix_steps),
        adaptive_commit_threshold_m=max(0.0, args.adaptive_commit_threshold_mm) / 1000.0,
        adaptive_commit_max_steps=max(0, args.adaptive_commit_max_steps),
        delta_continuation_alpha=float(args.delta_continuation_alpha),
        delta_continuation_steps=max(0, args.delta_continuation_steps),
        gripper_source=args.gripper_source,
        model_gripper_decode_source=args.model_gripper_decode_source,
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
            adaptive_commit_threshold_m=chunk_cfg.adaptive_commit_threshold_m,
            adaptive_commit_max_steps=chunk_cfg.adaptive_commit_max_steps,
            delta_continuation_alpha=chunk_cfg.delta_continuation_alpha,
            delta_continuation_steps=chunk_cfg.delta_continuation_steps,
            gripper_source=chunk_cfg.gripper_source,
            model_gripper_decode_source=chunk_cfg.model_gripper_decode_source,
            gripper_threshold=chunk_cfg.gripper_threshold,
            gripper_close_threshold=chunk_cfg.gripper_close_threshold,
            gripper_open_threshold=chunk_cfg.gripper_open_threshold,
            gripper_transition_debounce_steps=chunk_cfg.gripper_transition_debounce_steps,
        )
    phase_runner = PhaseAwareChunkRunner(
        chunk_cfg,
        phase_configs={} if approach_cfg is None else {"approach": approach_cfg},
    )
    debug_chunks = []
    close_step = None
    reopen_step = None
    close_snapshot = None
    reopen_snapshot = None
    reward = 0.0
    success_at = None
    ik_fail_total = 0
    first_ik_fail_chunk = None
    min_bowl_xy_m = float("inf")
    min_bowl_xyz_m = float("inf")
    min_plate_xy_m = float("inf")
    min_plate_xyz_m = float("inf")
    ee_trace = [np.asarray(robot.controller.ee_pos, dtype=np.float32).copy()]
    step_trace = []

    def reach_ready(ee_pos_xyz: np.ndarray) -> bool:
        rel = _relative_snapshot(ee_pos_xyz, scene_refs, env=env)
        if "ee_minus_bowl" not in rel:
            return False
        ee_minus_bowl = np.asarray(rel["ee_minus_bowl"], dtype=np.float32)
        return (
            float(np.linalg.norm(ee_minus_bowl[:2])) <= args.reach_threshold_xy_m
            and float(ee_minus_bowl[2]) <= args.reach_threshold_z_m
        )

    def update_min_dists(ee_pos_xyz: np.ndarray) -> None:
        nonlocal min_bowl_xy_m, min_bowl_xyz_m, min_plate_xy_m, min_plate_xyz_m
        rel = _relative_snapshot(ee_pos_xyz, scene_refs, env=env)
        if "ee_minus_bowl" in rel:
            ee_minus_bowl = np.asarray(rel["ee_minus_bowl"], dtype=np.float32)
            min_bowl_xy_m = min(min_bowl_xy_m, float(np.linalg.norm(ee_minus_bowl[:2])))
            min_bowl_xyz_m = min(min_bowl_xyz_m, float(np.linalg.norm(ee_minus_bowl)))
        if "ee_minus_plate" in rel:
            ee_minus_plate = np.asarray(rel["ee_minus_plate"], dtype=np.float32)
            min_plate_xy_m = min(min_plate_xy_m, float(np.linalg.norm(ee_minus_plate[:2])))
            min_plate_xyz_m = min(min_plate_xyz_m, float(np.linalg.norm(ee_minus_plate)))

    def record_step_trace(
        *,
        step: int,
        phase: str,
        desired_grip: float,
        effective_grip: float,
        reward_value: float,
        success_value: bool,
        ee_pos_xyz: np.ndarray,
        state_vector: np.ndarray | None = None,
        observed_state_vector: np.ndarray | None = None,
        close_ready: bool | None = None,
        release_ready: bool | None = None,
    ) -> None:
        if not args.record_step_trace:
            return
        entry = {
            "step": int(step),
            "phase": phase,
            "desired_grip": float(desired_grip),
            "effective_grip": float(effective_grip),
            "reward": float(reward_value),
            "success": bool(success_value),
            "close_ready": None if close_ready is None else bool(close_ready),
            "release_ready": None if release_ready is None else bool(release_ready),
            **_relative_snapshot(ee_pos_xyz, scene_refs, env=env),
        }
        if state_vector is not None and len(state_vector) > 0:
            entry["state_gripper"] = float(np.asarray(state_vector, dtype=np.float32)[-1])
        if observed_state_vector is not None and len(observed_state_vector) > 0:
            entry["observed_state_gripper"] = float(
                np.asarray(observed_state_vector, dtype=np.float32)[-1]
            )
        step_trace.append(entry)

    update_min_dists(np.asarray(robot.controller.ee_pos, dtype=np.float32))

    t = 0
    while t < rollout_steps:
        ee_pos_live = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        state = _policy_state_with_gripper_source(
            obs,
            gripper_cmd=last_effective_grip,
            gripper_source=args.state_gripper_source,
        )
        if not state_history or not np.allclose(state_history[-1], state):
            state_history.append(state.copy())
        state_window = build_state_history_window(
            list(state_history),
            history_window=state_history_window,
        )
        agentview = _preprocess_libero_image(obs["agentview_image"], args.image_orientation)
        wrist = _preprocess_libero_image(obs["robot0_eye_in_hand_image"], args.image_orientation)
        example = {
            "image": [agentview, wrist],
            "lang": task.language,
            "state": state_window,
        }

        with torch.inference_mode():
            result = model.predict_trajectory([example])

        if args.trajectory_source == "direct_action":
            normalized_actions = result.get("normalized_actions")
            if normalized_actions is None:
                raise RuntimeError(
                    "Requested --trajectory-source direct_action, but the model did not "
                    "return normalized_actions from predict_trajectory()."
                )
            raw_actions = unnormalize_actions(normalized_actions[0], action_norm_stats)
            xyz_scale = np.asarray(args.direct_action_xyz_scale, dtype=np.float32).reshape(1, 3)
            xyz_bias_m = (
                np.asarray(args.direct_action_xyz_bias_mm, dtype=np.float32).reshape(1, 3)
                / 1000.0
            )
            if not np.allclose(xyz_scale, 1.0) or not np.allclose(xyz_bias_m, 0.0):
                raw_actions = raw_actions.copy()
                raw_actions[:, :3] = raw_actions[:, :3] * xyz_scale + xyz_bias_m
            if last_effective_grip > 0.0:
                closed_xyz_scale = np.asarray(
                    args.closed_direct_action_xyz_scale,
                    dtype=np.float32,
                ).reshape(1, 3)
                closed_xyz_bias_m = (
                    np.asarray(args.closed_direct_action_xyz_bias_mm, dtype=np.float32).reshape(1, 3)
                    / 1000.0
                )
                if (
                    not np.allclose(closed_xyz_scale, 1.0)
                    or not np.allclose(closed_xyz_bias_m, 0.0)
                ):
                    raw_actions = raw_actions.copy()
                    raw_actions[:, :3] = raw_actions[:, :3] * closed_xyz_scale + closed_xyz_bias_m
            positions, quaternions = integrate_delta_action_chunk(
                start_pos=ee_pos_live,
                start_quat_wxyz=np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
                raw_actions=raw_actions,
            )
            result = dict(result)
            result["positions"] = positions[None]
            result["quaternions"] = quaternions[None]

        decoded_horizon = len(result["positions"][0])
        requested_horizon = min(args.replan_every, rollout_steps - t)
        if decoded_horizon < requested_horizon:
            raise RuntimeError(
                "Decoded DHB trajectory is shorter than the requested rollout chunk "
                f"({decoded_horizon} < {requested_horizon}). DHB-DR position decode "
                "drops two padded frames, so use action_horizon >= replan_every + 2 "
                "or lower --replan_every."
            )
        n_use = min(args.replan_every, decoded_horizon, rollout_steps - t)
        rel_live = _relative_snapshot(ee_pos_live, scene_refs, env=env)
        active_phase = "default"
        if approach_cfg is not None and "ee_minus_bowl" in rel_live:
            ee_minus_bowl_live = np.asarray(rel_live["ee_minus_bowl"], dtype=np.float32)
            if (
                float(np.linalg.norm(ee_minus_bowl_live[:2])) > args.approach_bowl_xy_m
                or float(ee_minus_bowl_live[2]) > args.approach_bowl_z_m
            ):
                active_phase = "approach"
        demo_gripper_chunk = None
        if t < num_demo_steps:
            demo_gripper_chunk = np.asarray(
                demo["gripper_cmd"][t : min(t + n_use, num_demo_steps)],
                dtype=np.float32,
            )
        if demo_gripper_chunk is None or len(demo_gripper_chunk) < n_use:
            pad_value = (
                float(demo_gripper_chunk[-1])
                if demo_gripper_chunk is not None and len(demo_gripper_chunk) > 0
                else float(last_effective_grip)
            )
            prefix = (
                np.asarray([], dtype=np.float32)
                if demo_gripper_chunk is None
                else demo_gripper_chunk.astype(np.float32)
            )
            demo_gripper_chunk = np.concatenate(
                [prefix, np.full((n_use - len(prefix),), pad_value, dtype=np.float32)]
            )
        prepared = phase_runner.prepare_chunk(
            result,
            anchor_pos=ee_pos_live,
            current_start=t,
            n_use=n_use,
            initial_gripper_cmd=last_effective_grip,
            demo_gripper_chunk=demo_gripper_chunk,
            phase_name=active_phase,
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

        gt_count = max(0, min(n_use, num_demo_steps - t))
        gt_chunk = demo["ee_pos"][t : t + gt_count] if gt_count > 0 else None
        if gt_count > 0:
            pos_err_mm = np.linalg.norm(chunk_pos_stable[:gt_count] - gt_chunk, axis=1) * 1000.0
            chunk_mean_err_mm = float(np.mean(pos_err_mm))
            chunk_max_err_mm = float(np.max(pos_err_mm))
            first_gt_pos = gt_chunk[0].tolist()
        else:
            chunk_mean_err_mm = None
            chunk_max_err_mm = None
            first_gt_pos = None
        debug_chunks.append(
            {
                "t_start": int(t),
                "n_use": int(n_use),
                "ik_failures": int(ik_failures),
                "chunk_mean_err_mm": chunk_mean_err_mm,
                "chunk_max_err_mm": chunk_max_err_mm,
                "first_live_ee_pos": ee_pos_live.tolist(),
                "first_pred_pos": prepared["first_pred_pos"],
                "first_stabilized_pos": prepared["first_stabilized_pos"],
                "first_gt_pos": first_gt_pos,
                "trajectory_source": args.trajectory_source,
                **gripper_debug,
            }
        )
        ik_fail_total += int(ik_failures)
        if ik_failures > 0 and first_ik_fail_chunk is None:
            first_ik_fail_chunk = len(debug_chunks) - 1

        for i in range(n_use):
            desired_grip = -1.0 if args.reach_only else float(gripper_cmds[i])
            ee_pos_pre = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            effective_grip = desired_grip
            rel_pre = _relative_snapshot(ee_pos_pre, scene_refs, env=env)
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

            # Keep LIBERO object-relative gating local to this evaluator instead
            # of baking bowl/plate semantics into the shared rollout module.
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
                ee_now = np.asarray(robot.controller.ee_pos, dtype=np.float32)
                ee_trace.append(ee_now.copy())
                update_min_dists(ee_now)
                if args.state_history_update_rate == "substep":
                    substep_state = _policy_state_with_gripper_source(
                        obs,
                        gripper_cmd=effective_grip,
                        gripper_source=args.state_gripper_source,
                    )
                    if not state_history or not np.allclose(state_history[-1], substep_state):
                        state_history.append(substep_state)

            observed_executed_state = build_libero_state_vector_from_obs(obs)
            executed_state = _policy_state_with_gripper_source(
                obs,
                gripper_cmd=effective_grip,
                gripper_source=args.state_gripper_source,
            )
            if (
                args.state_history_update_rate == "policy"
                and (not state_history or not np.allclose(state_history[-1], executed_state))
            ):
                state_history.append(executed_state)

            prev_grip = last_effective_grip
            last_effective_grip = float(effective_grip)
            ee_pos = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            if close_step is None and effective_grip > 0 and prev_grip <= 0:
                close_step = t + i
                close_snapshot = _relative_snapshot(ee_pos, scene_refs, env=env)
            if reopen_step is None and effective_grip <= 0 and prev_grip > 0:
                reopen_step = t + i
                reopen_snapshot = _relative_snapshot(ee_pos, scene_refs, env=env)

            success = env.check_success()
            if success and success_at is None:
                success_at = t + i
            if args.reach_only and success_at is None and reach_ready(ee_pos):
                success_at = t + i
            record_step_trace(
                step=t + i,
                phase=active_phase,
                desired_grip=desired_grip,
                effective_grip=effective_grip,
                reward_value=reward,
                success_value=success,
                ee_pos_xyz=ee_pos,
                state_vector=executed_state,
                observed_state_vector=observed_executed_state,
                close_ready=close_ready,
                release_ready=release_ready,
            )
            if args.reach_only and args.reach_stop_on_success and success_at is not None:
                break

        if args.reach_only and args.reach_stop_on_success and success_at is not None:
            break

        t += n_use

    if not (args.reach_only and args.reach_stop_on_success and success_at is not None):
        settle_grip = -1.0 if args.reach_only else last_effective_grip
        for i in range(args.settle_steps):
            action = np.zeros(7, dtype=np.float64)
            full_action = np.concatenate([action, [settle_grip]])
            obs, r, done, info = env.step(full_action)
            if r > reward:
                reward = float(r)
            ee_now = np.asarray(robot.controller.ee_pos, dtype=np.float32)
            ee_trace.append(ee_now.copy())
            success = env.check_success()
            if success and success_at is None:
                success_at = rollout_steps + i
            if args.reach_only and success_at is None and reach_ready(ee_now):
                success_at = rollout_steps + i
            observed_settle_state = build_libero_state_vector_from_obs(obs)
            settle_state = _policy_state_with_gripper_source(
                obs,
                gripper_cmd=settle_grip,
                gripper_source=args.state_gripper_source,
            )
            record_step_trace(
                step=rollout_steps + i,
                phase="settle",
                desired_grip=settle_grip,
                effective_grip=settle_grip,
                reward_value=reward,
                success_value=success,
                ee_pos_xyz=ee_now,
                state_vector=settle_state,
                observed_state_vector=observed_settle_state,
            )
            append_frame(writer, obs)

    final_success = bool(success_at is not None) if args.reach_only else env.check_success()
    ee_final = np.asarray(robot.controller.ee_pos, dtype=np.float32)
    motion_metrics = compute_motion_metrics(np.asarray(ee_trace, dtype=np.float32))
    final_snapshot = _relative_snapshot(ee_final, scene_refs, env=env)
    writer.close()
    env.close()

    summary = {
        "config_yaml": args.config_yaml,
        "checkpoint": args.checkpoint,
        "task_suite_name": args.task_suite_name,
        "task_id": int(args.task_id),
        "episode_idx": int(args.episode_idx),
        "offset_cm": float(args.offset_cm),
        "config_seed": int(cfg.seed),
        "rollout_seed": int(rollout_seed),
        "task_language": task.language,
        "num_demo_steps": int(num_demo_steps),
        "max_policy_steps": int(args.max_policy_steps),
        "rollout_steps": int(rollout_steps),
        "replan_every": int(args.replan_every),
        "max_first_offset_mm": float(args.max_first_offset_mm),
        "max_step_mm": float(args.max_step_mm),
        "smooth_window": int(args.smooth_window),
        "blend_overlap_alpha": float(args.blend_overlap_alpha),
        "blend_mode": args.blend_mode,
        "commit_prefix_steps": int(args.commit_prefix_steps),
        "adaptive_commit_threshold_mm": float(args.adaptive_commit_threshold_mm),
        "adaptive_commit_max_steps": int(args.adaptive_commit_max_steps),
        "delta_continuation_alpha": float(args.delta_continuation_alpha),
        "delta_continuation_steps": int(args.delta_continuation_steps),
        "approach_blend_mode": args.approach_blend_mode,
        "approach_commit_prefix_steps": int(args.approach_commit_prefix_steps),
        "approach_bowl_xy_m": float(args.approach_bowl_xy_m),
        "approach_bowl_z_m": float(args.approach_bowl_z_m),
        "state_history_window": int(state_history_window),
        "gripper_source": args.gripper_source,
        "model_gripper_decode_source": args.model_gripper_decode_source,
        "gripper_threshold": float(args.gripper_threshold),
        "gripper_close_threshold": float(args.gripper_close_threshold),
        "gripper_open_threshold": float(args.gripper_open_threshold),
        "gripper_transition_debounce_steps": int(args.gripper_transition_debounce_steps),
        "image_orientation": args.image_orientation,
        "trajectory_source": args.trajectory_source,
        "direct_action_xyz_scale": [float(v) for v in args.direct_action_xyz_scale],
        "direct_action_xyz_bias_mm": [float(v) for v in args.direct_action_xyz_bias_mm],
        "closed_direct_action_xyz_scale": [
            float(v) for v in args.closed_direct_action_xyz_scale
        ],
        "closed_direct_action_xyz_bias_mm": [
            float(v) for v in args.closed_direct_action_xyz_bias_mm
        ],
        "state_history_update_rate": args.state_history_update_rate,
        "state_gripper_source": args.state_gripper_source,
        "init_state_source": args.init_state_source,
        "init_state_l2_diff": float(np.linalg.norm(demo_init_state - benchmark_init_state)),
        "close_gate_xy_m": float(args.close_gate_xy_m),
        "close_gate_z_m": float(args.close_gate_z_m),
        "disable_gripper_gates": bool(args.disable_gripper_gates),
        "reach_only": bool(args.reach_only),
        "reach_threshold_xy_m": float(args.reach_threshold_xy_m),
        "reach_threshold_z_m": float(args.reach_threshold_z_m),
        "reach_stop_on_success": bool(args.reach_stop_on_success),
        "record_step_trace": bool(args.record_step_trace),
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
        "final_snapshot": final_snapshot,
        "min_bowl_xy_m": None if not np.isfinite(min_bowl_xy_m) else float(min_bowl_xy_m),
        "min_bowl_xyz_m": None if not np.isfinite(min_bowl_xyz_m) else float(min_bowl_xyz_m),
        "min_plate_xy_m": None if not np.isfinite(min_plate_xy_m) else float(min_plate_xy_m),
        "min_plate_xyz_m": None if not np.isfinite(min_plate_xyz_m) else float(min_plate_xyz_m),
        "scene_refs": {
            "bowl_ref": scene_refs.get("bowl_ref"),
            "plate_ref": scene_refs.get("plate_ref"),
            "initial_bowl_pos": None
            if scene_refs.get("bowl_pos") is None
            else np.asarray(scene_refs.get("bowl_pos"), dtype=np.float32).tolist(),
            "initial_plate_pos": None
            if scene_refs.get("plate_pos") is None
            else np.asarray(scene_refs.get("plate_pos"), dtype=np.float32).tolist(),
        },
        "motion_metrics": motion_metrics,
        "chunks": debug_chunks,
        "step_trace": step_trace if args.record_step_trace else None,
        "video_path": None if video_path is None else str(video_path),
        "overrides": dotlist,
    }
    save_json = Path(args.save_json) if args.save_json else output_dir / f"{label}_summary.json"
    save_json.write_text(json.dumps(summary, indent=2))
    printed_summary = dict(summary)
    printed_summary["chunks"] = f"<{len(debug_chunks)} chunks saved to {save_json}>"
    if args.record_step_trace:
        printed_summary["step_trace"] = f"<{len(step_trace)} entries saved to {save_json}>"
    print(json.dumps(printed_summary, indent=2))


if __name__ == "__main__":
    main()
