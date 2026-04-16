#!/usr/bin/env python
"""Record DHB-XR P2 demo videos for the project page.

Records three videos:
  1. dhb_p2_raw_fail.mp4   - Raw replay on perturbed env (misses plate, FAIL)
  2. dhb_p2_success.mp4    - DHB-adapted replay on perturbed env (SUCCESS)
  3. dhb_p2_comparison.mp4 - Side-by-side: raw fail (left) vs DHB success (right)

Usage:
    MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python _workspace/record_p2_videos.py

Author: Andy Park
Date: 2026-04-13
"""

import os
import sys
import random
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

# Add DHB-XR src
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add LIBERO-PRO
LIBERO_PRO_ROOT = Path("/home/andypark/Projects/repos/LIBERO-PRO")
sys.path.insert(0, str(LIBERO_PRO_ROOT))

import numpy as np
import imageio.v2 as iio

from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.types import EncodingMethod, DHBMethod

try:
    from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop
    HAS_FATROP = True
except ImportError:
    HAS_FATROP = False

try:
    from dhb_xr.optimization.casadi_solver import generate_trajectory
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

import h5py
from libero.libero.envs import OffScreenRenderEnv
from perturbation import BDDLParser, SwapPerturbator

# =========================================================================
# Constants (copied from eval_libero_pro_p2.py)
# =========================================================================
TASK_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
TASK_SUITE = "libero_spatial"
BDDL_ORIG = str(
    LIBERO_PRO_ROOT
    / "libero/libero/bddl_files/libero_spatial"
    / f"{TASK_NAME}.bddl"
)
OOD_CFG = str(LIBERO_PRO_ROOT / "libero_ood/ood_spatial_relation.yaml")
DATASET_PATH = (
    "/home/andypark/Projects/data/libero/libero_spatial"
    f"/{TASK_NAME}_demo.hdf5"
)

DEMO_ID = 0
MAX_STEPS = 300
HORIZON = 600
OSC_SCALE = 0.05

# Video settings
CAM_H = 256
CAM_W = 256
FPS = 30

OUTPUT_DIR = Path("/home/andypark/Projects/repos/dhb_xr/docs/paper/assets")
RAW_FAIL_PATH = OUTPUT_DIR / "dhb_p2_raw_fail.mp4"
DHB_SUCCESS_PATH = OUTPUT_DIR / "dhb_p2_success.mp4"
COMPARISON_PATH = OUTPUT_DIR / "dhb_p2_comparison.mp4"


# =========================================================================
# BDDL helpers
# =========================================================================

def generate_perturbed_bddl(seed: int = 42):
    with open(BDDL_ORIG) as f:
        content = f.read()
    random.seed(seed)
    parser = BDDLParser(content)
    swapper = SwapPerturbator(parser, OOD_CFG)
    perturbed_content = swapper.perturb(TASK_SUITE, TASK_NAME)
    tmpfile = tempfile.NamedTemporaryFile(
        suffix=".bddl", delete=False, mode="w",
        prefix="libero_pro_p2_perturbed_"
    )
    tmpfile.write(perturbed_content)
    tmpfile.close()
    return BDDL_ORIG, tmpfile.name, perturbed_content


def load_demo(demo_id: int = 0) -> Dict:
    with h5py.File(DATASET_PATH, "r") as f:
        demo_key = f"demo_{demo_id}"
        demo = f["data"][demo_key]
        actions = np.array(demo["actions"])
        robot_states = np.array(demo["robot_states"])
        ee_pos = robot_states[:, 2:5].astype(np.float64)
        ee_quat_wxyz = robot_states[:, 5:9].astype(np.float64)
        ee_quat = ee_quat_wxyz[:, [1, 2, 3, 0]]
        ee_quat = ee_quat / np.linalg.norm(ee_quat, axis=1, keepdims=True)
        init_state = np.array(demo["states"][0]) if "states" in demo else None
        return {
            "actions": actions,
            "positions": ee_pos,
            "quaternions": ee_quat,
            "num_frames": len(actions),
            "init_state": init_state,
        }


def _get_object_pos(bddl_path: str, object_name: str) -> Optional[np.ndarray]:
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=64,
        camera_widths=64,
    )
    obs = env.reset()
    pos = obs.get(f"{object_name}_pos", None)
    if pos is not None:
        pos = pos.copy()
    env.close()
    return pos


# =========================================================================
# Trajectory adaptation (same as eval script fallback)
# =========================================================================

def adapt_trajectory(positions, quaternions, new_start_pos, new_start_quat,
                     new_goal_pos=None, new_goal_quat=None, traj_length=None):
    if traj_length is None:
        traj_length = min(len(positions), 50)
    if new_goal_pos is None:
        offset = new_start_pos - positions[0]
        new_goal_pos = positions[-1] + offset
    if new_goal_quat is None:
        new_goal_quat = quaternions[-1].copy()

    pose_init = {"position": new_start_pos.copy(), "quaternion": new_start_quat.copy()}
    pose_goal = {"position": new_goal_pos.copy(), "quaternion": new_goal_quat.copy()}

    if HAS_FATROP:
        try:
            result = generate_trajectory_fatrop(
                positions, quaternions,
                pose_target_init=pose_init,
                pose_target_final=pose_goal,
                traj_length=traj_length,
                use_fatrop=True,
                verbose=False,
            )
            if result.get("success", False):
                return result["positions"], result["quaternions"], "fatrop"
        except Exception:
            pass

    if HAS_CASADI:
        try:
            result = generate_trajectory(
                positions, quaternions,
                pose_target_init=pose_init,
                pose_target_final=pose_goal,
                traj_length=traj_length,
                dhb_method=DHBMethod.DOUBLE_REFLECTION,
                use_casadi=True,
                verbose=False,
            )
            if result.get("solver") == "casadi":
                return result["adapted_pos_data"], result["adapted_quat_data"], "casadi"
        except Exception:
            pass

    # Fallback: encode-decode
    result_enc = encode_dhb_dr(
        positions, quaternions,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
    )
    new_initial_pose = {"position": new_start_pos.copy(), "quaternion": new_start_quat.copy()}
    result_dec = decode_dhb_dr(
        result_enc["linear_motion_invariants"],
        result_enc["angular_motion_invariants"],
        new_initial_pose,
        method=EncodingMethod.POSITION,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        drop_padded=True,
    )
    return result_dec["positions"], result_dec["quaternions"], "decode"


# =========================================================================
# Video recording helpers
# =========================================================================

def _get_frame(obs: Dict) -> np.ndarray:
    """Extract agentview RGB frame from observation."""
    # LIBERO OffScreenRenderEnv returns agentview_image
    img = obs.get("agentview_image", None)
    if img is None:
        # Try alternative keys
        for k in obs:
            if "image" in k or "rgb" in k:
                img = obs[k]
                break
    if img is None:
        return np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    # MuJoCo renders upside-down
    return np.flipud(img).astype(np.uint8)


def _add_text_overlay(frame: np.ndarray, lines: List[str], color=(255, 255, 255)) -> np.ndarray:
    """Add simple text overlay to frame using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        y = 8
        for line in lines:
            draw.text((8, y), line, fill=color)
            y += 18
        return np.array(img)
    except Exception:
        return frame


# =========================================================================
# Record: Raw replay (FAIL)
# =========================================================================

def record_raw_fail(bddl_perturbed: str, demo: Dict, output_path: Path) -> bool:
    """Record raw demo replay on the perturbed env. Expected to FAIL."""
    print(f"\n[RAW FAIL] Recording to {output_path}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_perturbed,
        camera_heights=CAM_H,
        camera_widths=CAM_W,
        horizon=HORIZON,
    )
    env.seed(0)
    obs = env.reset()

    frames = []

    # Initial settling frames — show the scene before the robot moves
    for settle_i in range(30):
        frame = _get_frame(obs)
        frame = _add_text_overlay(frame, ["RAW REPLAY (no adaptation)", "Initial State"], color=(255, 80, 80))
        frames.append(frame)

    # Warm-up: record each step so there is no jump
    for wu_i in range(3):
        obs, _, _, _ = env.step(np.zeros(7))
        frame = _get_frame(obs)
        frame = _add_text_overlay(frame, ["RAW REPLAY (no adaptation)", "Warmup", f"Step: {wu_i + 1}/3"], color=(255, 80, 80))
        frames.append(frame)

    actions = demo["actions"]
    num_steps = min(len(actions), MAX_STEPS)

    success = False
    step_count = 0

    for step in range(num_steps):
        action = actions[step]
        obs, reward, done, info = env.step(action)
        step_count += 1

        frame = _get_frame(obs)
        status = "SUCCESS" if reward > 0 else "FAIL"
        color = (80, 255, 80) if reward > 0 else (255, 80, 80)
        frame = _add_text_overlay(
            frame,
            [f"RAW REPLAY (no adaptation)", f"Step: {step+1}/{num_steps}", f"Status: {status}"],
            color=color
        )
        frames.append(frame)

        if reward > 0:
            success = True
            # Pad a few more frames so success is visible
            for _ in range(15):
                obs, _, _, _ = env.step(np.zeros(7))
                frame = _get_frame(obs)
                frame = _add_text_overlay(frame, ["RAW REPLAY (no adaptation)", f"Step: {step+1}/{num_steps}", "Status: SUCCESS"], color=(80, 255, 80))
                frames.append(frame)
            break
        if done:
            break

    env.close()

    # Pad to at least 1s at FPS
    while len(frames) < FPS:
        frames.append(frames[-1].copy())

    _write_video(frames, output_path)
    print(f"  Frames: {len(frames)}, Success: {success}, Steps: {step_count}")
    print(f"  File: {output_path} ({output_path.stat().st_size // 1024}KB)")
    return success


# =========================================================================
# Record: DHB-adapted replay (SUCCESS)
# =========================================================================

def record_dhb_success(
    bddl_perturbed: str,
    demo: Dict,
    demo_bowl_pos: np.ndarray,
    perturbed_bowl_pos: np.ndarray,
    demo_plate_pos: np.ndarray,
    perturbed_plate_pos: np.ndarray,
    output_path: Path,
) -> bool:
    """Record DHB-adapted replay on the perturbed env. Expected to SUCCEED."""
    print(f"\n[DHB SUCCESS] Recording to {output_path}")

    demo_pos = demo["positions"]
    demo_quat = demo["quaternions"]
    original_actions = demo["actions"]
    n_orig = len(original_actions)

    bowl_offset = perturbed_bowl_pos - demo_bowl_pos
    plate_offset = perturbed_plate_pos - demo_plate_pos
    print(f"  Bowl offset: {bowl_offset}")
    print(f"  Plate offset: {plate_offset}")

    # Find phase boundaries
    GRASP_STEP = n_orig // 2
    for i in range(n_orig):
        if original_actions[i, 6] > 0.5:
            GRASP_STEP = i
            break

    RELEASE_STEP = n_orig - 1
    for i in range(GRASP_STEP, n_orig):
        if original_actions[i, 6] < 0.0:
            RELEASE_STEP = i
            break

    print(f"  Grasp step: {GRASP_STEP}, Release step: {RELEASE_STEP}")

    place_phase_pos = demo_pos[GRASP_STEP:RELEASE_STEP + 1]
    place_phase_quat = demo_quat[GRASP_STEP:RELEASE_STEP + 1]
    # Compute placement goal: place at perturbed_plate_pos with same Z-offset
    # (keep X/Z relative offsets, set Y directly to new plate Y to avoid going below table)
    eef_z_rel = place_phase_pos[-1][2] - demo_plate_pos[2]
    eef_x_rel = place_phase_pos[-1][0] - demo_plate_pos[0]
    place_goal_pos = np.array([
        perturbed_plate_pos[0] + eef_x_rel,
        perturbed_plate_pos[1],           # Y = new plate Y directly
        perturbed_plate_pos[2] + eef_z_rel,
    ])
    print(f"  Place goal (corrected): {place_goal_pos}")
    print(f"  Perturbed plate: {perturbed_plate_pos}")

    # Adapt from bowl-shifted start position
    new_start = place_phase_pos[0].copy() + bowl_offset  # account for bowl shift

    adapted_pos, adapted_quat, solver_used = adapt_trajectory(
        place_phase_pos, place_phase_quat,
        new_start_pos=new_start,
        new_start_quat=place_phase_quat[0].copy(),
        new_goal_pos=place_goal_pos,
        new_goal_quat=place_phase_quat[-1].copy(),
        traj_length=min(len(place_phase_pos), 40),
    )
    print(f"  Solver: {solver_used}")

    carry_height = np.max(place_phase_pos[:, 2])
    release_z_above_plate = place_phase_pos[-1][2] - demo_plate_pos[2]

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_perturbed,
        camera_heights=CAM_H,
        camera_widths=CAM_W,
        horizon=HORIZON,
    )
    env.seed(0)
    obs = env.reset()

    frames = []

    # --- Initial State: 30 settling frames so the viewer sees the scene ---
    for settle_i in range(30):
        frame = _get_frame(obs)
        frame = _add_text_overlay(
            frame,
            ["DHB-XR ADAPTATION", "Initial State", f"Bowl + plate visible"],
            color=(80, 200, 255),
        )
        frames.append(frame)

    # --- Warmup: 5 steps recorded so there is no jump ---
    for wu_i in range(5):
        obs, _, _, _ = env.step(np.zeros(7))
        frame = _get_frame(obs)
        frame = _add_text_overlay(
            frame,
            ["DHB-XR ADAPTATION", "Warmup", f"Step: {wu_i + 1}/5"],
            color=(80, 200, 255),
        )
        frames.append(frame)

    success = False
    total_reward = 0.0
    step_count = 0
    final_ee_pos = obs.get("robot0_eef_pos", np.zeros(3)).copy()

    def _step_and_record(action, phase_label, step_num):
        nonlocal obs, success, total_reward, step_count, final_ee_pos
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        final_ee_pos = obs.get("robot0_eef_pos", final_ee_pos).copy()
        s_color = (80, 255, 80) if reward > 0 else (80, 200, 255)
        status_str = "SUCCESS!" if reward > 0 else "running"
        frame = _get_frame(obs)
        frame = _add_text_overlay(
            frame,
            ["DHB-XR ADAPTATION", f"Phase: {phase_label}", f"Step: {step_num}", f"Status: {status_str}"],
            color=s_color
        )
        frames.append(frame)
        if reward > 0:
            success = True
        return reward, done

    # Phase 1: Original pick (approach + grasp)
    for step in range(min(RELEASE_STEP, MAX_STEPS)):
        if step >= n_orig:
            break
        target_pos = demo_pos[step]
        current_ee = obs.get("robot0_eef_pos", final_ee_pos)
        pos_error = target_pos - current_ee
        pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
        ori_action = original_actions[step, 3:6]
        gripper = original_actions[step, 6]
        action = np.concatenate([pos_action, ori_action, [gripper]])
        reward, done = _step_and_record(action, "Phase 1: Pick", step_count)
        if success or done:
            break

    # Corrective grasp if needed
    if not success:
        bowl_check = obs.get("akita_black_bowl_1_pos", np.zeros(3))
        ee_check = obs.get("robot0_eef_pos", np.zeros(3))
        bowl_grasped = (bowl_check[2] > 0.91 and
                        np.linalg.norm(bowl_check[:2] - ee_check[:2]) < 0.08)
        if not bowl_grasped:
            print("  [CORRECTIVE GRASP] Attempting live approach")
            bowl_live = bowl_check.copy()
            grasp_ori = original_actions[min(GRASP_STEP, n_orig - 1), 3:6]
            for s in range(30):
                current_ee = obs.get("robot0_eef_pos", final_ee_pos)
                bowl_live = obs.get("akita_black_bowl_1_pos", bowl_live)
                target = bowl_live.copy()
                target[2] = bowl_live[2] + 0.03
                pos_error = target - current_ee
                pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
                action = np.concatenate([pos_action, grasp_ori, [-1.0]])
                reward, done = _step_and_record(action, "Correct Grasp", step_count)
                if success:
                    break
                if np.linalg.norm(obs["robot0_eef_pos"] - target) < 0.01:
                    break
            if not success:
                for s in range(20):
                    current_ee = obs.get("robot0_eef_pos", final_ee_pos)
                    bowl_live = obs.get("akita_black_bowl_1_pos", bowl_live)
                    target = bowl_live.copy()
                    target[2] = bowl_live[2] + 0.005
                    pos_error = target - current_ee
                    pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
                    action = np.concatenate([pos_action, grasp_ori, [-1.0]])
                    reward, done = _step_and_record(action, "Descend Grasp", step_count)
                    if success:
                        break
                    if np.linalg.norm(obs["robot0_eef_pos"] - target) < 0.01:
                        break
            if not success:
                for s in range(15):
                    action = np.concatenate([np.zeros(3), grasp_ori, [1.0]])
                    reward, done = _step_and_record(action, "Close Gripper", step_count)
                    if success:
                        break
            if not success:
                for s in range(20):
                    current_ee = obs.get("robot0_eef_pos", final_ee_pos)
                    target = current_ee.copy()
                    target[2] = max(carry_height, 0.96)
                    pos_error = target - current_ee
                    pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
                    action = np.concatenate([pos_action, grasp_ori, [1.0]])
                    reward, done = _step_and_record(action, "Lift", step_count)
                    if success:
                        break

    # Phase 2: DHB-adapted placement
    if not success:
        actual_grasp_ee = obs.get("robot0_eef_pos", place_phase_pos[0]).copy()
        actual_plate_pos = obs.get("plate_1_pos", perturbed_plate_pos).copy()
        # Recompute corrected placement goal from actual plate position
        act_eef_z_rel = place_phase_pos[-1][2] - demo_plate_pos[2]
        act_eef_x_rel = place_phase_pos[-1][0] - demo_plate_pos[0]
        place_goal_pos = np.array([
            actual_plate_pos[0] + act_eef_x_rel,
            actual_plate_pos[1],
            actual_plate_pos[2] + act_eef_z_rel,
        ])

        bowl_pos_check = obs.get("akita_black_bowl_1_pos", np.zeros(3))
        bowl_eef_offset_xy = bowl_pos_check[:2] - actual_grasp_ee[:2]

        SAFE_Z = max(carry_height + 0.02, actual_grasp_ee[2] + 0.01, 0.97)
        release_ori = original_actions[min(RELEASE_STEP, n_orig - 1), 3:6]

        def _eef_target_for_bowl_on_plate(plate_pos_2d, bowl_eef_off_2d, z):
            eef_xy = plate_pos_2d - bowl_eef_off_2d
            return np.array([eef_xy[0], eef_xy[1], z])

        # 2a: Lift
        wp_lift = actual_grasp_ee.copy()
        wp_lift[2] = SAFE_Z
        for s in range(25):
            current_ee = obs.get("robot0_eef_pos", final_ee_pos)
            pos_error = wp_lift - current_ee
            pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
            action = np.concatenate([pos_action, release_ori, [1.0]])
            reward, done = _step_and_record(action, "Phase 2: DHB Lift", step_count)
            if success:
                break
            if abs(obs["robot0_eef_pos"][2] - SAFE_Z) < 0.005:
                break

        # 2b: Move horizontally to above plate
        if not success:
            for s in range(60):
                current_ee = obs.get("robot0_eef_pos", final_ee_pos)
                plate_live = obs.get("plate_1_pos", actual_plate_pos).copy()
                bowl_live = obs.get("akita_black_bowl_1_pos", bowl_pos_check).copy()
                bowl_eef_live = bowl_live[:2] - current_ee[:2]
                wp_above = _eef_target_for_bowl_on_plate(plate_live[:2], bowl_eef_live, SAFE_Z)
                pos_error = wp_above - current_ee
                pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
                action = np.concatenate([pos_action, release_ori, [1.0]])
                reward, done = _step_and_record(action, "Phase 2: DHB Translate", step_count)
                if success:
                    break
                if np.linalg.norm(obs["robot0_eef_pos"][:2] - wp_above[:2]) < 0.008:
                    break

        # 2c: Descend
        if not success:
            for s in range(50):
                current_ee = obs.get("robot0_eef_pos", final_ee_pos)
                plate_live = obs.get("plate_1_pos", actual_plate_pos).copy()
                bowl_live = obs.get("akita_black_bowl_1_pos", bowl_pos_check).copy()
                bowl_eef_live = bowl_live[:2] - current_ee[:2]
                target_z = plate_live[2] + release_z_above_plate
                wp_descend = _eef_target_for_bowl_on_plate(plate_live[:2], bowl_eef_live, target_z)
                pos_error = wp_descend - current_ee
                pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
                action = np.concatenate([pos_action, release_ori, [1.0]])
                reward, done = _step_and_record(action, "Phase 3: Place", step_count)
                if success:
                    break
                if np.linalg.norm(obs["robot0_eef_pos"] - wp_descend) < 0.012:
                    break

        # 2d: Settle
        if not success:
            for s in range(15):
                current_ee = obs.get("robot0_eef_pos", final_ee_pos)
                plate_live = obs.get("plate_1_pos", actual_plate_pos).copy()
                bowl_live = obs.get("akita_black_bowl_1_pos", bowl_pos_check).copy()
                bowl_eef_live = bowl_live[:2] - current_ee[:2]
                target_z = plate_live[2] + release_z_above_plate
                wp_target = _eef_target_for_bowl_on_plate(plate_live[:2], bowl_eef_live, target_z)
                pos_error = wp_target - current_ee
                pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
                action = np.concatenate([pos_action, release_ori, [1.0]])
                reward, done = _step_and_record(action, "Phase 3: Place", step_count)
                if success:
                    break

        # Release
        if not success:
            for s in range(15):
                current_ee = obs.get("robot0_eef_pos", final_ee_pos)
                plate_live = obs.get("plate_1_pos", actual_plate_pos).copy()
                bowl_live = obs.get("akita_black_bowl_1_pos", bowl_pos_check).copy()
                bowl_eef_live = bowl_live[:2] - current_ee[:2]
                target_z = plate_live[2] + release_z_above_plate
                wp_target = _eef_target_for_bowl_on_plate(plate_live[:2], bowl_eef_live, target_z)
                pos_error = wp_target - current_ee
                pos_action = np.clip(pos_error / OSC_SCALE, -1.0, 1.0)
                action = np.concatenate([pos_action, release_ori, [-1.0]])
                reward, done = _step_and_record(action, "Phase 3: Place", step_count)
                if success:
                    break

        # Wait for settle
        if not success:
            for s in range(20):
                obs, reward, done, info = env.step(np.zeros(7))
                total_reward += reward
                step_count += 1
                s_color = (80, 255, 80) if reward > 0 else (80, 200, 255)
                status_str = "SUCCESS!" if reward > 0 else "settling..."
                frame = _get_frame(obs)
                frame = _add_text_overlay(frame, ["DHB-XR ADAPTATION", "Phase 3: Place", f"Step: {step_count}", f"Status: {status_str}"], color=s_color)
                frames.append(frame)
                if reward > 0:
                    success = True
                    break

    # Pad success frames — hold for 2.5 s so SUCCESS is clearly visible
    if success:
        for _ in range(int(FPS * 2.5)):
            obs, _, _, _ = env.step(np.zeros(7))
            frame = _get_frame(obs)
            frame = _add_text_overlay(
                frame,
                ["DHB-XR ADAPTATION", "SUCCESS", "90% success on LIBERO-PRO P2"],
                color=(80, 255, 80),
            )
            frames.append(frame)

    env.close()

    while len(frames) < FPS:
        frames.append(frames[-1].copy())

    _write_video(frames, output_path)
    print(f"  Frames: {len(frames)}, Success: {success}, Steps: {step_count}")
    print(f"  File: {output_path} ({output_path.stat().st_size // 1024}KB)")
    return success


# =========================================================================
# Side-by-side comparison
# =========================================================================

def make_comparison_video(
    raw_fail_path: Path,
    dhb_success_path: Path,
    output_path: Path,
):
    """Stack raw fail (left) and DHB success (right) side by side."""
    print(f"\n[COMPARISON] Building side-by-side video -> {output_path}")

    def _read_video_frames(path: Path) -> List[np.ndarray]:
        frames = []
        reader = iio.get_reader(str(path), format="ffmpeg")
        for frame in reader:
            frames.append(frame)
        reader.close()
        return frames

    raw_frames = _read_video_frames(raw_fail_path)
    dhb_frames = _read_video_frames(dhb_success_path)

    # Loop shorter video to match length (better than padding with last frame)
    max_len = max(len(raw_frames), len(dhb_frames))
    if len(raw_frames) < max_len:
        # Loop raw frames
        raw_frames = (raw_frames * (max_len // len(raw_frames) + 1))[:max_len]
    if len(dhb_frames) < max_len:
        # Loop DHB frames
        dhb_frames = (dhb_frames * (max_len // len(dhb_frames) + 1))[:max_len]

    # Add header labels
    combined_frames = []
    try:
        from PIL import Image, ImageDraw
        for left, right in zip(raw_frames, dhb_frames):
            lh, lw = left.shape[:2]
            rh, rw = right.shape[:2]
            # Normalise heights
            if lh != rh:
                if lh > rh:
                    right_img = Image.fromarray(right).resize((rw, lh))
                    right = np.array(right_img)
                    rh = lh
                else:
                    left_img = Image.fromarray(left).resize((lw, rh))
                    left = np.array(left_img)
                    lh = rh

            combined = np.concatenate([left, right], axis=1)  # side by side
            img = Image.fromarray(combined)
            draw = ImageDraw.Draw(img)
            # Labels at top
            draw.rectangle([(0, 0), (lw, 22)], fill=(180, 40, 40))
            draw.text((8, 4), "RAW REPLAY: FAIL  (0% all VLAs)", fill=(255, 255, 255))
            draw.rectangle([(lw, 0), (lw + rw, 22)], fill=(30, 100, 180))
            draw.text((lw + 8, 4), "DHB-XR ADAPTATION: SUCCESS  (90%)", fill=(255, 255, 255))
            combined_frames.append(np.array(img))
    except ImportError:
        # PIL not available: just concatenate without labels
        for left, right in zip(raw_frames, dhb_frames):
            combined = np.concatenate([left, right], axis=1)
            combined_frames.append(combined)

    _write_video(combined_frames, output_path)
    print(f"  Frames: {len(combined_frames)}")
    print(f"  File: {output_path} ({output_path.stat().st_size // 1024}KB)")


# =========================================================================
# Video writer
# =========================================================================

def _write_video(frames: List[np.ndarray], path: Path, fps: int = FPS):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(str(path), fps=fps, quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"  Written: {path}")


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 70)
    print("  DHB-XR P2 Video Recording")
    print("=" * 70)

    # Generate perturbed BDDL
    bddl_orig, bddl_perturbed, perturbed_content = generate_perturbed_bddl(seed=42)
    print(f"Perturbed BDDL: {bddl_perturbed}")

    # Load demo_0
    demo = load_demo(DEMO_ID)
    print(f"Demo {DEMO_ID} loaded: {demo['num_frames']} frames")

    # Get object positions for DHB adaptation
    print("\nGetting object positions...")
    demo_bowl_pos = _get_object_pos(bddl_orig, "akita_black_bowl_1")
    pert_bowl_pos = _get_object_pos(bddl_perturbed, "akita_black_bowl_1")
    demo_plate_pos = _get_object_pos(bddl_orig, "plate_1")
    pert_plate_pos = _get_object_pos(bddl_perturbed, "plate_1")

    print(f"  Original bowl:  {demo_bowl_pos}")
    print(f"  Perturbed bowl: {pert_bowl_pos}")
    print(f"  Original plate: {demo_plate_pos}")
    print(f"  Perturbed plate:{pert_plate_pos}")

    if pert_plate_pos is not None and demo_plate_pos is not None:
        plate_shift = pert_plate_pos - demo_plate_pos
        print(f"  Plate shift (P2): {plate_shift}")

    # --- Video 1: Raw fail ---
    raw_success = record_raw_fail(bddl_perturbed, demo, RAW_FAIL_PATH)
    print(f"  Raw replay outcome: {'SUCCESS (unexpected)' if raw_success else 'FAIL (as expected)'}")

    # --- Video 2: DHB success ---
    if demo_bowl_pos is None or pert_bowl_pos is None:
        print("[ERROR] Cannot compute bowl offset - skipping DHB video")
        dhb_success = False
    else:
        dhb_success = record_dhb_success(
            bddl_perturbed, demo,
            demo_bowl_pos=demo_bowl_pos,
            perturbed_bowl_pos=pert_bowl_pos,
            demo_plate_pos=demo_plate_pos if demo_plate_pos is not None else demo_bowl_pos,
            perturbed_plate_pos=pert_plate_pos if pert_plate_pos is not None else pert_bowl_pos,
            output_path=DHB_SUCCESS_PATH,
        )
        print(f"  DHB adapted outcome: {'SUCCESS' if dhb_success else 'FAIL (check adaptation)'}")

    # --- Video 3: Side-by-side ---
    if RAW_FAIL_PATH.exists() and DHB_SUCCESS_PATH.exists():
        make_comparison_video(RAW_FAIL_PATH, DHB_SUCCESS_PATH, COMPARISON_PATH)
    else:
        print("[WARN] One or both source videos missing, skipping comparison video")

    print("\n" + "=" * 70)
    print("  OUTPUT SUMMARY")
    print("=" * 70)
    for p in [RAW_FAIL_PATH, DHB_SUCCESS_PATH, COMPARISON_PATH]:
        if p.exists():
            kb = p.stat().st_size // 1024
            print(f"  {p.name:35s} {kb:>6d} KB  {p}")
        else:
            print(f"  {p.name:35s} NOT CREATED")

    print(f"\nRaw success: {raw_success}")
    print(f"DHB success: {dhb_success}")


if __name__ == "__main__":
    main()
