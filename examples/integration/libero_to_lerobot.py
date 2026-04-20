#!/usr/bin/env python3
"""Convert LIBERO HDF5 datasets to LeRobot v2.0 format for StarVLA training.

This converter reads LIBERO HDF5 trajectories and converts them to LeRobot's
parquet-based format, which is expected by StarVLA's dataloader.

Features:
- Converts LIBERO HDF5 demos to LeRobot parquet chunks
- Generates required LeRobot metadata files (modality, info, episodes, tasks)
- Supports precomputed DHB invariants as additional features
- Handles multiple tasks in a single dataset
- Computes dataset statistics for normalization

LeRobot v2.0 Directory Structure:
  dataset_root/
    meta/
      modality.json          - State/action/video modality definitions
      info.json              - Dataset-level metadata
      episodes.jsonl         - Episode metadata (one per line)
      tasks.jsonl            - Task descriptions
      stats_gr00t.json       - Dataset statistics for normalization
    data/
      chunk_0/
        episode_000.parquet
        episode_001.parquet
        ...
      chunk_1/
        episode_016.parquet
        ...
    videos/
      chunk_0/
        episode_000_primary_image.mp4
        episode_000_wrist_image.mp4
        ...

LIBERO HDF5 Structure:
  demo_X/
    actions: [T, 7] - joint target velocities
    robot_states: [T, 9] - [gripper_state(2), ee_pos(3), ee_quat_wxyz(4)]
    agentview_rgb: [T, H, W, 3]
    wrist_rgb: [T, H, W, 3]  (optional)

Usage:
    python libero_to_lerobot.py \
        --libero-root /path/to/libero/spatial \
        --output-root /path/to/output/libero_lerobot \
        --task-name "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate" \
        --dhb-invariants /path/to/precomputed_dhb.pkl

    # Convert all tasks
    python libero_to_lerobot.py \
        --libero-root /path/to/libero/spatial \
        --output-root /path/to/output/libero_lerobot \
        --all-tasks

Author: Generated for DHB-XR + StarVLA integration
Date: 2026-04-09
"""

import argparse
import json
import logging
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from dhb_xr.encoder.dhb_dr import encode_dhb_dr

try:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    HAS_LIBERO = True
except ImportError:  # pragma: no cover - optional dependency in unit tests
    HAS_LIBERO = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Configuration
# ============================================================================

# LeRobot v2.0 file structure
LEROBOT_CHUNK_SIZE = 16  # Episodes per chunk file
LEROBOT_VIDEO_CODEC = "libx264"
LEROBOT_VIDEO_FPS = 30.0

# LIBERO HDF5 structure
LIBERO_DEMO_PREFIX = "demo_"
LIBERO_ACTIONS_KEY = "actions"
LIBERO_ROBOT_STATES_KEY = "robot_states"
LIBERO_AGENTVIEW_KEY = "agentview_rgb"
LIBERO_WRIST_KEY = "eye_in_hand_rgb"

# Default video resolution
DEFAULT_VIDEO_RES = (512, 384)  # (width, height)

# Action mode configuration
ACTION_MODE = "delta"
LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD = 0.015
CONTACT_EVENT_LABEL_RADIUS = 2
LIBERO_CLOSE_GATE_XY_M = 0.05
LIBERO_CLOSE_GATE_Z_M = 0.05
# Offline label generation uses a slightly wider plate gate than the online
# execution veto. The stricter 0.04m runtime gate produced zero positive
# release labels on the reference demo, which starves retraining of reopen
# supervision entirely.
LIBERO_RELEASE_GATE_XY_M = 0.05
LIBERO_RELEASE_GATE_Z_M = 0.06


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_libero_demo(h5_path: Path, demo_id: int) -> Dict:
    """Load a single demo from LIBERO HDF5 file.
    
    Args:
        h5_path: Path to HDF5 file
        demo_id: Demo index (0-based)
        
    Returns:
        Dictionary with keys: actions, ee_pos, ee_quat, agentview_rgb, wrist_rgb
    """
    with h5py.File(h5_path, "r") as f:
        demo_key = f"{LIBERO_DEMO_PREFIX}{demo_id}"
        
        if f"data/{demo_key}" not in f:
            # Try alternative structure
            if demo_key in f.get("data", {}):
                demo = f["data"][demo_key]
            else:
                available = [k for k in f.get("data", {}).keys() if k.startswith(LIBERO_DEMO_PREFIX)]
                raise ValueError(f"Demo {demo_id} not found in {h5_path}. Available: {available[:5]}")
        else:
            demo = f[f"data/{demo_key}"]

        # Extract actions
        actions = np.array(demo[LIBERO_ACTIONS_KEY], dtype=np.float32)
        sim_states = np.array(demo["states"], dtype=np.float32) if "states" in demo else None
        
        # Extract robot states
        robot_states = np.array(demo[LIBERO_ROBOT_STATES_KEY], dtype=np.float64)
        
        # Parse robot states: [gripper_state(2), ee_pos(3), ee_quat_wxyz(4)]
        gripper_state = robot_states[:, 0:2].astype(np.float32)  # [T, 2]
        ee_pos = robot_states[:, 2:5].astype(np.float32)  # [T, 3]
        ee_quat_wxyz = robot_states[:, 5:9].astype(np.float64)  # [T, 4] in (w, x, y, z)
        
        # Convert quaternion from (w, x, y, z) to (x, y, z, w)
        ee_quat = ee_quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32)
        
        # Normalize quaternions
        ee_quat = ee_quat / np.linalg.norm(ee_quat, axis=1, keepdims=True)
        
        # Convert quaternion to Euler angles (roll, pitch, yaw) in radians
        rotations = Rotation.from_quat(ee_quat)
        euler_angles = rotations.as_euler("xyz", degrees=False).astype(np.float32)  # [T, 3]
        ee_axis_angle = rotations.as_rotvec().astype(np.float32)  # [T, 3]

        # Derived StarVLA-compatible delta action targets.
        delta_pos = np.zeros_like(ee_pos, dtype=np.float32)
        if len(ee_pos) > 1:
            delta_pos[:-1] = ee_pos[1:] - ee_pos[:-1]

        delta_rot = np.zeros_like(ee_axis_angle, dtype=np.float32)
        if len(ee_quat) > 1:
            q_curr = Rotation.from_quat(ee_quat[:-1])
            q_next = Rotation.from_quat(ee_quat[1:])
            # Delta in the current EE frame: current * delta = next.
            delta_rot[:-1] = (q_curr.inv() * q_next).as_rotvec().astype(np.float32)

        # LIBERO gripper qpos is small-valued (~0.0-0.04), not binary 0/1.
        # Use the same open/close threshold as the validated SmolVLA/embodik pipeline.
        gripper_open = (gripper_state[:, 0] > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD).astype(np.float32).reshape(-1, 1)
        next_gripper_open = np.concatenate([gripper_open[1:], gripper_open[-1:]], axis=0)
        delta_actions = np.concatenate([delta_pos, delta_rot, next_gripper_open], axis=1).astype(np.float32)
        
        # Extract images. Current LIBERO spatial demos store image streams under
        # demo["obs"], not at the demo root.
        agentview_rgb = None
        wrist_rgb = None

        obs_group = demo["obs"] if "obs" in demo else demo
        if LIBERO_AGENTVIEW_KEY in obs_group:
            agentview_rgb = np.array(obs_group[LIBERO_AGENTVIEW_KEY], dtype=np.uint8)  # [T, H, W, 3]

        if LIBERO_WRIST_KEY in obs_group:
            wrist_rgb = np.array(obs_group[LIBERO_WRIST_KEY], dtype=np.uint8)  # [T, H, W, 3]
        
        return {
            "actions": actions,  # [T, 7] - joint velocities
            "gripper_state": gripper_state,  # [T, 2]
            "ee_pos": ee_pos,  # [T, 3]
            "ee_quat": ee_quat,  # [T, 4] in (x, y, z, w)
            "euler_angles": euler_angles,  # [T, 3] roll, pitch, yaw
            "ee_axis_angle": ee_axis_angle,  # [T, 3] axis-angle / rotvec
            "delta_actions": delta_actions,  # [T, 7] delta_pos + delta_rotvec + next_gripper_open
            "sim_states": sim_states,  # [T, nq+nv] flattened MuJoCo states for exact scene replay
            "agentview_rgb": agentview_rgb,  # [T, H, W, 3]
            "wrist_rgb": wrist_rgb,  # [T, H, W, 3]
            "length": len(actions),
        }


class _LiberoGeometryLabeler:
    """Replay LIBERO demo states to derive bowl/plate-relative gripper labels."""

    def __init__(
        self,
        env,
        *,
        bowl_key: str,
        plate_key: str,
        close_gate_xy_m: float = LIBERO_CLOSE_GATE_XY_M,
        close_gate_z_m: float = LIBERO_CLOSE_GATE_Z_M,
        release_gate_xy_m: float = LIBERO_RELEASE_GATE_XY_M,
        release_gate_z_m: float = LIBERO_RELEASE_GATE_Z_M,
    ) -> None:
        self.env = env
        self.bowl_key = bowl_key
        self.plate_key = plate_key
        self.close_gate_xy_m = float(close_gate_xy_m)
        self.close_gate_z_m = float(close_gate_z_m)
        self.release_gate_xy_m = float(release_gate_xy_m)
        self.release_gate_z_m = float(release_gate_z_m)

    def close(self) -> None:
        self.env.close()

    def replay_relative_snapshots(self, sim_states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bowl_rel = []
        plate_rel = []
        for mujoco_state in np.asarray(sim_states, dtype=np.float32):
            obs = self.env.set_init_state(mujoco_state)
            bowl_rel.append(np.asarray(obs[self.bowl_key], dtype=np.float32))
            plate_rel.append(np.asarray(obs[self.plate_key], dtype=np.float32))
        return np.stack(bowl_rel, axis=0), np.stack(plate_rel, axis=0)

    def build_labels(self, sim_states: np.ndarray, gripper_open: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        bowl_rel, plate_rel = self.replay_relative_snapshots(sim_states)
        return _geometry_contact_labels(
            bowl_rel=bowl_rel,
            plate_rel=plate_rel,
            gripper_open=gripper_open,
            close_gate_xy_m=self.close_gate_xy_m,
            close_gate_z_m=self.close_gate_z_m,
            release_gate_xy_m=self.release_gate_xy_m,
            release_gate_z_m=self.release_gate_z_m,
        )


def _find_task_by_name(task_suite_name: str, task_name: str):
    if not HAS_LIBERO:
        raise ImportError("libero is required for geometry-conditioned label generation.")

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise KeyError(f"Unknown LIBERO task suite '{task_suite_name}'")

    task_suite = benchmark_dict[task_suite_name]()
    for task_id in range(task_suite.get_num_tasks()):
        task = task_suite.get_task(task_id)
        if task.name == task_name:
            return task
    raise KeyError(f"Task '{task_name}' not found in LIBERO suite '{task_suite_name}'")


def _build_geometry_labeler(task_suite_name: str, task_name: str) -> _LiberoGeometryLabeler:
    task = _find_task_by_name(task_suite_name, task_name)
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        controller="OSC_POSE",
        camera_heights=128,
        camera_widths=128,
    )
    env.reset()
    obs = env.env._get_observations()

    def _pick_rel_key(target: str) -> str:
        matches = sorted(
            key
            for key in obs.keys()
            if key.endswith("_to_robot0_eef_pos") and target in key.lower()
        )
        if not matches:
            raise KeyError(
                f"Could not find '*_to_robot0_eef_pos' key containing '{target}' "
                f"for task '{task_name}'. Available keys sample: {sorted(obs.keys())[:20]}"
            )
        return matches[0]

    return _LiberoGeometryLabeler(
        env,
        bowl_key=_pick_rel_key("bowl"),
        plate_key=_pick_rel_key("plate"),
    )


def _event_transition_labels(
    gripper_open: np.ndarray,
    transition: str,
    radius: int = CONTACT_EVENT_LABEL_RADIUS,
) -> np.ndarray:
    """Build soft close/open event labels from binary demo gripper states.

    These labels stay tied to demonstrated contact transitions, but they are
    represented as local event windows rather than absolute-timestep replay.
    The model can then learn chunk-relative contact phase jointly with the DHB
    trajectory invariants.
    """
    if transition not in {"close", "open"}:
        raise ValueError(f"Unsupported transition type: {transition}")

    open_now = np.asarray(gripper_open, dtype=np.float32).reshape(-1)
    if open_now.size == 0:
        return np.zeros(0, dtype=np.float32)

    next_open = np.concatenate([open_now[1:], open_now[-1:]], axis=0)
    if transition == "close":
        event_indices = np.where((open_now > 0.5) & (next_open <= 0.5))[0]
    else:
        event_indices = np.where((open_now <= 0.5) & (next_open > 0.5))[0]

    labels = np.zeros_like(open_now, dtype=np.float32)
    if radius < 0:
        raise ValueError("radius must be non-negative")

    for idx in event_indices:
        for offset in range(-radius, radius + 1):
            target = idx + offset
            if 0 <= target < len(labels):
                weight = 1.0 - (abs(offset) / float(radius + 1))
                labels[target] = max(labels[target], weight)

    return labels.astype(np.float32)


def _event_class_labels(
    gripper_open: np.ndarray,
    radius: int = CONTACT_EVENT_LABEL_RADIUS,
) -> np.ndarray:
    """Build integer gripper event labels: 0=hold, 1=close, 2=open."""
    open_now = np.asarray(gripper_open, dtype=np.float32).reshape(-1)
    if open_now.size == 0:
        return np.zeros(0, dtype=np.int64)

    labels = np.zeros_like(open_now, dtype=np.int64)
    next_open = np.concatenate([open_now[1:], open_now[-1:]], axis=0)
    close_indices = np.where((open_now > 0.5) & (next_open <= 0.5))[0]
    open_indices = np.where((open_now <= 0.5) & (next_open > 0.5))[0]

    for idx in close_indices:
        start = max(0, idx - radius)
        end = min(len(labels), idx + radius + 1)
        labels[start:end] = 1

    for idx in open_indices:
        start = max(0, idx - radius)
        end = min(len(labels), idx + radius + 1)
        labels[start:end] = 2

    return labels.astype(np.int64)


def _geometry_contact_labels(
    *,
    bowl_rel: np.ndarray,
    plate_rel: np.ndarray,
    gripper_open: np.ndarray,
    close_gate_xy_m: float = LIBERO_CLOSE_GATE_XY_M,
    close_gate_z_m: float = LIBERO_CLOSE_GATE_Z_M,
    release_gate_xy_m: float = LIBERO_RELEASE_GATE_XY_M,
    release_gate_z_m: float = LIBERO_RELEASE_GATE_Z_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build close/open labels from replayed object-relative geometry.

    This keeps the same exported schema as the transition-based path:
    ``dhb.close_contact`` and ``dhb.open_contact`` are float masks marking
    bowl-/plate-ready occupancy, while ``dhb.gripper_event`` is a sparse
    transition token: ``0=hold, 1=close, 2=open``.
    """
    bowl_rel = np.asarray(bowl_rel, dtype=np.float32)
    plate_rel = np.asarray(plate_rel, dtype=np.float32)
    open_now = np.asarray(gripper_open, dtype=np.float32).reshape(-1)
    if bowl_rel.shape[0] != len(open_now) or plate_rel.shape[0] != len(open_now):
        raise ValueError(
            "Geometry label inputs must share the same length: "
            f"bowl={bowl_rel.shape}, plate={plate_rel.shape}, open={open_now.shape}"
        )

    next_open = np.concatenate([open_now[1:], open_now[-1:]], axis=0)
    close_transitions = (open_now > 0.5) & (next_open <= 0.5)
    open_transitions = (open_now <= 0.5) & (next_open > 0.5)
    has_closed_before = np.cumsum(close_transitions.astype(np.int32)) > 0
    has_opened_after_close = np.cumsum(open_transitions.astype(np.int32)) > 0

    close_ready = (
        (np.abs(bowl_rel[:, 0]) <= close_gate_xy_m)
        & (np.abs(bowl_rel[:, 1]) <= close_gate_xy_m)
        & (bowl_rel[:, 2] <= close_gate_z_m)
    )
    release_ready = (
        (np.abs(plate_rel[:, 0]) <= release_gate_xy_m)
        & (np.abs(plate_rel[:, 1]) <= release_gate_xy_m)
        & (plate_rel[:, 2] <= release_gate_z_m)
    )

    close_contact = (
        (open_now > 0.5)
        & close_ready
        & (~has_opened_after_close)
    ).astype(np.float32)
    open_contact = (
        (open_now <= 0.5)
        & release_ready
        & has_closed_before
    ).astype(np.float32)
    close_edge = close_contact > 0.5
    close_edge[1:] &= ~(close_contact[:-1] > 0.5)
    open_edge = open_contact > 0.5
    open_edge[1:] &= ~(open_contact[:-1] > 0.5)

    gripper_event = np.zeros(len(open_now), dtype=np.int64)
    gripper_event[close_edge] = 1
    gripper_event[open_edge] = 2
    return close_contact, open_contact, gripper_event


def list_demos_in_file(h5_path: Path) -> List[int]:
    """List all demo IDs in an HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        demo_group = f.get("data", f)
        demos = sorted([
            int(k.replace(LIBERO_DEMO_PREFIX, ""))
            for k in demo_group.keys()
            if k.startswith(LIBERO_DEMO_PREFIX)
        ])
    return demos


# ============================================================================
# Metadata Generation Functions
# ============================================================================

def create_modality_metadata() -> Dict:
    """Create LeRobot modality metadata for LIBERO data.
    
    This defines the state/action/video structure for StarVLA's dataloader.
    """
    modality_metadata = {
        "state": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "pad": {"start": 6, "end": 7},
            "gripper": {"start": 7, "end": 8},
        },
        "action": {
            "x": {"start": 0, "end": 1},
            "y": {"start": 1, "end": 2},
            "z": {"start": 2, "end": 3},
            "roll": {"start": 3, "end": 4},
            "pitch": {"start": 4, "end": 5},
            "yaw": {"start": 5, "end": 6},
            "gripper": {"start": 6, "end": 7},
        },
        "video": {
            "primary_image": {"original_key": "observation.images.image"},
            "wrist_image": {"original_key": "observation.images.wrist_image"},
        },
        "annotation": {
            "human.action.task_description": {"original_key": "task_index"}
        }
    }
    return modality_metadata


def create_info_metadata(num_tasks: int = 1) -> Dict:
    """Create LeRobot info metadata.
    
    Provides dataset-level information and feature schemas.
    """
    info_metadata = {
        "name": "libero_lerobot",
        "version": "2.0",
        "data_path": "data/{episode_chunk:06d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{episode_chunk:06d}/episode_{episode_index:06d}_{video_key}.mp4",
        "chunks_size": LEROBOT_CHUNK_SIZE,
        "num_episodes": 0,  # Will be updated
        "num_tasks": num_tasks,
        "num_frames": 0,  # Will be updated
        "features": {
            "observation.images.image": {
                "shape": [DEFAULT_VIDEO_RES[1], DEFAULT_VIDEO_RES[0], 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": LEROBOT_VIDEO_FPS,
                    "video.channels": 3,
                },
            },
            "observation.images.wrist_image": {
                "shape": [DEFAULT_VIDEO_RES[1], DEFAULT_VIDEO_RES[0], 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": LEROBOT_VIDEO_FPS,
                    "video.channels": 3,
                },
            },
            "observation.state": {
                "shape": [8],  # x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, pad, gripper
                "dtype": "float32",
            },
            "action": {
                "shape": [7],  # delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, open_gripper
                "dtype": "float32",
            },
            "timestamp": {
                "shape": [],
                "dtype": "float32",
            },
            "task_index": {
                "shape": [],
                "dtype": "int32",
            },
        }
    }
    return info_metadata


def create_episode_metadata(episode_index: int, length: int, task_index: int = 0) -> Dict:
    """Create metadata for a single episode."""
    return {
        "episode_index": episode_index,
        "length": length,
        "task_index": task_index,
    }


def create_task_metadata(task_index: int, task_name: str) -> Dict:
    """Create metadata for a task."""
    return {
        "task_index": task_index,
        "task": task_name,
    }


# ============================================================================
# Parquet Data Generation Functions
# ============================================================================

def create_episode_dataframe(
    demo_data: Dict,
    episode_index: int,
    task_index: int = 0,
    dhb_invariants: Optional[Dict] = None,
    geometry_labeler: Optional[_LiberoGeometryLabeler] = None,
) -> pd.DataFrame:
    """Create a pandas DataFrame for an episode in LeRobot format.
    
    Args:
        demo_data: Loaded demo data from LIBERO
        episode_index: Global episode index
        task_index: Task index for this episode
        dhb_invariants: Optional precomputed DHB features
        
    Returns:
        DataFrame with columns for observation.state, action, etc.
    """
    length = demo_data["length"]
    
    # Create timestamp (assuming 30 FPS)
    timestamps = np.arange(length, dtype=np.float32) / LEROBOT_VIDEO_FPS
    
    # Prepare state vector: [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z, pad, gripper]
    state_vector = np.column_stack([
        demo_data["ee_pos"],  # x, y, z
        demo_data["ee_axis_angle"],  # axis-angle / rotvec
        np.zeros((length, 1), dtype=np.float32),  # pad dimension for libero_franka config
        demo_data["gripper_state"][:, 0:1],  # gripper (take first component)
    ]).astype(np.float32)

    # Prepare action vector (StarVLA LIBERO-style deltas):
    # [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, next_open_gripper]
    action_vector = demo_data["delta_actions"].astype(np.float32)
    gripper_open = (demo_data["gripper_state"][:, 0] > LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD).astype(np.float32)
    if geometry_labeler is not None and demo_data.get("sim_states") is not None:
        close_contact, open_contact, gripper_event = geometry_labeler.build_labels(
            demo_data["sim_states"],
            gripper_open,
        )
    else:
        close_contact = _event_transition_labels(gripper_open, transition="close")
        open_contact = _event_transition_labels(gripper_open, transition="open")
        gripper_event = _event_class_labels(gripper_open)
    
    # Build one parquet row per environment step. The StarVLA LeRobot loader
    # expects trajectory data to be indexed by timestep inside each episode
    # parquet, while RGB frames are read from side-car mp4 files using the
    # timestamps stored here.
    data_dict = {
        "observation.state": list(state_vector),
        "action": list(action_vector),
        "timestamp": timestamps.astype(np.float32),
        "task_index": np.full(length, task_index, dtype=np.int32),
    }

    dhb_feats = None
    if dhb_invariants and episode_index in dhb_invariants:
        dhb_feats = dhb_invariants[episode_index]
    else:
        if length < 3:
            lin = np.zeros((length, 4), dtype=np.float32)
            ang = np.zeros((length, 4), dtype=np.float32)
        else:
            enc = encode_dhb_dr(
                demo_data["ee_pos"].astype(np.float64),
                demo_data["ee_quat"][:, [3, 0, 1, 2]].astype(np.float64),
            )
            lin = np.asarray(enc["linear_motion_invariants"], dtype=np.float32)
            ang = np.asarray(enc["angular_motion_invariants"], dtype=np.float32)
            if lin.shape[0] > length:
                lin = lin[-length:]
            if ang.shape[0] > length:
                ang = ang[-length:]
            if lin.shape[0] == length - 1:
                lin = np.concatenate([lin, lin[-1:]], axis=0)
            if ang.shape[0] == length - 1:
                ang = np.concatenate([ang, ang[-1:]], axis=0)
        dhb_feats = {
            "linear_invariants": lin,
            "angular_invariants": ang,
        }

    for key, values in dhb_feats.items():
        arr = np.array(values, dtype=np.float32)
        if len(arr) != length:
            raise ValueError(
                f"DHB invariant '{key}' length mismatch for episode {episode_index}: "
                f"{len(arr)} != {length}"
            )
        data_dict[f"dhb.{key}"] = list(arr)

    data_dict["dhb.close_contact"] = close_contact.astype(np.float32).tolist()
    data_dict["dhb.open_contact"] = open_contact.astype(np.float32).tolist()
    data_dict["dhb.gripper_event"] = gripper_event.astype(np.int64).tolist()
    
    df = pd.DataFrame(data_dict)
    return df


def _resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize LIBERO RGB frames to the configured LeRobot resolution."""
    from PIL import Image

    if frame.shape == DEFAULT_VIDEO_RES[::-1] + (3,):
        return frame.astype(np.uint8)
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((DEFAULT_VIDEO_RES[0], DEFAULT_VIDEO_RES[1]))
    return np.array(img, dtype=np.uint8)


def write_episode_videos(
    demo_data: Dict,
    output_root: Path,
    episode_index: int,
) -> None:
    """Write MP4 side-car videos for an episode in LeRobot v2.0 layout."""
    chunk_index = episode_index // LEROBOT_CHUNK_SIZE
    video_dir = output_root / "videos" / f"{chunk_index:06d}"
    video_dir.mkdir(parents=True, exist_ok=True)

    def _write_stream(frames: np.ndarray | None, video_key: str) -> None:
        if frames is None:
            frames = np.zeros((demo_data["length"], DEFAULT_VIDEO_RES[1], DEFAULT_VIDEO_RES[0], 3), dtype=np.uint8)
        video_path = video_dir / f"episode_{episode_index:06d}_{video_key}.mp4"
        writer = imageio.get_writer(video_path, fps=LEROBOT_VIDEO_FPS)
        try:
            for frame in frames:
                writer.append_data(_resize_frame(frame))
        finally:
            writer.close()

    _write_stream(demo_data.get("agentview_rgb"), "observation.images.image")
    _write_stream(demo_data.get("wrist_rgb"), "observation.images.wrist_image")


# ============================================================================
# Main Conversion Function
# ============================================================================

def convert_libero_to_lerobot(
    libero_h5_path: Path,
    output_root: Path,
    task_name: str,
    task_index: int = 0,
    dhb_invariants: Optional[Dict] = None,
    demo_ids: Optional[List[int]] = None,
    episode_offset: int = 0,
) -> Tuple[int, int, List[Dict], Dict[int, str]]:
    """Convert a single LIBERO HDF5 file to LeRobot format.
    
    Args:
        libero_h5_path: Path to LIBERO HDF5 file
        output_root: Root output directory
        task_name: Name of the task
        task_index: Index of this task
        dhb_invariants: Optional precomputed DHB features
        demo_ids: List of demo IDs to convert (None = all)
        
    Returns:
        Tuple of (num_episodes_converted, total_frames)
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    normalized_task_name = task_name[:-5] if task_name.endswith("_demo") else task_name
    
    meta_dir = output_root / "meta"
    data_dir = output_root / "data"
    meta_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    # Load demo IDs if not provided
    if demo_ids is None:
        demo_ids = list_demos_in_file(libero_h5_path)
    
    logger.info(f"Converting {len(demo_ids)} demos from {libero_h5_path.name}")
    
    # Collect episodes and tasks
    all_episodes = []
    all_tasks = {task_index: task_name}
    total_frames = 0
    geometry_labeler = None

    task_suite_name = libero_h5_path.parent.name
    if HAS_LIBERO and task_suite_name.startswith("libero_"):
        try:
            geometry_labeler = _build_geometry_labeler(task_suite_name, normalized_task_name)
            logger.info(
                "Using geometry-conditioned gripper labels for %s via suite '%s'",
                normalized_task_name,
                task_suite_name,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to transition-based gripper labels for %s: %s",
                task_name,
                exc,
            )
    
    # Convert each demo
    try:
        for local_idx, demo_id in enumerate(demo_ids):
            try:
                global_idx = episode_offset + local_idx
                logger.info(f"  Loading demo {demo_id}...")
                demo_data = load_libero_demo(libero_h5_path, demo_id)

                # Create episode metadata
                episode_meta = create_episode_metadata(global_idx, demo_data["length"], task_index)
                all_episodes.append(episode_meta)

                # Create dataframe
                df = create_episode_dataframe(
                    demo_data,
                    global_idx,
                    task_index,
                    dhb_invariants,
                    geometry_labeler=geometry_labeler,
                )

                # Determine chunk index
                chunk_index = global_idx // LEROBOT_CHUNK_SIZE
                chunk_dir = data_dir / f"{chunk_index:06d}"
                chunk_dir.mkdir(exist_ok=True)

                # Save parquet
                parquet_path = chunk_dir / f"episode_{global_idx:06d}.parquet"
                logger.info(f"  Saving to {parquet_path.relative_to(output_root)}")
                df.to_parquet(parquet_path, index=False)
                write_episode_videos(demo_data, output_root, global_idx)

                total_frames += demo_data["length"]

            except Exception as e:
                logger.error(f"  Error converting demo {demo_id}: {e}")
                continue
    finally:
        if geometry_labeler is not None:
            geometry_labeler.close()
    
    logger.info(f"Successfully converted {len(all_episodes)} episodes ({total_frames} frames)")
    
    return len(all_episodes), total_frames, all_episodes, all_tasks


def write_lerobot_metadata(
    output_root: Path,
    all_episodes: List[Dict],
    all_tasks: Dict[int, str],
    total_frames: int,
) -> None:
    """Write shared LeRobot metadata after all tasks are converted."""
    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing metadata files...")

    modality_metadata = create_modality_metadata()
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality_metadata, f, indent=2)

    info_metadata = create_info_metadata(num_tasks=len(all_tasks))
    info_metadata["num_episodes"] = len(all_episodes)
    info_metadata["num_frames"] = total_frames
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info_metadata, f, indent=2)

    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_meta in all_episodes:
            f.write(json.dumps(ep_meta) + "\n")

    with open(meta_dir / "tasks.jsonl", "w") as f:
        for task_idx, task_name_str in all_tasks.items():
            task_meta = create_task_metadata(task_idx, task_name_str)
            f.write(json.dumps(task_meta) + "\n")


def compute_dataset_statistics(data_dir: Path) -> Dict:
    """Compute statistics for all parquet files in the dataset.
    
    This creates the stats_gr00t.json file needed by StarVLA's dataloader.
    """
    logger.info("Computing dataset statistics...")
    
    stats = defaultdict(lambda: {
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "q01": None,
        "q99": None,
    })
    
    all_data = defaultdict(list)
    
    # Collect all parquet files
    parquet_files = sorted(data_dir.glob("*/episode_*.parquet"))
    
    for pq_path in parquet_files:
        df = pd.read_parquet(pq_path)
        
        # Collect data for each modality
        for col in df.columns:
            if col in ["task_index"]:
                continue
            
            try:
                # Stack all arrays in this column
                col_data = np.vstack([np.asarray(x) for x in df[col]])
                all_data[col].append(col_data)
            except Exception as e:
                logger.warning(f"Could not process column {col}: {e}")
                continue
    
    # Compute statistics
    for col, data_list in all_data.items():
        if not data_list:
            continue
        
        combined = np.concatenate(data_list, axis=0).astype(np.float32)
        
        stats[col]["min"] = np.min(combined, axis=0).tolist()
        stats[col]["max"] = np.max(combined, axis=0).tolist()
        stats[col]["mean"] = np.mean(combined, axis=0).tolist()
        stats[col]["std"] = np.std(combined, axis=0).tolist()
        stats[col]["q01"] = np.quantile(combined, 0.01, axis=0).tolist()
        stats[col]["q99"] = np.quantile(combined, 0.99, axis=0).tolist()
    
    return dict(stats)


# ============================================================================
# Command-line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 datasets to LeRobot v2.0 format for StarVLA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert single task
  python libero_to_lerobot.py \\
      --libero-root /data/libero/spatial \\
      --output-root /data/libero_lerobot \\
      --task-name "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"

  # Convert all tasks in directory
  python libero_to_lerobot.py \\
      --libero-root /data/libero/spatial \\
      --output-root /data/libero_lerobot \\
      --all-tasks

  # With precomputed DHB invariants
  python libero_to_lerobot.py \\
      --libero-root /data/libero/spatial \\
      --output-root /data/libero_lerobot \\
      --all-tasks \\
      --dhb-invariants /path/to/dhb_features.pkl
        """
    )
    
    parser.add_argument(
        "--libero-root",
        type=Path,
        required=True,
        help="Root directory containing LIBERO HDF5 files",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for LeRobot dataset",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Specific task name to convert (HDF5 filename without .hdf5)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Convert all tasks in libero-root",
    )
    parser.add_argument(
        "--dhb-invariants",
        type=Path,
        default=None,
        help="Path to pickle file with precomputed DHB invariants",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Maximum demos to convert per task (for testing)",
    )
    parser.add_argument(
        "--compute-stats",
        action="store_true",
        default=True,
        help="Compute dataset statistics after conversion",
    )
    
    args = parser.parse_args()
    
    # Load DHB invariants if provided
    dhb_invariants = None
    if args.dhb_invariants:
        logger.info(f"Loading DHB invariants from {args.dhb_invariants}")
        with open(args.dhb_invariants, "rb") as f:
            dhb_invariants = pickle.load(f)
    else:
        logger.info("No DHB invariant pickle provided; computing invariants inline from EE trajectories.")
    
    # Determine which tasks to convert
    libero_root = Path(args.libero_root)
    hdf5_files = list(libero_root.glob("*.hdf5"))
    
    if not hdf5_files:
        logger.error(f"No HDF5 files found in {libero_root}")
        sys.exit(1)
    
    if args.task_name:
        # Convert specific task
        matching = [f for f in hdf5_files if args.task_name in f.stem]
        if not matching:
            logger.error(f"No HDF5 file matching task name: {args.task_name}")
            sys.exit(1)
        hdf5_files = matching[:1]
    elif not args.all_tasks:
        logger.error("Specify either --task-name or --all-tasks")
        sys.exit(1)
    
    # Convert each task
    total_episodes = 0
    total_frames = 0
    all_episode_meta = []
    all_task_meta = {}
    
    output_root = Path(args.output_root)
    
    for task_idx, h5_path in enumerate(sorted(hdf5_files)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting task {task_idx}: {h5_path.stem}")
        logger.info(f"{'='*60}")
        
        # Get demo IDs
        demo_ids = list_demos_in_file(h5_path)
        if args.max_demos:
            demo_ids = demo_ids[:args.max_demos]
        
        # Convert
        num_eps, num_frames, episode_meta, task_meta = convert_libero_to_lerobot(
            h5_path,
            output_root,
            task_name=h5_path.stem,
            task_index=task_idx,
            dhb_invariants=dhb_invariants,
            demo_ids=demo_ids,
            episode_offset=total_episodes,
        )
        
        total_episodes += num_eps
        total_frames += num_frames
        all_episode_meta.extend(episode_meta)
        all_task_meta.update(task_meta)

    write_lerobot_metadata(output_root, all_episode_meta, all_task_meta, total_frames)
    
    # Compute statistics
    if args.compute_stats:
        logger.info("\n" + "="*60)
        logger.info("Computing dataset statistics...")
        logger.info("="*60)
        
        stats = compute_dataset_statistics(output_root / "data")
        
        stats_output = {
            "__format_version": 2,
            "__cache_config": {"mode": "abs"},
            "statistics": stats,
        }
        
        stats_path = output_root / "meta" / "stats_gr00t.json"
        with open(stats_path, "w") as f:
            json.dump(stats_output, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_path}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Conversion complete!")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"Output directory: {output_root}")
    logger.info("="*60)
    
    logger.info("\nNext steps:")
    logger.info("1. Verify the dataset can be loaded by StarVLA:")
    logger.info("   python -c \"from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset\"")
    logger.info("2. Update your StarVLA data_config.py to reference this dataset")
    logger.info("3. Train StarVLA with the new dataset")


if __name__ == "__main__":
    main()
