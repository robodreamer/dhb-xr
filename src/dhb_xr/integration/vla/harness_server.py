"""DHB-XR model server for the AllenAI vla-evaluation-harness.

Usage:
    # Start server
    vla-eval serve --config configs/model_servers/dhbxr.yaml

    # Or run directly
    python harness_server.py --demo-dataset /path/to/libero_spatial --chunk-size 8

Pipeline per observation:
    obs (ee_pos + ee_quat) → LiberoAdapter.encode() → DHB invariant tokens
    → FatropSolver.adapt(new_pose, 7ms) → decode_dhb_dr() → SE(3) trajectory
    → embodik.VelocityIK → joint commands → {"actions": (chunk_size, 7)}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

# vla_eval imports (available when running inside the harness environment)
try:
    from vla_eval.model_servers.predict import PredictModelServer
    from vla_eval.model_servers.base import SessionContext
    from vla_eval.specs import (
        DimSpec,
        POSITION_DELTA,
        ROTATION_EULER,
        GRIPPER_01,
        IMAGE_RGB,
        LANGUAGE,
    )
    from vla_eval.types import Action, Observation
    HAS_HARNESS = True
except ImportError:
    # Stub base class for development without harness installed
    class PredictModelServer:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            pass

    class SessionContext:  # type: ignore[no-redef]
        session_id: str = "dev"
        episode_id: str = "dev"
        step: int = 0
        is_first: bool = True

    HAS_HARNESS = False

# DHB-XR imports
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.core.geometry import rot_to_quat, quat_to_rot

try:
    from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop
    HAS_FATROP = True
except ImportError:
    HAS_FATROP = False

try:
    from dhb_xr.optimization.casadi_solver import generate_trajectory_casadi
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

try:
    import embodik as eik
    HAS_EMBODIK = True
except ImportError:
    HAS_EMBODIK = False

logger = logging.getLogger(__name__)


class DHBModelServer(PredictModelServer):
    """DHB-XR retrieval-based model server for AllenAI vla-evaluation-harness.

    Strategy: retrieval + geometric adaptation.
    1. On episode start: load the reference demo for this task.
    2. On each observation: adapt the reference trajectory to current robot pose.
    3. Return adapted joint commands.

    This directly validates the core CoRL 2026 claim:
    "encode once in invariant space, adapt geometrically at 7ms".

    Args:
        demo_dataset: path to LIBERO dataset directory (*.hdf5 files per task)
        chunk_size: number of action steps to predict per inference call
        solver: "fatrop" | "casadi" | "auto" — geometric adaptation solver
        use_embodik_ik: whether to convert SE(3) trajectory to joint velocities via embodik
        action_dim: output action dimension (default 7: 6-DOF + gripper)
        verbose: print solver timing information
    """

    def __init__(
        self,
        demo_dataset: str = "",
        chunk_size: int = 8,
        solver: str = "auto",
        use_embodik_ik: bool = False,
        action_dim: int = 7,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, **kwargs)
        self.demo_dataset = Path(demo_dataset) if demo_dataset else None
        self.solver = solver
        self.use_embodik_ik = use_embodik_ik and HAS_EMBODIK
        self.action_dim = action_dim
        self.verbose = verbose

        # Per-episode state (reset in on_episode_start)
        self._ref_positions: np.ndarray | None = None   # (N, 3)
        self._ref_quaternions: np.ndarray | None = None  # (N, 4) xyzw
        self._ref_lin_inv: np.ndarray | None = None      # (N-1, 4)
        self._ref_ang_inv: np.ndarray | None = None      # (N-1, 4)
        self._ref_initial_pose: dict | None = None
        self._adapted_trajectory: np.ndarray | None = None  # (N, 7) joint actions
        self._traj_cursor: int = 0

        # embodik IK solver (lazy init)
        self._ik_solver: Any = None

        # Demo cache: task_name → {positions, quaternions}
        self._demo_cache: dict[str, dict] = {}

        solver_info = "fatrop" if HAS_FATROP else ("casadi" if HAS_CASADI else "encode-decode")
        logger.info(
            "DHBModelServer initialized | solver=%s | embodik=%s",
            solver_info,
            "yes" if self.use_embodik_ik else "no",
        )

    # ── Harness API ───────────────────────────────────────────────────────────

    def get_action_spec(self) -> dict:
        if not HAS_HARNESS:
            return {}
        return {
            "position": POSITION_DELTA,
            "rotation": ROTATION_EULER,
            "gripper": GRIPPER_01,
        }

    def get_observation_spec(self) -> dict:
        if not HAS_HARNESS:
            return {}
        return {
            "agentview_rgb": IMAGE_RGB,
            "language": LANGUAGE,
            "state": DimSpec("state", 9, "raw"),  # ee_pos(3) + ee_quat(4) + gripper(2)
        }

    def get_observation_params(self) -> dict:
        return {
            "image_keys": ["agentview_rgb", "robot0_eye_in_hand_rgb"],
            "state_keys": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        }

    async def on_episode_start(
        self, config: dict[str, Any], ctx: SessionContext
    ) -> None:
        """Load and encode reference demo for this episode's task."""
        await super().on_episode_start(config, ctx)

        task_name = config.get("task_name", config.get("task_id", "task_0"))
        self._traj_cursor = 0
        self._adapted_trajectory = None

        demo = self._load_demo(task_name)
        if demo is None:
            logger.warning("No demo found for task '%s' — returning zeros", task_name)
            self._ref_positions = None
            return

        self._ref_positions = demo["positions"]
        self._ref_quaternions = demo["quaternions"]

        # Encode to DHB invariants once
        enc = encode_dhb_dr(self._ref_positions, self._ref_quaternions)
        self._ref_lin_inv = enc["linear_invariants"]
        self._ref_ang_inv = enc["angular_invariants"]
        self._ref_initial_pose = enc["initial_pose"]

        logger.info(
            "Episode '%s': encoded %d-step demo into DHB invariants",
            task_name,
            len(self._ref_positions),
        )

    def predict(self, obs: dict, ctx: SessionContext) -> dict:
        """Adapt reference trajectory to current robot pose and return next actions.

        On first step: runs Fatrop adaptation (~7ms) to get adapted SE(3) trajectory,
        then converts to joint actions via embodik IK or direct EEF delta.
        On subsequent steps: steps through the pre-adapted trajectory.
        """
        if self._ref_positions is None:
            return {"actions": np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)}

        # Extract current robot state
        current_pos, current_quat = self._extract_state(obs)

        # On first step: adapt the full trajectory to current robot pose
        if ctx.is_first or self._adapted_trajectory is None:
            self._adapted_trajectory = self._adapt_and_convert(current_pos, current_quat)
            self._traj_cursor = 0

        # Return next chunk of actions
        start = self._traj_cursor
        end = min(start + self.chunk_size, len(self._adapted_trajectory))
        chunk = self._adapted_trajectory[start:end]

        # Pad with last action if chunk is shorter than expected
        if len(chunk) < self.chunk_size:
            pad = np.tile(chunk[-1:], (self.chunk_size - len(chunk), 1))
            chunk = np.concatenate([chunk, pad], axis=0)

        self._traj_cursor = end

        return {"actions": chunk.astype(np.float32)}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_state(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Extract current ee_pos (3,) and ee_quat (4, xyzw) from observation."""
        state = obs.get("state", obs.get("robot_state", None))
        if state is not None and len(state) >= 7:
            return np.array(state[:3]), np.array(state[3:7])

        # Fallback: extract from separate keys
        pos = obs.get("robot0_eef_pos", np.array([0.0, 0.0, 0.0]))
        quat = obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0]))  # wxyz
        # Convert wxyz → xyzw for DHB
        if abs(quat[0]) > abs(quat[3]):  # likely wxyz format
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        return np.array(pos), np.array(quat)

    def _adapt_and_convert(
        self, start_pos: np.ndarray, start_quat: np.ndarray
    ) -> np.ndarray:
        """Run Fatrop adaptation and return (N, action_dim) joint action sequence."""
        goal_pos = self._ref_positions[-1]
        goal_quat = self._ref_quaternions[-1]

        adapted = self._run_solver(
            self._ref_positions,
            self._ref_quaternions,
            start_pos,
            start_quat,
            goal_pos,
            goal_quat,
        )

        adapted_pos = adapted["positions"]   # (N, 3)
        adapted_quat = adapted["quaternions"]  # (N, 4) xyzw

        if self.verbose:
            logger.info("Adaptation: solver=%s, time=%.1fms", adapted.get("solver", "?"),
                        adapted.get("solve_time", 0) * 1000)

        if self.use_embodik_ik:
            return self._ik_to_joints(adapted_pos, adapted_quat)

        # Return EEF delta actions (fallback: no IK)
        return self._eef_to_delta_actions(adapted_pos, adapted_quat)

    def _run_solver(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        new_start_pos: np.ndarray,
        new_start_quat: np.ndarray,
        new_goal_pos: np.ndarray,
        new_goal_quat: np.ndarray,
    ) -> dict:
        """Run Fatrop → CasADi → encode-decode fallback chain."""
        pose_init = {"position": new_start_pos, "quaternion": new_start_quat}
        pose_final = {"position": new_goal_pos, "quaternion": new_goal_quat}

        if HAS_FATROP and self.solver in ("fatrop", "auto"):
            try:
                return generate_trajectory_fatrop(
                    pos, quat, pose_init, pose_final, traj_length=len(pos)
                )
            except Exception as e:
                logger.debug("Fatrop failed: %s — falling back", e)

        if HAS_CASADI and self.solver in ("casadi", "auto"):
            try:
                return generate_trajectory_casadi(
                    pos, quat, pose_init, pose_final, traj_length=len(pos)
                )
            except Exception as e:
                logger.debug("CasADi failed: %s — falling back to decode", e)

        # Last resort: decode from reference invariants at new start pose
        dec = decode_dhb_dr(
            self._ref_lin_inv,
            self._ref_ang_inv,
            {"position": new_start_pos, "quaternion": new_start_quat},
        )
        return {
            "positions": dec["positions"],
            "quaternions": dec["quaternions"],
            "solver": "decode",
        }

    def _eef_to_delta_actions(
        self, positions: np.ndarray, quaternions: np.ndarray
    ) -> np.ndarray:
        """Convert SE(3) trajectory to delta EEF actions (7-dim: dx,dy,dz,dRx,dRy,dRz,gripper)."""
        N = len(positions)
        actions = np.zeros((N, self.action_dim), dtype=np.float32)

        for i in range(N - 1):
            actions[i, :3] = positions[i + 1] - positions[i]  # delta position
            # Delta rotation as axis-angle (simplified)
            R_curr = quat_to_rot(quaternions[i])
            R_next = quat_to_rot(quaternions[i + 1])
            R_delta = R_next @ R_curr.T
            # Rodrigues formula for axis-angle
            theta = np.arccos(np.clip((np.trace(R_delta) - 1) / 2, -1, 1))
            if abs(theta) > 1e-8:
                axis = np.array([
                    R_delta[2, 1] - R_delta[1, 2],
                    R_delta[0, 2] - R_delta[2, 0],
                    R_delta[1, 0] - R_delta[0, 1],
                ]) / (2 * np.sin(theta))
                actions[i, 3:6] = axis * theta
            actions[i, 6] = 1.0  # gripper open by default

        actions[-1] = actions[-2]  # duplicate last
        return actions

    def _ik_to_joints(
        self, positions: np.ndarray, quaternions: np.ndarray
    ) -> np.ndarray:
        """Convert SE(3) trajectory to joint velocities via embodik IK."""
        if not HAS_EMBODIK:
            return self._eef_to_delta_actions(positions, quaternions)

        if self._ik_solver is None:
            robot = eik.RobotModel("panda", floating_base=False)
            self._ik_solver = eik.KinematicsSolver(robot)

        N = len(positions)
        joint_actions = np.zeros((N, self.action_dim), dtype=np.float32)

        q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # Panda home

        for i in range(N):
            target_pos = positions[i]
            target_quat = quaternions[i]  # xyzw
            # embodik expects wxyz
            target_quat_wxyz = np.array([target_quat[3], target_quat[0],
                                          target_quat[1], target_quat[2]])
            try:
                result = self._ik_solver.solve_position_ik(
                    q, target_pos, target_quat_wxyz, frame="panda_hand"
                )
                if result is not None:
                    q = result
            except Exception:
                pass
            joint_actions[i, :7] = q[:7] if len(q) >= 7 else q

        return joint_actions

    def _load_demo(self, task_name: str) -> dict | None:
        """Load reference demo trajectory for a task (with caching)."""
        if task_name in self._demo_cache:
            return self._demo_cache[task_name]

        if self.demo_dataset is None or not self.demo_dataset.exists():
            return None

        # Find matching HDF5 file
        candidates = list(self.demo_dataset.glob(f"*{task_name}*.hdf5"))
        if not candidates:
            candidates = sorted(self.demo_dataset.glob("*.hdf5"))

        if not candidates:
            return None

        hdf5_path = candidates[0]
        try:
            import h5py
            from dhb_xr.integration.vla.libero import LiberoAdapter

            adapter = LiberoAdapter()
            demos = list(adapter.load_dataset(str(hdf5_path)))
            if not demos:
                return None

            demo = demos[0]
            self._demo_cache[task_name] = demo
            return demo
        except Exception as e:
            logger.warning("Failed to load demo from %s: %s", hdf5_path, e)
            return None


# ── Config generation helper ──────────────────────────────────────────────────

def write_harness_config(output_path: str = "configs/model_servers/dhbxr.yaml") -> None:
    """Write a template harness config for DHBModelServer."""
    config = {
        "script": "src/dhb_xr/integration/vla/harness_server.py",
        "args": {
            "demo_dataset": "/path/to/libero_spatial",
            "chunk_size": 8,
            "solver": "auto",
            "use_embodik_ik": False,
            "action_dim": 7,
            "verbose": False,
        },
    }
    import yaml
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config written to {output_path}")


if __name__ == "__main__":
    if HAS_HARNESS:
        from vla_eval.model_servers.serve import run_server
        run_server(DHBModelServer)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="DHB-XR harness server (dev mode)")
        parser.add_argument("--write-config", action="store_true")
        parser.add_argument("--config-path", default="configs/model_servers/dhbxr.yaml")
        args = parser.parse_args()
        if args.write_config:
            write_harness_config(args.config_path)
        else:
            print("vla_eval not installed. Install AllenAI harness or run with --write-config")
