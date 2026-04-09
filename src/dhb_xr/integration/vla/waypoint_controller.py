"""Waypoint PD controller for LIBERO OSC_POSE action space.

LIBERO's OSC_POSE controller applies variable per-step scaling (20x-577x)
to convert action-space deltas into joint torques. This means a constant
scale factor between DHB-decoded positions and OSC actions will fail --
the robot barely moves or overshoots unpredictably.

Solution: treat DHB-decoded EE positions as *waypoints* and drive toward
each one with proportional control. The OSC controller handles the
low-level torque computation; we just need to produce action-space
commands that point toward the target.

Quaternion convention: this module uses xyzw (scipy / robosuite convention)
for orientation inputs, matching LIBERO's obs keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class WaypointPDController:
    """Drive robot to waypoints using proportional control in OSC action space.

    Instead of replaying precomputed actions (which fails due to variable
    OSC scaling), use target EE positions as waypoints and compute PD-based
    OSC commands.

    Usage::

        ctrl = WaypointPDController(Kp_pos=8.0, Kp_ori=3.0)
        for waypoint_pos, waypoint_ori, gripper_cmd in adapted_trajectory:
            step_count = 0
            while not ctrl.reached(current_ee_pos, waypoint_pos):
                action = ctrl.compute_action(
                    current_ee_pos, current_ee_ori,
                    waypoint_pos, waypoint_ori, gripper_cmd,
                )
                obs, reward, done, info = env.step(action)
                current_ee_pos = obs['ee_pos']
                current_ee_ori = obs['ee_states'][3:7]  # xyzw quat
                step_count += 1
                if step_count >= ctrl.max_steps_per_waypoint:
                    break
    """

    # Proportional gains
    Kp_pos: float = 8.0
    Kp_ori: float = 3.0

    # Waypoint convergence threshold (meters)
    waypoint_threshold: float = 0.015  # 15 mm

    # Orientation convergence threshold (radians)
    ori_threshold: float = 0.1  # ~5.7 degrees

    # Maximum steps to spend on a single waypoint before advancing
    max_steps_per_waypoint: int = 20

    # Action clipping bounds (LIBERO OSC actions are in [-1, 1])
    action_clip: float = 1.0

    def compute_action(
        self,
        current_pos: np.ndarray,
        current_ori: np.ndarray,
        target_pos: np.ndarray,
        target_ori: np.ndarray,
        gripper: float,
    ) -> np.ndarray:
        """Compute a 7-dim OSC action to drive toward the target waypoint.

        Parameters
        ----------
        current_pos : (3,) current end-effector position
        current_ori : (4,) current end-effector orientation quaternion (xyzw)
        target_pos : (3,) target end-effector position
        target_ori : (4,) target end-effector orientation quaternion (xyzw)
        gripper : float, gripper command (pass through from demo timing)

        Returns
        -------
        action : (7,) array [dx, dy, dz, dax, day, daz, gripper]
        """
        current_pos = np.asarray(current_pos, dtype=np.float64).ravel()[:3]
        target_pos = np.asarray(target_pos, dtype=np.float64).ravel()[:3]
        current_ori = np.asarray(current_ori, dtype=np.float64).ravel()[:4]
        target_ori = np.asarray(target_ori, dtype=np.float64).ravel()[:4]

        # --- Position: P control ---
        pos_error = target_pos - current_pos
        pos_action = np.clip(
            self.Kp_pos * pos_error, -self.action_clip, self.action_clip
        )

        # --- Orientation: P control via axis-angle error ---
        ori_action = self._orientation_error(current_ori, target_ori)
        ori_action = np.clip(
            self.Kp_ori * ori_action, -self.action_clip, self.action_clip
        )

        action = np.zeros(7, dtype=np.float64)
        action[:3] = pos_action
        action[3:6] = ori_action
        action[6] = float(gripper)
        return action

    def reached(
        self,
        current_pos: np.ndarray,
        target_pos: np.ndarray,
        threshold: Optional[float] = None,
    ) -> bool:
        """Check if the current position is within threshold of the target.

        Parameters
        ----------
        current_pos : (3,) current end-effector position
        target_pos : (3,) target end-effector position
        threshold : optional override for waypoint_threshold (meters)

        Returns
        -------
        bool : True if within threshold
        """
        if threshold is None:
            threshold = self.waypoint_threshold
        current_pos = np.asarray(current_pos, dtype=np.float64).ravel()[:3]
        target_pos = np.asarray(target_pos, dtype=np.float64).ravel()[:3]
        dist = np.linalg.norm(target_pos - current_pos)
        return dist < threshold

    def reached_full(
        self,
        current_pos: np.ndarray,
        current_ori: np.ndarray,
        target_pos: np.ndarray,
        target_ori: np.ndarray,
    ) -> bool:
        """Check convergence on both position and orientation."""
        if not self.reached(current_pos, target_pos):
            return False
        ori_err = np.linalg.norm(
            self._orientation_error(
                np.asarray(current_ori).ravel()[:4],
                np.asarray(target_ori).ravel()[:4],
            )
        )
        return ori_err < self.ori_threshold

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _orientation_error(
        current_quat_xyzw: np.ndarray,
        target_quat_xyzw: np.ndarray,
    ) -> np.ndarray:
        """Compute orientation error as axis-angle (rotation vector).

        The error is the rotation that takes current -> target, expressed
        in the world frame as a 3-vector (axis * angle).

        Parameters
        ----------
        current_quat_xyzw : (4,) quaternion in xyzw order
        target_quat_xyzw : (4,) quaternion in xyzw order

        Returns
        -------
        error : (3,) axis-angle rotation vector
        """
        R_current = Rotation.from_quat(current_quat_xyzw)
        R_target = Rotation.from_quat(target_quat_xyzw)
        # R_error = R_target * R_current^{-1}  (world-frame error)
        R_error = R_target * R_current.inv()
        return R_error.as_rotvec()

    # ------------------------------------------------------------------
    # Convenience: run a full trajectory
    # ------------------------------------------------------------------

    def execute_trajectory(
        self,
        env,
        waypoint_positions: np.ndarray,
        waypoint_orientations: np.ndarray,
        gripper_commands: np.ndarray,
        obs_pos_key: str = "ee_pos",
        obs_ori_key: str = "ee_states",
        obs_ori_slice: Tuple[int, int] = (3, 7),
        initial_obs: Optional[dict] = None,
        verbose: bool = False,
    ) -> dict:
        """Execute a full waypoint trajectory in a LIBERO environment.

        Parameters
        ----------
        env : LIBERO OffScreenRenderEnv (or compatible)
        waypoint_positions : (N, 3) target EE positions
        waypoint_orientations : (N, 4) target EE orientations (xyzw)
        gripper_commands : (N,) gripper commands
        obs_pos_key : obs dict key for EE position
        obs_ori_key : obs dict key containing orientation data
        obs_ori_slice : slice indices into obs[obs_ori_key] for xyzw quat
        initial_obs : if provided, use as the first observation
        verbose : print per-waypoint info

        Returns
        -------
        dict with keys:
            'total_steps': int, total env steps taken
            'waypoints_reached': int, how many waypoints converged
            'final_reward': float
            'done': bool
            'trajectory_pos': list of observed EE positions
            'trajectory_ori': list of observed EE orientations
        """
        N = len(waypoint_positions)
        assert len(waypoint_orientations) == N
        assert len(gripper_commands) == N

        if initial_obs is not None:
            obs = initial_obs
        else:
            obs = env._get_observations()

        current_pos = np.array(obs[obs_pos_key])
        ori_start, ori_end = obs_ori_slice
        current_ori = np.array(obs[obs_ori_key][ori_start:ori_end])

        total_steps = 0
        waypoints_reached = 0
        final_reward = 0.0
        is_done = False
        traj_pos = [current_pos.copy()]
        traj_ori = [current_ori.copy()]

        for i in range(N):
            wp_pos = waypoint_positions[i]
            wp_ori = waypoint_orientations[i]
            grip = gripper_commands[i]

            steps_this_wp = 0
            for _ in range(self.max_steps_per_waypoint):
                if self.reached(current_pos, wp_pos):
                    waypoints_reached += 1
                    break

                action = self.compute_action(
                    current_pos, current_ori, wp_pos, wp_ori, grip
                )
                obs, reward, done, info = env.step(action)

                current_pos = np.array(obs[obs_pos_key])
                current_ori = np.array(obs[obs_ori_key][ori_start:ori_end])
                traj_pos.append(current_pos.copy())
                traj_ori.append(current_ori.copy())

                total_steps += 1
                steps_this_wp += 1
                final_reward = reward

                if done:
                    is_done = True
                    break
            else:
                # max steps exceeded without reaching waypoint
                if verbose:
                    dist = np.linalg.norm(wp_pos - current_pos)
                    print(
                        f"  Waypoint {i}: not reached after "
                        f"{self.max_steps_per_waypoint} steps "
                        f"(dist={dist:.4f}m)"
                    )

            if is_done:
                if verbose:
                    print(f"  Environment done at waypoint {i}")
                break

        if verbose:
            print(
                f"  Trajectory complete: {total_steps} steps, "
                f"{waypoints_reached}/{N} waypoints reached, "
                f"reward={final_reward}"
            )

        return {
            "total_steps": total_steps,
            "waypoints_reached": waypoints_reached,
            "total_waypoints": N,
            "final_reward": final_reward,
            "done": is_done,
            "trajectory_pos": np.array(traj_pos),
            "trajectory_ori": np.array(traj_ori),
        }
