"""Tests for LIBERO waypoint controller utilities."""

import numpy as np
from scipy.spatial.transform import Rotation

from dhb_xr.integration.vla.waypoint_controller import WaypointPDController


def test_waypoint_controller_clips_position_and_passes_gripper():
    controller = WaypointPDController(Kp_pos=10.0, Kp_ori=1.0, action_clip=0.25)
    identity = np.array([0.0, 0.0, 0.0, 1.0])

    action = controller.compute_action(
        current_pos=np.zeros(3),
        current_ori=identity,
        target_pos=np.array([1.0, -1.0, 0.01]),
        target_ori=identity,
        gripper=-1.0,
    )

    np.testing.assert_allclose(action[:3], [0.25, -0.25, 0.1], atol=1e-6)
    np.testing.assert_allclose(action[3:6], [0.0, 0.0, 0.0], atol=1e-6)
    assert action[6] == -1.0


def test_waypoint_controller_orientation_error_uses_xyzw_quaternions():
    controller = WaypointPDController(Kp_pos=1.0, Kp_ori=2.0, action_clip=10.0)
    current = np.array([0.0, 0.0, 0.0, 1.0])
    target = Rotation.from_rotvec([0.0, 0.0, 0.25]).as_quat()

    action = controller.compute_action(
        current_pos=np.zeros(3),
        current_ori=current,
        target_pos=np.zeros(3),
        target_ori=target,
        gripper=1.0,
    )

    np.testing.assert_allclose(action[3:6], [0.0, 0.0, 0.5], atol=1e-6)


def test_waypoint_controller_reached_full_checks_position_and_orientation():
    controller = WaypointPDController(waypoint_threshold=0.02, ori_threshold=0.1)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    near_rot = Rotation.from_rotvec([0.0, 0.0, 0.05]).as_quat()
    far_rot = Rotation.from_rotvec([0.0, 0.0, 0.2]).as_quat()

    assert controller.reached_full(np.zeros(3), identity, np.array([0.01, 0.0, 0.0]), near_rot)
    assert not controller.reached_full(np.zeros(3), identity, np.array([0.03, 0.0, 0.0]), near_rot)
    assert not controller.reached_full(np.zeros(3), identity, np.array([0.01, 0.0, 0.0]), far_rot)
