import numpy as np
from scipy.spatial.transform import Rotation

import h5py

from dhb_xr.integration.vla.libero import (
    LiberoAdapter,
    decode_contact_probs_to_gripper_cmds,
    libero_obs_quat_to_axis_angle,
    libero_obs_quat_wxyz_to_xyzw,
)


def test_libero_obs_quat_conversion_matches_hdf5_convention():
    """Raw LIBERO env quaternions should be treated as wxyz, not xyzw."""
    target_euler = np.array([-0.00942506, -0.05467911, -0.00308998], dtype=np.float64)
    target_xyzw = Rotation.from_euler("xyz", target_euler, degrees=False).as_quat()

    # Same orientation as above, but in scalar-first wxyz order as observed from LIBERO env.
    obs_quat_wxyz = np.array([0.9995, -0.0005, -0.0298, -0.0040], dtype=np.float64)

    reordered = libero_obs_quat_wxyz_to_xyzw(obs_quat_wxyz)
    rotvec = libero_obs_quat_to_axis_angle(obs_quat_wxyz)

    reordered = reordered / np.linalg.norm(reordered)
    target_xyzw = target_xyzw / np.linalg.norm(target_xyzw)
    angle_err = 2 * np.arccos(np.clip(abs(np.dot(reordered, target_xyzw)), -1.0, 1.0))

    assert np.degrees(angle_err) < 1.0
    np.testing.assert_allclose(
        Rotation.from_rotvec(rotvec).as_euler("xyz", degrees=False),
        Rotation.from_quat(target_xyzw).as_euler("xyz", degrees=False),
        atol=1e-2,
    )


def test_libero_adapter_loads_binary_gripper_from_small_qpos_values(tmp_path):
    h5_path = tmp_path / "demo.hdf5"
    with h5py.File(h5_path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")

        robot_states = np.zeros((3, 9), dtype=np.float32)
        robot_states[:, 0] = np.array([0.0, 0.02, 0.01], dtype=np.float32)
        robot_states[:, 2:5] = np.array(
            [[0.1, 0.2, 0.3], [0.11, 0.21, 0.31], [0.12, 0.22, 0.32]],
            dtype=np.float32,
        )

        quat_xyzw = Rotation.from_euler("xyz", [0.0, 0.0, 0.0]).as_quat()
        robot_states[:, 5:9] = quat_xyzw[[3, 0, 1, 2]]

        demo.create_dataset("robot_states", data=robot_states)
        obs = demo.create_group("obs")
        obs.create_dataset("ee_pos", data=robot_states[:, 2:5])

    sample = next(LiberoAdapter().load_dataset(str(h5_path)))
    np.testing.assert_allclose(sample["gripper_qpos"], [0.0, 0.02, 0.01], atol=1e-6)
    np.testing.assert_allclose(sample["gripper"], [0.0, 1.0, 0.0], atol=1e-6)


def test_decode_contact_probs_to_gripper_cmds_tracks_event_state():
    contact_probs = np.array(
        [
            [0.1, 0.2],  # remain open
            [0.8, 0.1],  # close event
            [0.2, 0.1],  # stay closed
            [0.1, 0.9],  # open event
            [0.1, 0.2],  # stay open
        ],
        dtype=np.float32,
    )

    cmds = decode_contact_probs_to_gripper_cmds(contact_probs, initial_gripper_cmd=-1.0, threshold=0.5)
    np.testing.assert_allclose(cmds, [-1.0, 1.0, 1.0, -1.0, -1.0], atol=1e-6)


def test_decode_contact_probs_to_gripper_cmds_supports_three_way_events():
    contact_probs = np.array(
        [
            [0.9, 0.05, 0.05],  # hold open
            [0.1, 0.8, 0.1],    # close event
            [0.9, 0.05, 0.05],  # hold closed
            [0.1, 0.1, 0.8],    # open event
            [0.9, 0.05, 0.05],  # hold open
        ],
        dtype=np.float32,
    )
    cmds = decode_contact_probs_to_gripper_cmds(contact_probs, initial_gripper_cmd=-1.0)
    np.testing.assert_allclose(cmds, [-1.0, 1.0, 1.0, -1.0, -1.0], atol=1e-6)


def test_decode_contact_probs_to_gripper_cmds_supports_dense_gripper_state():
    contact_probs = np.array(
        [
            [0.9, 0.1],  # open
            [0.2, 0.8],  # close
            [0.1, 0.9],  # stay closed
            [0.7, 0.3],  # reopen
        ],
        dtype=np.float32,
    )
    cmds = decode_contact_probs_to_gripper_cmds(
        contact_probs,
        initial_gripper_cmd=-1.0,
        threshold=0.5,
        contact_mode="gripper_state",
    )
    np.testing.assert_allclose(cmds, [-1.0, 1.0, 1.0, -1.0], atol=1e-6)


def test_decode_contact_probs_to_gripper_cmds_supports_hysteresis_for_gripper_state():
    contact_probs = np.array(
        [
            [0.45, 0.55],  # stay open: below close threshold
            [0.42, 0.58],  # stay open: still below close threshold
            [0.38, 0.62],  # close
            [0.43, 0.57],  # stay closed in hysteresis band
            [0.61, 0.39],  # reopen
            [0.55, 0.45],  # stay open in hysteresis band
        ],
        dtype=np.float32,
    )
    cmds = decode_contact_probs_to_gripper_cmds(
        contact_probs,
        initial_gripper_cmd=-1.0,
        threshold=0.5,
        contact_mode="gripper_state",
        close_threshold=0.6,
        open_threshold=0.4,
    )
    np.testing.assert_allclose(cmds, [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0], atol=1e-6)
