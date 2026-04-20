from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from examples.integration.libero_to_lerobot import (
    _event_class_labels,
    _event_transition_labels,
    create_episode_dataframe,
    load_libero_demo,
)


def test_create_episode_dataframe_uses_axis_angle_state_and_delta_actions(tmp_path: Path):
    h5_path = tmp_path / "demo.hdf5"
    with h5py.File(h5_path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")

        robot_states = np.zeros((2, 9), dtype=np.float32)
        # LIBERO gripper qpos is small-valued (~0.0-0.04), so the converter
        # must not assume a 0/1 threshold.
        robot_states[:, 0] = np.array([0.0, 0.02], dtype=np.float32)
        robot_states[:, 2:5] = np.array([[0.1, 0.2, 0.3], [0.15, 0.18, 0.32]], dtype=np.float32)

        quat0_xyzw = Rotation.from_euler("xyz", [0.0, 0.0, 0.0]).as_quat()
        quat1_xyzw = Rotation.from_euler("xyz", [0.0, 0.0, np.deg2rad(10.0)]).as_quat()
        robot_states[0, 5:9] = quat0_xyzw[[3, 0, 1, 2]]
        robot_states[1, 5:9] = quat1_xyzw[[3, 0, 1, 2]]

        demo.create_dataset("robot_states", data=robot_states)
        demo.create_dataset("actions", data=np.zeros((2, 7), dtype=np.float32))

        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((2, 4, 4, 3), dtype=np.uint8))

    demo_data = load_libero_demo(h5_path, 0)
    df = create_episode_dataframe(demo_data, episode_index=0)

    state0 = np.asarray(df.iloc[0]["observation.state"], dtype=np.float32)
    action0 = np.asarray(df.iloc[0]["action"], dtype=np.float32)
    action1 = np.asarray(df.iloc[1]["action"], dtype=np.float32)

    np.testing.assert_allclose(state0[:3], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(state0[3:6], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(state0[6:], [0.0, 0.0], atol=1e-6)

    np.testing.assert_allclose(action0[:3], [0.05, -0.02, 0.02], atol=1e-6)
    np.testing.assert_allclose(action0[3:6], [0.0, 0.0, np.deg2rad(10.0)], atol=1e-5)
    np.testing.assert_allclose(action0[6], 1.0, atol=1e-6)

    np.testing.assert_allclose(action1[:6], np.zeros(6, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(action1[6], 1.0, atol=1e-6)

    assert "dhb.linear_invariants" in df.columns
    assert "dhb.angular_invariants" in df.columns
    assert "dhb.close_contact" in df.columns
    assert "dhb.open_contact" in df.columns
    assert "dhb.gripper_event" in df.columns
    lin0 = np.asarray(df.iloc[0]["dhb.linear_invariants"], dtype=np.float32)
    ang0 = np.asarray(df.iloc[0]["dhb.angular_invariants"], dtype=np.float32)
    close = np.asarray(df["dhb.close_contact"], dtype=np.float32)
    open_ = np.asarray(df["dhb.open_contact"], dtype=np.float32)
    gripper_event = np.asarray(df["dhb.gripper_event"], dtype=np.int64)
    assert lin0.shape == (4,)
    assert ang0.shape == (4,)
    assert np.isfinite(lin0).all()
    assert np.isfinite(ang0).all()
    np.testing.assert_allclose(close, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(open_, [1.0, 2.0 / 3.0], atol=1e-6)
    np.testing.assert_array_equal(gripper_event, [2, 2])


def test_event_transition_labels_mark_close_and_open_windows():
    gripper_open = np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)

    close = _event_transition_labels(gripper_open, transition="close", radius=1)
    open_ = _event_transition_labels(gripper_open, transition="open", radius=1)

    np.testing.assert_allclose(close, [0.5, 1.0, 0.5, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(open_, [0.0, 0.0, 0.5, 1.0, 0.5], atol=1e-6)


def test_event_class_labels_mark_hold_close_open():
    gripper_open = np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32)
    labels = _event_class_labels(gripper_open, radius=1)
    np.testing.assert_array_equal(labels, [1, 1, 2, 2, 2])
