from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from examples.integration.libero_to_lerobot import create_episode_dataframe, load_libero_demo


def test_create_episode_dataframe_uses_axis_angle_state_and_delta_actions(tmp_path: Path):
    h5_path = tmp_path / "demo.hdf5"
    with h5py.File(h5_path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")

        robot_states = np.zeros((2, 9), dtype=np.float32)
        robot_states[:, 0] = np.array([0.0, 1.0], dtype=np.float32)
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
