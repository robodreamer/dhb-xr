from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.integration.libero_to_lerobot import (
    _event_class_labels,
    _event_transition_labels,
    _gripper_state_from_events,
    _geometry_contact_labels,
    _infer_close_affordance_target,
    _object_minus_target_from_eef_relative,
    _object_target_release_features,
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
    assert "dhb.gripper_state" in df.columns
    assert "dhb.gripper_event" in df.columns
    assert "dhb.gripper_phase" in df.columns
    lin0 = np.asarray(df.iloc[0]["dhb.linear_invariants"], dtype=np.float32)
    ang0 = np.asarray(df.iloc[0]["dhb.angular_invariants"], dtype=np.float32)
    close = np.asarray(df["dhb.close_contact"], dtype=np.float32)
    open_ = np.asarray(df["dhb.open_contact"], dtype=np.float32)
    gripper_state = np.asarray(df["dhb.gripper_state"], dtype=np.int64)
    gripper_event = np.asarray(df["dhb.gripper_event"], dtype=np.int64)
    gripper_phase = np.asarray(df["dhb.gripper_phase"], dtype=np.int64)
    assert lin0.shape == (4,)
    assert ang0.shape == (4,)
    assert np.isfinite(lin0).all()
    assert np.isfinite(ang0).all()
    np.testing.assert_allclose(close, [0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(open_, [1.0, 2.0 / 3.0], atol=1e-6)
    np.testing.assert_array_equal(gripper_state, [0, 0])
    np.testing.assert_array_equal(gripper_event, [2, 2])
    np.testing.assert_array_equal(gripper_phase, [2, 2])


def test_create_episode_dataframe_exports_offline_geometry_features(tmp_path: Path):
    h5_path = tmp_path / "demo_geometry.hdf5"
    with h5py.File(h5_path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")

        robot_states = np.zeros((3, 9), dtype=np.float32)
        robot_states[:, 0] = np.array([0.02, 0.0, 0.0], dtype=np.float32)
        robot_states[:, 2:5] = np.array(
            [[0.1, 0.2, 0.3], [0.12, 0.21, 0.31], [0.14, 0.23, 0.32]],
            dtype=np.float32,
        )
        robot_states[:, 5] = 1.0
        demo.create_dataset("robot_states", data=robot_states)
        demo.create_dataset("actions", data=np.zeros((3, 7), dtype=np.float32))
        demo.create_dataset("states", data=np.zeros((3, 4), dtype=np.float32))

        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))

    class DummyGeometryLabeler:
        def build_features(self, sim_states, gripper_open):
            assert sim_states.shape == (3, 4)
            object_rel = np.array(
                [[0.10, 0.00, 0.02], [0.05, 0.01, 0.02], [0.02, 0.01, 0.02]],
                dtype=np.float32,
            )
            target_rel = np.array(
                [[0.20, 0.02, 0.03], [0.15, 0.02, 0.03], [0.08, 0.01, 0.02]],
                dtype=np.float32,
            )
            return {
                "close_contact": np.array([0, 1, 0], dtype=np.float32),
                "open_contact": np.array([0, 0, 1], dtype=np.float32),
                "gripper_event": np.array([0, 1, 2], dtype=np.int64),
                "gripper_phase": np.array([0, 1, 2], dtype=np.int64),
                "gripper_state": np.array([0, 1, 0], dtype=np.int64),
                "object_to_eef_pos": object_rel,
                "target_to_eef_pos": target_rel,
                "object_minus_target_pos": _object_minus_target_from_eef_relative(
                    object_rel=object_rel,
                    target_rel=target_rel,
                ),
                "transport_phase": np.array([0, 1, 0], dtype=np.float32),
                "object_release_ready": np.array([0, 0, 1], dtype=np.float32),
                "object_release_contact": np.array([0, 0, 1], dtype=np.float32),
                "object_release_event": np.array([0, 1, 2], dtype=np.int64),
                "object_release_phase": np.array([0, 1, 2], dtype=np.int64),
                "object_release_gripper_state": np.array([0, 1, 0], dtype=np.int64),
                "object_hold_transport": np.array([0, 1, 0], dtype=np.float32),
            }

    demo_data = load_libero_demo(h5_path, 0)
    df = create_episode_dataframe(
        demo_data,
        episode_index=0,
        geometry_labeler=DummyGeometryLabeler(),
    )

    assert "dhb.object_to_eef_pos" in df.columns
    assert "dhb.target_to_eef_pos" in df.columns
    assert "dhb.object_minus_target_pos" in df.columns
    assert "dhb.transport_phase" in df.columns
    assert "dhb.object_release_ready" in df.columns
    assert "dhb.object_release_contact" in df.columns
    assert "dhb.object_release_event" in df.columns
    assert "dhb.object_release_phase" in df.columns
    assert "dhb.object_release_gripper_state" in df.columns
    assert "dhb.object_hold_transport" in df.columns
    np.testing.assert_allclose(
        np.stack(df["dhb.object_minus_target_pos"].map(np.asarray)),
        [[-0.10, -0.02, -0.01], [-0.10, -0.01, -0.01], [-0.06, 0.0, 0.0]],
        atol=1e-6,
    )
    np.testing.assert_array_equal(np.asarray(df["dhb.gripper_state"], dtype=np.int64), [0, 1, 0])
    np.testing.assert_array_equal(np.asarray(df["dhb.object_release_event"], dtype=np.int64), [0, 1, 2])
    np.testing.assert_array_equal(np.asarray(df["dhb.object_release_phase"], dtype=np.int64), [0, 1, 2])
    np.testing.assert_array_equal(
        np.asarray(df["dhb.object_release_gripper_state"], dtype=np.int64),
        [0, 1, 0],
    )


def test_create_episode_dataframe_exports_dhb_oat_token_targets(tmp_path: Path):
    pytest.importorskip("torch")

    from dhb_xr.integration.vla.pipeline import DHBVLAPipeline, DHBVLAPipelineConfig

    h5_path = tmp_path / "demo_oat.hdf5"
    with h5py.File(h5_path, "w") as f:
        data = f.create_group("data")
        demo = data.create_group("demo_0")

        robot_states = np.zeros((5, 9), dtype=np.float32)
        robot_states[:, 0] = np.array([0.0, 0.02, 0.02, 0.02, 0.02], dtype=np.float32)
        robot_states[:, 2:5] = np.array(
            [
                [0.10, 0.20, 0.30],
                [0.12, 0.22, 0.31],
                [0.15, 0.23, 0.33],
                [0.18, 0.24, 0.35],
                [0.20, 0.26, 0.36],
            ],
            dtype=np.float32,
        )

        quat_xyzw = Rotation.from_euler(
            "xyz",
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, np.deg2rad(5.0)],
                [0.0, 0.0, np.deg2rad(10.0)],
                [0.0, 0.0, np.deg2rad(15.0)],
                [0.0, 0.0, np.deg2rad(20.0)],
            ],
        ).as_quat()
        robot_states[:, 5:9] = quat_xyzw[:, [3, 0, 1, 2]]

        demo.create_dataset("robot_states", data=robot_states)
        demo.create_dataset("actions", data=np.zeros((5, 7), dtype=np.float32))

        obs = demo.create_group("obs")
        obs.create_dataset("agentview_rgb", data=np.zeros((5, 4, 4, 3), dtype=np.uint8))

    demo_data = load_libero_demo(h5_path, 0)
    pipeline = DHBVLAPipeline(
        DHBVLAPipelineConfig(
            tokenizer_kind="oat",
            oat_num_registers=4,
            oat_emb_dim=32,
            oat_num_heads=4,
            oat_encoder_depth=1,
            oat_decoder_depth=1,
            oat_init_seed=0,
        )
    )
    df = create_episode_dataframe(
        demo_data,
        episode_index=0,
        dhb_oat_pipeline=pipeline,
        dhb_oat_action_horizon=4,
    )

    assert "dhb.oat_token_ids" in df.columns
    assert "dhb.oat_chunk_horizon" in df.columns
    assert "dhb.oat_chunk_step_mask" in df.columns

    combined = np.stack(
        [
            np.concatenate(
                [
                    np.asarray(df.iloc[i]["dhb.linear_invariants"], dtype=np.float32),
                    np.asarray(df.iloc[i]["dhb.angular_invariants"], dtype=np.float32),
                ]
            )
            for i in range(len(df))
        ],
        axis=0,
    )
    expected = pipeline.build_oat_token_targets(combined, chunk_horizon=4)

    token_ids = np.stack(df["dhb.oat_token_ids"].map(lambda x: np.asarray(x, dtype=np.int64)), axis=0)
    chunk_horizon = np.asarray(df["dhb.oat_chunk_horizon"], dtype=np.int32)
    step_mask = np.stack(df["dhb.oat_chunk_step_mask"].map(lambda x: np.asarray(x, dtype=np.float32)), axis=0)

    np.testing.assert_array_equal(token_ids, expected["token_ids"])
    np.testing.assert_array_equal(chunk_horizon, np.array([4, 4, 3, 2, 1], dtype=np.int32))
    np.testing.assert_array_equal(chunk_horizon, expected["chunk_horizon"])
    np.testing.assert_array_equal(step_mask, expected["chunk_step_mask"])
    np.testing.assert_array_equal(step_mask[-1], np.array([1, 0, 0, 0], dtype=np.float32))


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


def test_geometry_contact_labels_follow_object_relative_gates():
    gripper_open = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    bowl_rel = np.array(
        [
            [0.08, 0.00, 0.08],   # too far to close
            [0.01, -0.02, 0.03],  # close-ready while open
            [0.00, -0.01, 0.02],  # still near bowl, but already closed
            [0.10, 0.00, 0.10],   # far from bowl
        ],
        dtype=np.float32,
    )
    plate_rel = np.array(
        [
            [0.20, 0.00, 0.10],   # far from plate
            [0.18, 0.00, 0.10],   # far from plate
            [0.03, 0.02, 0.05],   # release-ready while closed
            [0.02, 0.01, 0.04],   # still release-ready while closed
        ],
        dtype=np.float32,
    )

    close_contact, open_contact, gripper_event, gripper_phase = _geometry_contact_labels(
        bowl_rel=bowl_rel,
        plate_rel=plate_rel,
        gripper_open=gripper_open,
    )

    np.testing.assert_array_equal(close_contact, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(open_contact, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(gripper_event, [0, 1, 2, 0])
    np.testing.assert_array_equal(gripper_phase, [0, 1, 2, 2])


def test_geometry_contact_labels_use_short_phase_windows_after_onset():
    gripper_open = np.ones(10, dtype=np.float32)
    bowl_rel = np.array(
        [
            [0.08, 0.00, 0.08],  # not close-ready yet
            [0.01, 0.00, 0.03],
            [0.01, 0.00, 0.03],
            [0.01, 0.00, 0.03],
            [0.01, 0.00, 0.03],
            [0.01, 0.00, 0.03],
            [0.01, 0.00, 0.03],
            [0.01, 0.00, 0.03],
            [0.01, 0.00, 0.03],
            [0.08, 0.00, 0.08],  # leaves the region later
        ],
        dtype=np.float32,
    )
    plate_rel = np.full((10, 3), 0.20, dtype=np.float32)

    close_contact, open_contact, gripper_event, gripper_phase = _geometry_contact_labels(
        bowl_rel=bowl_rel,
        plate_rel=plate_rel,
        gripper_open=gripper_open,
        close_phase_steps=6,
    )

    np.testing.assert_array_equal(close_contact, [0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(open_contact, np.zeros(10, dtype=np.float32))
    np.testing.assert_array_equal(gripper_event, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(gripper_phase, [0, 1, 1, 1, 1, 1, 1, 0, 0, 0])


def test_close_affordance_target_uses_physical_close_vector() -> None:
    gripper_open = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    bowl_rel = np.array(
        [
            [0.00, 0.00, 0.02],
            [0.01, 0.03, -0.01],
            [0.02, 0.05, -0.03],
            [0.03, 0.05, -0.03],
        ],
        dtype=np.float32,
    )

    target = _infer_close_affordance_target(
        bowl_rel=bowl_rel,
        gripper_open=gripper_open,
    )

    np.testing.assert_allclose(target, [0.02, 0.05, -0.03], atol=1e-6)


def test_geometry_contact_labels_can_target_side_grasp_affordance() -> None:
    gripper_open = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    bowl_rel = np.array(
        [
            [0.00, 0.00, 0.02],   # legacy center contact, but not the side grasp
            [0.01, 0.045, -0.025],
            [0.02, 0.050, -0.030],
            [0.02, 0.050, -0.030],
            [0.00, 0.00, 0.02],
        ],
        dtype=np.float32,
    )
    plate_rel = np.full((5, 3), 0.20, dtype=np.float32)

    close_contact, open_contact, gripper_event, gripper_phase = _geometry_contact_labels(
        bowl_rel=bowl_rel,
        plate_rel=plate_rel,
        gripper_open=gripper_open,
        close_affordance_target_m=np.array([0.02, 0.05, -0.03], dtype=np.float32),
        close_affordance_tolerance_m=0.013,
    )

    np.testing.assert_array_equal(close_contact, [0, 1, 1, 0, 0])
    np.testing.assert_array_equal(open_contact, np.zeros(5, dtype=np.float32))
    np.testing.assert_array_equal(gripper_event, [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(gripper_phase, [0, 1, 1, 0, 0])


def test_gripper_state_from_events_tracks_geometry_aligned_state():
    gripper_open = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    close_contact = np.array([0, 1, 1, 0, 0, 0], dtype=np.float32)
    open_contact = np.array([0, 0, 0, 0, 1, 1], dtype=np.float32)

    gripper_state = _gripper_state_from_events(
        gripper_open=gripper_open,
        close_contact=close_contact,
        open_contact=open_contact,
    )

    np.testing.assert_array_equal(gripper_state, [0, 1, 1, 1, 0, 0])


def test_object_target_release_features_delay_until_sustained_target_contact():
    object_minus_target = np.array(
        [
            [0.12, 0.00, 0.00],
            [0.10, 0.00, 0.00],
            [0.04, 0.00, 0.00],
            [0.03, 0.00, 0.00],
            [0.02, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.01, 0.00, 0.00],
            [0.08, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    close_contact = np.array([0, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)

    features = _object_target_release_features(
        object_minus_target_pos=object_minus_target,
        close_contact=close_contact,
        release_gate_xy_m=0.05,
        release_gate_z_m=0.02,
        ready_sustain_steps=3,
        release_phase_steps=2,
    )

    np.testing.assert_array_equal(features["object_release_ready"], [0, 0, 0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(features["object_release_contact"], [0, 0, 0, 0, 1, 1, 0, 0])
    np.testing.assert_array_equal(features["object_release_event"], [0, 1, 0, 0, 2, 0, 0, 0])
    np.testing.assert_array_equal(features["object_release_phase"], [0, 1, 1, 0, 2, 2, 0, 0])
    np.testing.assert_array_equal(features["object_release_gripper_state"], [0, 1, 1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(features["object_hold_transport"], [0, 1, 1, 1, 0, 0, 0, 0])
