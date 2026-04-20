import numpy as np

from dhb_xr.integration.vla.chunk_rollout import (
    ChunkRolloutConfig,
    TrajectoryChunkRunner,
    stabilize_chunk_positions,
)


def test_stabilize_chunk_positions_clamps_first_offset_and_step_size():
    chunk = np.array(
        [
            [0.10, 0.00, 0.00],
            [0.16, 0.00, 0.00],
            [0.25, 0.00, 0.00],
        ],
        dtype=np.float32,
    )
    stabilized, debug = stabilize_chunk_positions(
        chunk,
        anchor_pos=np.zeros(3, dtype=np.float32),
        max_first_offset_m=0.04,
        max_step_m=0.02,
        smooth_window=0,
    )

    assert np.isclose(np.linalg.norm(stabilized[0]), 0.04, atol=1e-6)
    steps = np.linalg.norm(np.diff(stabilized, axis=0), axis=1)
    assert np.all(steps <= 0.020001)
    assert debug["raw_first_offset_mm"] > debug["stabilized_first_offset_mm"]


def test_runner_blends_overlap_from_previous_chunk():
    runner = TrajectoryChunkRunner(
        ChunkRolloutConfig(
            blend_overlap_alpha=0.5,
            gripper_source="demo",
        )
    )
    first = runner.prepare_chunk(
        {
            "positions": np.array([[[0.00, 0.00, 0.00], [0.02, 0.00, 0.00], [0.04, 0.00, 0.00]]], dtype=np.float32),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 3], dtype=np.float32),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=3,
        initial_gripper_cmd=-1.0,
        demo_gripper_chunk=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    )
    second = runner.prepare_chunk(
        {
            "positions": np.array([[[0.10, 0.00, 0.00], [0.12, 0.00, 0.00], [0.14, 0.00, 0.00]]], dtype=np.float32),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 3], dtype=np.float32),
        },
        anchor_pos=np.array([0.02, 0.0, 0.0], dtype=np.float32),
        current_start=2,
        n_use=3,
        initial_gripper_cmd=-1.0,
        demo_gripper_chunk=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
    )

    assert first["debug"]["overlap_steps"] == 0
    assert second["debug"]["overlap_steps"] == 1
    assert np.allclose(second["positions_full"][0], np.array([0.07, 0.0, 0.0], dtype=np.float32))


def test_runner_decodes_three_way_contact_probs_to_persistent_gripper_cmds():
    runner = TrajectoryChunkRunner(ChunkRolloutConfig(gripper_source="model"))
    prepared = runner.prepare_chunk(
        {
            "positions": np.array([[[0.00, 0.00, 0.00], [0.01, 0.00, 0.00], [0.02, 0.00, 0.00]]], dtype=np.float32),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 3], dtype=np.float32),
            "pred_contact_probs": np.array(
                [[
                    [0.90, 0.05, 0.05],  # hold
                    [0.05, 0.90, 0.05],  # close
                    [0.05, 0.05, 0.90],  # open
                ]],
                dtype=np.float32,
            ),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=3,
        initial_gripper_cmd=-1.0,
    )

    assert prepared["debug"]["gripper_source_used"] == "contact_probs"
    assert prepared["gripper_cmds"].tolist() == [-1.0, 1.0, -1.0]
