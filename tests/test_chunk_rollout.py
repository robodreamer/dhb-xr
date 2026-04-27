import numpy as np

from dhb_xr.integration.vla.chunk_rollout import (
    ChunkRolloutConfig,
    PhaseAwareChunkRunner,
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


def test_runner_records_event_gripper_thresholds_in_debug():
    runner = TrajectoryChunkRunner(
        ChunkRolloutConfig(
            gripper_source="model",
            model_gripper_decode_source="contact_phase",
            gripper_close_threshold=0.7,
            gripper_open_threshold=0.6,
        )
    )
    prepared = runner.prepare_chunk(
        {
            "positions": np.array([[[0.00, 0.00, 0.00], [0.01, 0.00, 0.00]]], dtype=np.float32),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 2], dtype=np.float32),
            "pred_contact_phase_probs": np.array(
                [[
                    [0.95, 0.04, 0.01],
                    [0.20, 0.75, 0.05],
                ]],
                dtype=np.float32,
            ),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=2,
        initial_gripper_cmd=-1.0,
    )

    assert prepared["debug"]["gripper_source_used"] == "contact_phase_probs"
    assert prepared["debug"]["gripper_close_threshold"] == 0.7
    assert prepared["debug"]["gripper_open_threshold"] == 0.6


def test_runner_debounces_dense_gripper_state_predictions():
    runner = TrajectoryChunkRunner(
        ChunkRolloutConfig(
            gripper_source="model",
            gripper_close_threshold=0.6,
            gripper_open_threshold=0.4,
        )
    )
    prepared = runner.prepare_chunk(
        {
            "positions": np.array([[[0.00, 0.00, 0.00], [0.01, 0.00, 0.00], [0.02, 0.00, 0.00], [0.03, 0.00, 0.00]]], dtype=np.float32),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 4], dtype=np.float32),
            "pred_gripper_state_probs": np.array(
                [[
                    [0.45, 0.55],  # stay open
                    [0.38, 0.62],  # close
                    [0.43, 0.57],  # remain closed in hysteresis band
                    [0.61, 0.39],  # reopen
                ]],
                dtype=np.float32,
            ),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=4,
        initial_gripper_cmd=-1.0,
    )

    assert prepared["debug"]["gripper_source_used"] == "gripper_state_probs"
    assert prepared["gripper_cmds"].tolist() == [-1.0, 1.0, 1.0, -1.0]
    assert prepared["debug"]["gripper_close_threshold"] == 0.6
    assert prepared["debug"]["gripper_open_threshold"] == 0.4


def test_runner_hybrid_contact_ready_closes_and_dense_state_reopens():
    runner = TrajectoryChunkRunner(
        ChunkRolloutConfig(
            gripper_source="model",
            model_gripper_decode_source="contact_gripper_state",
            gripper_close_threshold=0.6,
            gripper_open_threshold=0.4,
            gripper_transition_debounce_steps=2,
        )
    )
    prepared = runner.prepare_chunk(
        {
            "positions": np.array(
                [[[0.00, 0.00, 0.00], [0.01, 0.00, 0.00], [0.02, 0.00, 0.00], [0.03, 0.00, 0.00], [0.04, 0.00, 0.00]]],
                dtype=np.float32,
            ),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 5], dtype=np.float32),
            "contact_mode": "gripper_state",
            "pred_contact_probs": np.array(
                [[
                    [0.90, 0.10],
                    [0.35, 0.65],
                    [0.34, 0.66],
                    [0.95, 0.05],
                    [0.95, 0.05],
                ]],
                dtype=np.float32,
            ),
            "pred_gripper_state_probs": np.array(
                [[
                    [0.95, 0.05],
                    [0.90, 0.10],
                    [0.10, 0.90],
                    [0.70, 0.30],
                    [0.72, 0.28],
                ]],
                dtype=np.float32,
            ),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=5,
        initial_gripper_cmd=-1.0,
    )

    assert prepared["debug"]["gripper_source_used"] == "contact_gripper_state_probs"
    assert prepared["gripper_cmds"].tolist() == [-1.0, -1.0, 1.0, 1.0, -1.0]
    assert prepared["debug"]["gripper_close_threshold"] == 0.6
    assert prepared["debug"]["gripper_open_threshold"] == 0.4


def test_runner_applies_temporal_debounce_to_gripper_state_predictions():
    runner = TrajectoryChunkRunner(
        ChunkRolloutConfig(
            gripper_source="model",
            gripper_close_threshold=0.6,
            gripper_open_threshold=0.4,
            gripper_transition_debounce_steps=2,
        )
    )
    prepared = runner.prepare_chunk(
        {
            "positions": np.array(
                [[[0.00, 0.00, 0.00], [0.01, 0.00, 0.00], [0.02, 0.00, 0.00], [0.03, 0.00, 0.00], [0.04, 0.00, 0.00]]],
                dtype=np.float32,
            ),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 5], dtype=np.float32),
            "pred_gripper_state_probs": np.array(
                [[
                    [0.45, 0.55],  # below close threshold
                    [0.38, 0.62],  # close evidence 1
                    [0.37, 0.63],  # close evidence 2 -> close
                    [0.61, 0.39],  # open evidence 1
                    [0.62, 0.38],  # open evidence 2 -> reopen
                ]],
                dtype=np.float32,
            ),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=5,
        initial_gripper_cmd=-1.0,
    )

    assert prepared["gripper_cmds"].tolist() == [-1.0, -1.0, 1.0, 1.0, -1.0]
    assert prepared["debug"]["gripper_transition_debounce_steps"] == 2


def test_runner_carries_gripper_debounce_across_replanned_chunks():
    runner = TrajectoryChunkRunner(
        ChunkRolloutConfig(
            gripper_source="model",
            gripper_close_threshold=0.6,
            gripper_open_threshold=0.4,
            gripper_transition_debounce_steps=2,
        )
    )
    first = runner.prepare_chunk(
        {
            "positions": np.array(
                [[[0.00, 0.00, 0.00], [0.01, 0.00, 0.00], [0.02, 0.00, 0.00]]],
                dtype=np.float32,
            ),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 3], dtype=np.float32),
            "pred_gripper_state_probs": np.array(
                [[
                    [0.35, 0.65],  # close evidence 1
                    [0.34, 0.66],  # close evidence 2 -> close
                    [0.70, 0.30],  # open evidence 1
                ]],
                dtype=np.float32,
            ),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=3,
        initial_gripper_cmd=-1.0,
    )
    second = runner.prepare_chunk(
        {
            "positions": np.array(
                [[[0.03, 0.00, 0.00], [0.04, 0.00, 0.00], [0.05, 0.00, 0.00]]],
                dtype=np.float32,
            ),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 3], dtype=np.float32),
            "pred_gripper_state_probs": np.array(
                [[
                    [0.72, 0.28],  # open evidence 2 -> reopen across chunk boundary
                    [0.45, 0.55],
                    [0.44, 0.56],
                ]],
                dtype=np.float32,
            ),
        },
        anchor_pos=np.array([0.02, 0.0, 0.0], dtype=np.float32),
        current_start=3,
        n_use=3,
        initial_gripper_cmd=float(first["gripper_cmds"][-1]),
    )

    assert first["gripper_cmds"].tolist() == [-1.0, 1.0, 1.0]
    assert first["debug"]["gripper_debounce_pending_count"] == 1
    assert first["debug"]["gripper_debounce_pending_target"] is False
    assert second["gripper_cmds"].tolist() == [-1.0, -1.0, -1.0]
    assert second["debug"]["gripper_debounce_pending_count"] == 0


def test_phase_aware_runner_applies_phase_override_with_shared_state():
    runner = PhaseAwareChunkRunner(
        ChunkRolloutConfig(gripper_source="demo"),
        phase_configs={
            "approach": ChunkRolloutConfig(
                gripper_source="demo",
                commit_prefix_steps=2,
                blend_overlap_alpha=0.5,
            )
        },
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
        phase_name="default",
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
        phase_name="approach",
    )

    assert first["debug"]["phase_mode"] == "default"
    assert second["debug"]["phase_mode"] == "approach"
    assert second["debug"]["commit_prefix_steps"] == 1


def test_runner_adaptively_commits_overlap_when_replans_disagree():
    runner = TrajectoryChunkRunner(
        ChunkRolloutConfig(
            blend_overlap_alpha=0.6,
            adaptive_commit_threshold_m=0.05,
            adaptive_commit_max_steps=2,
            gripper_source="demo",
        )
    )
    runner.prepare_chunk(
        {
            "positions": np.array(
                [[[0.00, 0.00, 0.00], [0.02, 0.00, 0.00], [0.04, 0.00, 0.00], [0.06, 0.00, 0.00]]],
                dtype=np.float32,
            ),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 4], dtype=np.float32),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=4,
        initial_gripper_cmd=-1.0,
        demo_gripper_chunk=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
    )
    prepared = runner.prepare_chunk(
        {
            "positions": np.array(
                [[[0.18, 0.00, 0.00], [0.20, 0.00, 0.00], [0.22, 0.00, 0.00], [0.24, 0.00, 0.00]]],
                dtype=np.float32,
            ),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 4], dtype=np.float32),
        },
        anchor_pos=np.array([0.02, 0.0, 0.0], dtype=np.float32),
        current_start=2,
        n_use=4,
        initial_gripper_cmd=-1.0,
        demo_gripper_chunk=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
    )

    assert prepared["debug"]["overlap_steps"] == 2
    assert prepared["debug"]["adaptive_commit_triggered"] == 1
    assert prepared["debug"]["adaptive_commit_steps"] == 2
    assert prepared["debug"]["commit_prefix_steps"] == 2
    assert np.allclose(
        prepared["positions_full"][:2],
        np.array([[0.04, 0.0, 0.0], [0.06, 0.0, 0.0]], dtype=np.float32),
    )


def test_runner_smooths_step_prefix_against_previous_tail_velocity():
    runner = TrajectoryChunkRunner(
        ChunkRolloutConfig(
            delta_continuation_alpha=0.5,
            delta_continuation_steps=2,
            gripper_source="demo",
        )
    )
    runner.prepare_chunk(
        {
            "positions": np.array(
                [[[0.00, 0.00, 0.00], [0.02, 0.00, 0.00], [0.04, 0.00, 0.00], [0.06, 0.00, 0.00]]],
                dtype=np.float32,
            ),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 4], dtype=np.float32),
        },
        anchor_pos=np.zeros(3, dtype=np.float32),
        current_start=0,
        n_use=4,
        initial_gripper_cmd=-1.0,
        demo_gripper_chunk=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
    )
    prepared = runner.prepare_chunk(
        {
            "positions": np.array(
                [[[0.06, 0.00, 0.00], [0.10, 0.00, 0.00], [0.14, 0.00, 0.00], [0.18, 0.00, 0.00]]],
                dtype=np.float32,
            ),
            "quaternions": np.array([[[1.0, 0.0, 0.0, 0.0]] * 4], dtype=np.float32),
        },
        anchor_pos=np.array([0.06, 0.0, 0.0], dtype=np.float32),
        current_start=4,
        n_use=4,
        initial_gripper_cmd=-1.0,
        demo_gripper_chunk=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
    )

    step_norms = np.linalg.norm(np.diff(prepared["positions_full"], axis=0), axis=1)
    assert prepared["debug"]["delta_continuation_applied"] == 1
    assert prepared["debug"]["delta_continuation_steps_used"] == 2
    assert np.allclose(step_norms[:2], np.array([0.03, 0.04], dtype=np.float32), atol=1e-6)
