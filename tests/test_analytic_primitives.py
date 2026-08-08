"""Tests for analytic primitive geometry and the common source interface."""

import numpy as np
import pytest

from dhb_xr.primitives import (
    AnalyticPrimitiveSource,
    EvidenceLevel,
    PrimitiveContractError,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitiveRequest,
    arc_path,
    minimum_jerk_blend,
    minimum_jerk_path,
    screw_path,
    via_point_path,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def test_minimum_jerk_path_has_exact_pose_endpoints_and_slow_boundary_steps():
    path = minimum_jerk_path(
        np.zeros(3),
        IDENTITY,
        np.array([1.0, -0.5, 0.2]),
        np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        samples=101,
        duration_s=2.0,
    )

    np.testing.assert_allclose(path.positions[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(path.positions[-1], [1.0, -0.5, 0.2], atol=1e-14)
    goal_alignment = abs(
        float(
            np.dot(
                path.quaternions_wxyz[-1],
                [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
            )
        )
    )
    assert goal_alignment == pytest.approx(1.0)
    first_step = np.linalg.norm(path.positions[1] - path.positions[0])
    middle_step = np.linalg.norm(path.positions[51] - path.positions[50])
    assert first_step < 0.01 * middle_step
    np.testing.assert_allclose(minimum_jerk_blend(np.array([0.0, 0.5, 1.0])), [0.0, 0.5, 1.0])


def test_arc_path_rotates_position_and_tool_about_axis():
    path = arc_path(
        np.array([1.0, 0.0, 0.0]),
        IDENTITY,
        center=np.zeros(3),
        axis=np.array([0.0, 0.0, 1.0]),
        angle_rad=np.pi / 2,
        samples=21,
        duration_s=1.0,
    )

    np.testing.assert_allclose(np.linalg.norm(path.positions[:, :2], axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(path.positions[-1], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        np.abs(path.quaternions_wxyz[-1]),
        [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)],
        atol=1e-12,
    )


def test_screw_path_couples_rotation_and_translation():
    path = screw_path(
        np.array([1.0, 0.0, 0.0]),
        IDENTITY,
        axis_point=np.zeros(3),
        axis=np.array([0.0, 0.0, 1.0]),
        angle_rad=np.pi,
        translation_m=0.25,
        samples=31,
        duration_s=3.0,
    )

    np.testing.assert_allclose(path.positions[-1], [-1.0, 0.0, 0.25], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(path.positions[:, :2], axis=1), 1.0, atol=1e-12)


def test_via_point_path_preserves_shared_waypoint_once():
    waypoint = np.array([0.2, 0.3, 0.4])
    path = via_point_path(
        [np.zeros(3), waypoint, np.array([0.4, 0.0, 0.2])],
        [IDENTITY, IDENTITY, IDENTITY],
        samples_per_segment=5,
        duration_s=2.0,
    )

    assert path.samples == 9
    matches = np.where(np.all(np.isclose(path.positions, waypoint), axis=1))[0]
    np.testing.assert_array_equal(matches, [4])
    assert np.all(np.diff(path.timestamps_s) > 0.0)


def test_analytic_source_preserves_request_context_and_provenance():
    source = AnalyticPrimitiveSource(
        PrimitiveFamily.MINIMUM_JERK,
        PrimitiveEvidence(EvidenceLevel.PLANNED, "analytic-generator-v1"),
    )
    request = PrimitiveRequest(
        primitive_id="approach",
        start_position=np.zeros(3),
        start_quaternion_wxyz=IDENTITY,
        goal_position=np.ones(3),
        goal_quaternion_wxyz=IDENTITY,
        samples=11,
        duration_s=1.0,
        observation_epoch=7,
        reference_frames={"goal": "object/grasp"},
        metadata={"task": "pick"},
    )

    (candidate,) = source.propose(request)

    assert source.source_id == "analytic:minimum_jerk"
    assert candidate.observation_epoch == 7
    assert candidate.reference_frames == {"goal": "object/grasp"}
    assert candidate.metadata["candidate_source"] == source.source_id
    assert candidate.metadata["task"] == "pick"


def test_primitive_request_detaches_and_normalizes_boundary_arrays():
    start = np.zeros(3)
    goal_quaternion = np.array([2.0, 0.0, 0.0, 0.0])
    request = PrimitiveRequest(
        primitive_id="move",
        start_position=start,
        start_quaternion_wxyz=IDENTITY,
        goal_position=np.ones(3),
        goal_quaternion_wxyz=goal_quaternion,
    )

    start[0] = 99.0
    goal_quaternion[0] = 10.0
    np.testing.assert_allclose(request.start_position, np.zeros(3))
    np.testing.assert_allclose(request.goal_quaternion_wxyz, IDENTITY)
    assert request.start_position.flags.writeable is False


def test_analytic_paths_reject_degenerate_axes():
    with pytest.raises(PrimitiveContractError, match="axis must be non-zero"):
        arc_path(
            np.ones(3),
            IDENTITY,
            center=np.zeros(3),
            axis=np.zeros(3),
            angle_rad=1.0,
        )
