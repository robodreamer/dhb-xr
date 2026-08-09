"""Tests for OCP, learned, and DHB-retarget primitive adapters."""

import numpy as np
import pytest

from dhb_xr.primitives import (
    BackendPrimitiveSource,
    DHBRetargetSource,
    EndpointCorrectedReferenceSource,
    EvidenceLevel,
    PrimitiveContractError,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitiveRequest,
    RelationalAnchorGate,
    RelationalAnchorReferenceSource,
    RelationalPoseAnchor,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
EVIDENCE = PrimitiveEvidence(EvidenceLevel.PLANNED, "adapter-test")


def _request() -> PrimitiveRequest:
    return PrimitiveRequest(
        primitive_id="transport",
        start_position=np.zeros(3),
        start_quaternion_wxyz=IDENTITY,
        goal_position=np.ones(3),
        goal_quaternion_wxyz=IDENTITY,
        samples=6,
        duration_s=1.5,
        observation_epoch=9,
        metadata={"task": "sort"},
    )


def _result(offset: float = 0.0):
    t = np.linspace(0.0, 1.0, 6)
    return {
        "positions": t[:, None] * np.ones((1, 3)) + offset,
        "quaternions": np.tile(IDENTITY, (6, 1)),
        "solver": "fake",
        "loss": np.float64(0.25 + offset),
    }


def test_learned_source_preserves_multiple_candidate_ranks():
    source = BackendPrimitiveSource(
        "learned-flow:test",
        PrimitiveFamily.LEARNED_FLOW,
        EVIDENCE,
        backend=lambda request: [_result(0.0), _result(0.1)],
    )

    candidates = source.propose(_request())

    assert [candidate.primitive_id for candidate in candidates] == [
        "transport:rank-0",
        "transport:rank-1",
    ]
    assert all(candidate.observation_epoch == 9 for candidate in candidates)
    assert candidates[0].metadata["backend"]["solver"] == "fake"
    assert candidates[0].metadata["task"] == "sort"


def test_dhb_retarget_source_calls_existing_optimizer_contract():
    calls = {}

    def fake_generator(**kwargs):
        calls.update(kwargs)
        return _result()

    source = DHBRetargetSource(
        reference_positions=np.zeros((6, 3)),
        reference_quaternions_wxyz=np.tile(IDENTITY, (6, 1)),
        evidence=EVIDENCE,
        backend="fatrop",
        allow_fallback=False,
        generator=fake_generator,
    )

    (candidate,) = source.propose(_request())

    assert candidate.family == PrimitiveFamily.DHB_MEMORY
    assert candidate.metadata["candidate_source"] == "dhb-retarget:fatrop"
    assert calls["backend"] == "fatrop"
    assert calls["allow_fallback"] is False
    assert calls["traj_length"] == 6
    np.testing.assert_allclose(calls["pose_target_final"]["position"], np.ones(3))


def test_endpoint_corrected_reference_preserves_shape_and_exact_boundaries():
    phase = np.linspace(0.0, 1.0, 31)
    positions = np.column_stack(
        [phase, 0.2 * np.sin(np.pi * phase), 0.1 * np.sin(2.0 * np.pi * phase)]
    )
    angles = 0.75 * np.pi * phase
    quaternions = np.column_stack([np.cos(angles / 2.0), np.zeros((31, 2)), np.sin(angles / 2.0)])
    request = PrimitiveRequest(
        "retarget",
        np.array([-0.1, 0.05, 0.02]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.15, -0.04, 0.08]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        samples=41,
    )
    source = EndpointCorrectedReferenceSource(positions, quaternions, EVIDENCE)

    (candidate,) = source.propose(request)

    assert source.source_id == "reference:endpoint_corrected_se3"
    assert candidate.family == PrimitiveFamily.CUSTOM
    np.testing.assert_allclose(candidate.motion.positions[0], request.start_position)
    np.testing.assert_allclose(candidate.motion.positions[-1], request.goal_position)
    assert abs(
        float(
            np.dot(
                candidate.motion.quaternions_wxyz[-1],
                request.goal_quaternion_wxyz,
            )
        )
    ) == pytest.approx(1.0)
    assert float(np.ptp(candidate.motion.positions[:, 1])) > 0.1


def test_ocp_source_uses_same_candidate_contract():
    source = BackendPrimitiveSource(
        "ocp:fake",
        PrimitiveFamily.CONSTRAINED_OCP,
        EVIDENCE,
        backend=lambda request: _result(),
    )

    (candidate,) = source.propose(_request())

    assert candidate.family == PrimitiveFamily.CONSTRAINED_OCP
    assert candidate.motion.samples == 6
    assert candidate.motion.duration_s == 1.5


def test_dhb_retarget_source_rejects_malformed_reference_before_backend_call():
    with pytest.raises(PrimitiveContractError, match="reference_quaternions_wxyz"):
        DHBRetargetSource(
            reference_positions=np.zeros((6, 3)),
            reference_quaternions_wxyz=np.zeros((5, 4)),
            evidence=EVIDENCE,
        )


def test_relational_anchor_contract_and_gate_are_phase_aligned():
    anchor = RelationalPoseAnchor(
        phase=0.4,
        position=np.array([0.4, 0.1, 0.0]),
        quaternion_wxyz=IDENTITY,
        relation_id="aperture-a",
    )
    request = PrimitiveRequest(
        primitive_id="anchored",
        start_position=np.zeros(3),
        start_quaternion_wxyz=IDENTITY,
        goal_position=np.array([1.0, 0.0, 0.0]),
        goal_quaternion_wxyz=IDENTITY,
        samples=6,
        relational_anchors=(anchor,),
    )
    result = _result()
    result["positions"][2] = anchor.position
    source = BackendPrimitiveSource(
        "ocp:anchored",
        PrimitiveFamily.CONSTRAINED_OCP,
        EVIDENCE,
        backend=lambda ignored: result,
    )

    (candidate,) = source.propose(request)
    decision = RelationalAnchorGate(position_tolerance_m=1.0e-6).evaluate(candidate, request)

    assert decision.accepted
    assert decision.metrics["relation_ids"] == ["aperture-a"]


def test_relational_anchor_phases_must_be_unique_and_increasing():
    first = RelationalPoseAnchor(0.6, np.zeros(3), IDENTITY, "late")
    second = RelationalPoseAnchor(0.4, np.zeros(3), IDENTITY, "early")
    with pytest.raises(PrimitiveContractError, match="strictly increasing"):
        PrimitiveRequest(
            primitive_id="bad-anchors",
            start_position=np.zeros(3),
            start_quaternion_wxyz=IDENTITY,
            goal_position=np.ones(3),
            goal_quaternion_wxyz=IDENTITY,
            relational_anchors=(first, second),
        )


def test_raw_relational_anchor_control_hits_intermediate_pose():
    phase = np.linspace(0.0, 1.0, 11)
    reference = np.column_stack([phase, np.zeros((11, 2))])
    anchor = RelationalPoseAnchor(0.5, [0.5, 0.2, 0.0], IDENTITY, "fixture")
    request = PrimitiveRequest(
        primitive_id="raw-anchor",
        start_position=reference[0],
        start_quaternion_wxyz=IDENTITY,
        goal_position=reference[-1],
        goal_quaternion_wxyz=IDENTITY,
        samples=11,
        relational_anchors=(anchor,),
    )

    (candidate,) = RelationalAnchorReferenceSource(
        reference,
        np.tile(IDENTITY, (11, 1)),
        EVIDENCE,
    ).propose(request)

    np.testing.assert_allclose(candidate.motion.positions[5], anchor.position)
    assert candidate.metadata["candidate_source"] == "reference:relational_anchor_se3"


def test_raw_relational_anchor_control_snaps_non_grid_phases_at_short_horizon():
    phase = np.linspace(0.0, 1.0, 16)
    reference = np.column_stack([phase, np.zeros((16, 2))])
    anchors = (
        RelationalPoseAnchor(0.50, [0.50, 0.10, 0.0], IDENTITY, "early"),
        RelationalPoseAnchor(0.68, [0.68, 0.12, 0.0], IDENTITY, "middle"),
        RelationalPoseAnchor(0.86, [0.86, 0.08, 0.0], IDENTITY, "late"),
    )
    request = PrimitiveRequest(
        primitive_id="short-raw-anchor",
        start_position=reference[0],
        start_quaternion_wxyz=IDENTITY,
        goal_position=reference[-1],
        goal_quaternion_wxyz=IDENTITY,
        samples=16,
        relational_anchors=anchors,
    )

    (candidate,) = RelationalAnchorReferenceSource(
        reference,
        np.tile(IDENTITY, (16, 1)),
        EVIDENCE,
    ).propose(request)

    indices = [round(anchor.phase * 15) for anchor in anchors]
    for index, anchor in zip(indices, anchors):
        np.testing.assert_allclose(candidate.motion.positions[index], anchor.position)
