"""Tests for OCP, learned, and DHB-retarget primitive adapters."""

import numpy as np
import pytest

from dhb_xr.primitives import (
    BackendPrimitiveSource,
    DHBRetargetSource,
    EvidenceLevel,
    PrimitiveContractError,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitiveRequest,
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
