"""Tests for canonical, provenance-preserving demonstration ingestion."""

import numpy as np
import pytest

from dhb_xr.primitives import (
    DemonstrationTrace,
    EvidenceLevel,
    MotionPath,
    PrimitiveContractError,
    PrimitiveEvidence,
    mine_demonstration_segments,
    segment_to_candidate,
)


def _trace() -> DemonstrationTrace:
    samples = 8
    positions = np.column_stack(
        [np.linspace(0.0, 0.2, samples), np.zeros(samples), np.zeros(samples)]
    )
    object_positions = np.zeros((samples, 3))
    object_positions[5:, 0] = [0.003, 0.006, 0.009]
    return DemonstrationTrace(
        motion=MotionPath(
            positions,
            np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (samples, 1)),
            np.linspace(0.0, 0.7, samples),
        ),
        gripper_commands=np.array([1.0, 1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        contacts=np.array([False, False, False, True, True, True, True, True]),
        object_positions=object_positions,
        source_uri="dataset://episode/42",
        dataset="synthetic-ingestion-test",
        action_convention="normalized_aperture_open_is_one",
        license_id="MIT",
        metadata={"calibration": "test-v1"},
    )


def test_mining_preserves_shared_endpoints_and_provenance():
    trace = _trace()
    segments = mine_demonstration_segments(trace, min_dwell_samples=1)

    assert len(segments) >= 3
    for previous, following in zip(segments[:-1], segments[1:]):
        assert previous.end_index == following.start_index
        np.testing.assert_allclose(previous.motion.positions[-1], following.motion.positions[0])
    assert all(segment.source_uri == "dataset://episode/42" for segment in segments)
    assert any("contact_transition" in segment.end_reasons for segment in segments)
    assert any("object_motion_onset" in segment.end_reasons for segment in segments)


def test_segment_requires_explicit_semantics_before_candidate_promotion():
    segment = mine_demonstration_segments(_trace(), min_dwell_samples=1)[0]
    evidence = PrimitiveEvidence(
        EvidenceLevel.PLANNED,
        "mined-boundary-proposal",
        dataset=segment.dataset,
        license_id=segment.license_id,
    )
    with pytest.raises(PrimitiveContractError, match="semantic_label is required"):
        segment_to_candidate(
            segment,
            primitive_id="segment-0",
            semantic_label="",
            evidence=evidence,
        )

    candidate = segment_to_candidate(
        segment,
        primitive_id="segment-0",
        semantic_label="approach",
        evidence=evidence,
    )
    assert candidate.metadata["semantic_label"] == "approach"
    assert candidate.metadata["source_uri"] == "dataset://episode/42"
    assert candidate.metadata["ingestion_status"].startswith("boundary_proposal")
