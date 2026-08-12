"""Tests for evidence-preserving primitive memory and demonstration boundaries."""

import numpy as np
import pytest

from dhb_xr.primitives import (
    ActionTiming,
    ContactMode,
    EndEffectorCommand,
    EndEffectorCommandType,
    EvidenceLevel,
    MotionPath,
    PrimitiveCandidate,
    PrimitiveContractError,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitiveMemory,
    ScheduledEndEffectorAction,
    VerificationOutcome,
    VerificationRecord,
    exact_gripper_run_control_indices,
    hysteretic_gripper_control_indices,
    propose_primitive_boundaries,
    split_motion_at_indices,
    stitch_motion_paths,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _candidate(command: EndEffectorCommandType, offset: float = 0.0) -> PrimitiveCandidate:
    t = np.linspace(0.0, 1.0, 20)
    positions = np.column_stack(
        [t + offset, 0.05 * np.sin(np.pi * t), 0.1 * t * (1.0 - t)]
    )
    motion = MotionPath(
        positions,
        np.tile(IDENTITY, (len(t), 1)),
        t,
    )
    return PrimitiveCandidate(
        primitive_id=f"candidate-{command.value}-{offset}",
        family=PrimitiveFamily.DHB_MEMORY,
        evidence=PrimitiveEvidence(
            EvidenceLevel.PLANNED,
            "synthetic-test",
            source_revision="abc123",
            license_id="MIT",
        ),
        motion=motion,
        end_effector_actions=(
            ScheduledEndEffectorAction(
                EndEffectorCommand(command),
                ActionTiming.AFTER_MOTION,
            ),
        ),
        contact_mode=ContactMode.RELEASE,
        observation_epoch=1,
    )


def test_memory_filters_interaction_semantics_before_geometry(tmp_path):
    memory = PrimitiveMemory()
    release = memory.add(_candidate(EndEffectorCommandType.RELEASE, 0.0), entry_id="release")
    memory.add(_candidate(EndEffectorCommandType.OPEN, 0.0), entry_id="open")

    matches = memory.retrieve(_candidate(EndEffectorCommandType.RELEASE, 0.01), k=5)

    assert [match.record.entry_id for match in matches] == [release.entry_id]
    assert matches[0].geometric_distance >= 0.0

    path = tmp_path / "primitive-memory.json"
    memory.save(path)
    restored = PrimitiveMemory.load(path)
    assert len(restored) == 2
    assert restored.get("release").candidate.evidence.license_id == "MIT"
    np.testing.assert_allclose(
        restored.get("release").invariants,
        memory.get("release").invariants,
    )


def test_only_passing_verification_promotes_evidence():
    memory = PrimitiveMemory()
    memory.add(_candidate(EndEffectorCommandType.RELEASE), entry_id="release")
    failed = memory.record_verification(
        "release",
        VerificationRecord(
            VerificationOutcome.FAILED,
            EvidenceLevel.SIM_VERIFIED,
            "mujoco-pick-place-v1",
            metadata={"reason": "object_slip"},
        ),
    )
    assert failed.candidate.evidence.level == EvidenceLevel.PLANNED
    passed = memory.record_verification(
        "release",
        VerificationRecord(
            VerificationOutcome.PASSED,
            EvidenceLevel.SIM_VERIFIED,
            "mujoco-pick-place-v1",
            artifact_hash="sha256:result",
            metadata={"episodes": 20, "successes": 20},
        ),
    )
    assert passed.candidate.evidence.level == EvidenceLevel.SIM_VERIFIED
    assert passed.candidate.evidence.artifact_hash == "sha256:result"
    assert [record.outcome for record in passed.verification_history] == [
        VerificationOutcome.FAILED,
        VerificationOutcome.PASSED,
    ]


def test_hysteresis_avoids_gripper_ramp_segment_explosion():
    commands = np.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1])

    exact = exact_gripper_run_control_indices(commands)
    hysteretic = hysteretic_gripper_control_indices(commands, min_dwell_samples=2)

    assert len(exact) > len(hysteretic)
    assert hysteretic == (0, 5, 7)


def test_boundary_proposals_fuse_gripper_contact_and_object_motion():
    gripper = np.array([1.0, 1.0, 0.2, 0.2, 0.2, 0.2])
    contact = np.array([False, False, False, True, True, True])
    object_positions = np.zeros((6, 3))
    object_positions[4:, 0] = [0.003, 0.006]

    proposals = propose_primitive_boundaries(
        gripper,
        contacts=contact,
        object_positions=object_positions,
        min_dwell_samples=1,
    )
    reasons = {proposal.index: set(proposal.reasons) for proposal in proposals}

    assert "gripper_state_transition" in reasons[2]
    assert "contact_transition" in reasons[3]
    assert "object_motion_onset" in reasons[4]


def test_shared_waypoint_split_and_stitch_round_trip():
    motion = _candidate(EndEffectorCommandType.RELEASE).motion
    assert motion is not None
    pieces = split_motion_at_indices(motion, (0, 7, 12, motion.samples - 1))

    np.testing.assert_allclose(pieces[0].positions[-1], pieces[1].positions[0])
    stitched = stitch_motion_paths(pieces)
    np.testing.assert_allclose(stitched.positions, motion.positions)
    np.testing.assert_allclose(stitched.quaternions_wxyz, motion.quaternions_wxyz)
    np.testing.assert_allclose(stitched.timestamps_s, motion.timestamps_s)


def test_memory_rejects_duplicate_entry_ids():
    memory = PrimitiveMemory()
    memory.add(_candidate(EndEffectorCommandType.OPEN), entry_id="duplicate")
    with pytest.raises(PrimitiveContractError, match="already exists"):
        memory.add(_candidate(EndEffectorCommandType.OPEN), entry_id="duplicate")
