"""Contract tests for typed DHB-XR manipulation primitives."""

import json

import numpy as np
import pytest

from dhb_xr.primitives import (
    ActionTiming,
    Comparison,
    ContactMode,
    EndEffectorCommand,
    EndEffectorCommandType,
    EvidenceLevel,
    GuardExpression,
    MotionPath,
    PrimitiveCandidate,
    PrimitiveContractError,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitiveProgram,
    PrimitiveTransition,
    ScheduledEndEffectorAction,
    TransitionEvent,
)
from dhb_xr.primitives.contracts import GuardCondition, program_from_sequence


def _path() -> MotionPath:
    positions = np.array([[0.0, 0.0, 0.1], [0.1, 0.0, 0.2], [0.2, 0.1, 0.2]])
    quaternions = np.array(
        [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    return MotionPath(positions, quaternions, np.array([0.0, 0.5, 1.0]))


def _evidence() -> PrimitiveEvidence:
    return PrimitiveEvidence(
        level=EvidenceLevel.SIM_VERIFIED,
        provenance="unit-test-fixture",
        source_revision="0123456789abcdef",
        compatible_robot_controller_revisions=("panda-test-v1",),
        metadata={"episodes": np.int64(20)},
    )


def _candidate(primitive_id: str = "approach") -> PrimitiveCandidate:
    completion = GuardExpression(
        (
            GuardCondition(
                signal="tcp.position_error_m",
                comparison=Comparison.WITHIN,
                expected=0.0,
                tolerance=0.008,
                dwell_s=0.1,
            ),
        )
    )
    action = ScheduledEndEffectorAction(
        command=EndEffectorCommand(
            EndEffectorCommandType.PRESHAPE,
            target_width_m=0.04,
            speed_m_s=0.03,
        ),
        timing=ActionTiming.DURING_MOTION,
        phase_start=0.0,
        phase_end=0.6,
        completion=GuardExpression(
            (
                GuardCondition(
                    signal="gripper.width_m",
                    comparison=Comparison.WITHIN,
                    expected=0.04,
                    tolerance=0.002,
                ),
            )
        ),
    )
    return PrimitiveCandidate(
        primitive_id=primitive_id,
        family=PrimitiveFamily.MINIMUM_JERK,
        motion=_path(),
        end_effector_actions=(action,),
        contact_mode=ContactMode.FREE_SPACE,
        reference_frames={"goal": "bottle/grasp"},
        completion_guard=completion,
        observation_epoch=4,
        evidence=_evidence(),
        metadata={"generator": "minimum_jerk"},
    )


def test_motion_path_normalizes_quaternions_and_detaches_input():
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    quaternions = np.array([[2.0, 0.0, 0.0, 0.0], [-3.0, 0.0, 0.0, 0.0]])
    path = MotionPath(positions, quaternions, np.array([0.0, 1.0]))

    positions[1, 0] = 99.0
    np.testing.assert_allclose(path.positions[:, 0], [0.0, 1.0])
    np.testing.assert_allclose(path.quaternions_wxyz[:, 0], [1.0, 1.0])
    assert path.positions.flags.writeable is False
    assert path.quaternions_wxyz.flags.writeable is False


@pytest.mark.parametrize(
    ("timestamps", "quaternions", "message"),
    [
        ([0.0, 0.0], [[1.0, 0.0, 0.0, 0.0]] * 2, "strictly increasing"),
        ([0.0, 1.0], [[0.0, 0.0, 0.0, 0.0]] * 2, "non-zero"),
    ],
)
def test_motion_path_rejects_invalid_time_or_quaternion(timestamps, quaternions, message):
    with pytest.raises(PrimitiveContractError, match=message):
        MotionPath(np.zeros((2, 3)), np.asarray(quaternions), np.asarray(timestamps))


def test_end_effector_commands_enforce_semantic_parameters():
    with pytest.raises(PrimitiveContractError, match="requires target_width_m"):
        EndEffectorCommand(EndEffectorCommandType.PRESHAPE)
    with pytest.raises(PrimitiveContractError, match="positive max_force_n"):
        EndEffectorCommand(EndEffectorCommandType.CLOSE_UNTIL_CONTACT)
    with pytest.raises(PrimitiveContractError, match="cannot specify finger width"):
        EndEffectorCommand(EndEffectorCommandType.VACUUM_ON, target_width_m=0.01)


def test_on_event_action_requires_a_trigger():
    with pytest.raises(PrimitiveContractError, match="requires a trigger"):
        ScheduledEndEffectorAction(
            EndEffectorCommand(EndEffectorCommandType.OPEN),
            ActionTiming.ON_EVENT,
        )


def test_candidate_and_program_round_trip_through_json():
    approach = _candidate("approach")
    grasp = PrimitiveCandidate(
        primitive_id="grasp",
        family=PrimitiveFamily.END_EFFECTOR_ONLY,
        evidence=_evidence(),
        end_effector_actions=(
            ScheduledEndEffectorAction(
                EndEffectorCommand(
                    EndEffectorCommandType.CLOSE_UNTIL_CONTACT,
                    max_force_n=30.0,
                    timeout_s=1.0,
                ),
                ActionTiming.BEFORE_MOTION,
            ),
        ),
        contact_mode=ContactMode.GUARDED,
        observation_epoch=4,
    )
    program = program_from_sequence("pick", (approach, grasp), metadata={"task": "pick"})

    payload = json.loads(json.dumps(program.to_dict()))
    restored = PrimitiveProgram.from_dict(payload)

    assert restored.program_id == "pick"
    assert restored.start_id == "approach"
    assert restored.primitive("grasp").family == PrimitiveFamily.END_EFFECTOR_ONLY
    assert restored.primitive("approach").observation_epoch == 4
    assert restored.primitive("approach").evidence.level == EvidenceLevel.SIM_VERIFIED
    np.testing.assert_allclose(
        restored.primitive("approach").motion.positions,
        approach.motion.positions,
    )


def test_program_rejects_dangling_and_unreachable_nodes():
    approach = _candidate("approach")
    grasp = _candidate("grasp")
    with pytest.raises(PrimitiveContractError, match="transition target"):
        PrimitiveProgram(
            "bad-target",
            (approach,),
            (
                PrimitiveTransition(
                    "approach", TransitionEvent.SUCCEEDED, target_id="missing"
                ),
            ),
            "approach",
        )
    with pytest.raises(PrimitiveContractError, match="unreachable"):
        PrimitiveProgram("unreachable", (approach, grasp), (), "approach")


def test_motionless_non_end_effector_primitive_fails_closed():
    with pytest.raises(PrimitiveContractError, match="requires a motion path"):
        PrimitiveCandidate(
            primitive_id="invalid",
            family=PrimitiveFamily.DHB_MEMORY,
            evidence=_evidence(),
        )


def test_metadata_must_remain_json_serializable():
    with pytest.raises(PrimitiveContractError, match="unsupported type"):
        PrimitiveEvidence(
            EvidenceLevel.PLANNED,
            "bad-metadata",
            metadata={"callback": lambda: None},
        )
