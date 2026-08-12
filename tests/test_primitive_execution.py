"""Tests for guarded, event-driven primitive-program execution."""

import numpy as np
import pytest

from dhb_xr.primitives import (
    ActionTiming,
    Comparison,
    EndEffectorCommand,
    EndEffectorCommandType,
    EvidenceLevel,
    ExecutionCommandType,
    ExecutionObservation,
    ExecutorStatus,
    GuardCondition,
    GuardEvaluator,
    MotionPath,
    PrimitiveCandidate,
    PrimitiveEvidence,
    PrimitiveExecutionError,
    PrimitiveFamily,
    PrimitiveFeedback,
    PrimitiveProgram,
    PrimitiveProgramExecutor,
    PrimitiveTransition,
    ScheduledEndEffectorAction,
    TransitionEvent,
    guard_all,
)

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
EVIDENCE = PrimitiveEvidence(EvidenceLevel.PLANNED, "execution-test")


def _path() -> MotionPath:
    return MotionPath(
        np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.1]]),
        np.stack([IDENTITY, IDENTITY]),
        np.array([0.0, 1.0]),
    )


def _observation(time_s: float, epoch: int = 2, **signals) -> ExecutionObservation:
    return ExecutionObservation(epoch=epoch, timestamp_s=time_s, signals=signals)


def test_before_action_completion_gates_motion_then_sequence_advances():
    preshape = ScheduledEndEffectorAction(
        EndEffectorCommand(EndEffectorCommandType.PRESHAPE, target_width_m=0.04),
        ActionTiming.BEFORE_MOTION,
        completion=guard_all(
            GuardCondition(
                "gripper.width_m",
                Comparison.WITHIN,
                expected=0.04,
                tolerance=0.001,
            )
        ),
    )
    approach = PrimitiveCandidate(
        "approach",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        end_effector_actions=(preshape,),
        completion_guard=guard_all(GuardCondition("tcp.reached", Comparison.TRUTHY)),
        observation_epoch=2,
    )
    grasp = PrimitiveCandidate(
        "grasp",
        PrimitiveFamily.END_EFFECTOR_ONLY,
        EVIDENCE,
        end_effector_actions=(
            ScheduledEndEffectorAction(
                EndEffectorCommand(
                    EndEffectorCommandType.CLOSE_UNTIL_CONTACT,
                    max_force_n=30.0,
                ),
                ActionTiming.BEFORE_MOTION,
            ),
        ),
        completion_guard=guard_all(GuardCondition("gripper.contact", Comparison.TRUTHY)),
        observation_epoch=2,
    )
    program = PrimitiveProgram(
        "pick",
        (approach, grasp),
        (PrimitiveTransition("approach", TransitionEvent.SUCCEEDED, "grasp"),),
        "approach",
    )
    executor = PrimitiveProgramExecutor(program)

    started = executor.start(_observation(0.0, gripper={"width_m": 0.08}))
    assert [command.command for command in started.commands] == [
        ExecutionCommandType.END_EFFECTOR
    ]

    ready = executor.update(
        PrimitiveFeedback(_observation(0.1, gripper={"width_m": 0.04}))
    )
    assert [command.command for command in ready.commands] == [ExecutionCommandType.START_MOTION]

    advanced = executor.update(
        PrimitiveFeedback(
            _observation(
                1.1,
                gripper={"width_m": 0.04},
                tcp={"reached": True},
            ),
            motion_phase=1.0,
            motion_complete=True,
        )
    )
    assert advanced.current_id == "grasp"
    assert advanced.event == TransitionEvent.SUCCEEDED
    assert advanced.next_id == "grasp"
    assert [command.command for command in advanced.commands] == [
        ExecutionCommandType.END_EFFECTOR
    ]

    completed = executor.update(
        PrimitiveFeedback(_observation(1.2, gripper={"contact": True}))
    )
    assert completed.status == ExecutorStatus.SUCCEEDED


def test_new_observation_epoch_cancels_stale_motion_and_requests_replan():
    candidate = PrimitiveCandidate(
        "move",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        observation_epoch=2,
    )
    executor = PrimitiveProgramExecutor(PrimitiveProgram("move", (candidate,), (), "move"))
    executor.start(_observation(0.0))

    decision = executor.update(PrimitiveFeedback(_observation(0.2, epoch=3)))

    assert decision.status == ExecutorStatus.REPLAN_REQUIRED
    assert [command.command for command in decision.commands] == [
        ExecutionCommandType.STOP_MOTION,
        ExecutionCommandType.REQUEST_REOBSERVATION,
    ]
    assert "stale observation epoch" in decision.reason


def test_failure_retry_is_bounded_by_transition_traversal_limit():
    candidate = PrimitiveCandidate(
        "grasp",
        PrimitiveFamily.END_EFFECTOR_ONLY,
        EVIDENCE,
        end_effector_actions=(
            ScheduledEndEffectorAction(
                EndEffectorCommand(EndEffectorCommandType.OPEN),
                ActionTiming.BEFORE_MOTION,
            ),
        ),
        observation_epoch=2,
    )
    program = PrimitiveProgram(
        "retry",
        (candidate,),
        (
            PrimitiveTransition(
                "grasp",
                TransitionEvent.FAILED,
                "grasp",
                max_traversals=1,
                label="retry-once",
            ),
        ),
        "grasp",
    )
    executor = PrimitiveProgramExecutor(program)
    executor.start(_observation(0.0))

    retried = executor.update(
        PrimitiveFeedback(_observation(0.1), motion_failed=True, reason="no contact")
    )
    assert retried.status == ExecutorStatus.RUNNING
    assert retried.next_id == "grasp"

    failed = executor.update(
        PrimitiveFeedback(_observation(0.2), motion_failed=True, reason="no contact")
    )
    assert failed.status == ExecutorStatus.FAILED
    assert failed.reason == "no contact"


def test_transition_can_require_reobservation_before_downstream_activation():
    first = PrimitiveCandidate(
        "first",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        observation_epoch=2,
    )
    second = PrimitiveCandidate(
        "second",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        observation_epoch=2,
    )
    program = PrimitiveProgram(
        "observe",
        (first, second),
        (
            PrimitiveTransition(
                "first",
                TransitionEvent.SUCCEEDED,
                "second",
                requires_reobservation=True,
                label="object moved after grasp",
            ),
        ),
        "first",
    )
    executor = PrimitiveProgramExecutor(program)
    executor.start(_observation(0.0))

    decision = executor.update(
        PrimitiveFeedback(_observation(1.0), motion_phase=1.0, motion_complete=True)
    )

    assert decision.status == ExecutorStatus.REPLAN_REQUIRED
    assert decision.next_id == "second"
    assert decision.commands[-1].command == ExecutionCommandType.REQUEST_REOBSERVATION


def test_guard_dwell_requires_continuous_true_interval():
    expression = guard_all(
        GuardCondition("contact", Comparison.TRUTHY, dwell_s=0.2)
    )
    evaluator = GuardEvaluator()

    assert not evaluator.evaluate(expression, _observation(0.0, contact=True), scope="contact")
    assert not evaluator.evaluate(expression, _observation(0.1, contact=False), scope="contact")
    assert not evaluator.evaluate(expression, _observation(0.2, contact=True), scope="contact")
    assert evaluator.evaluate(expression, _observation(0.4, contact=True), scope="contact")


def test_new_executor_can_resume_regrounded_program_at_downstream_node():
    first = PrimitiveCandidate(
        "first",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        observation_epoch=2,
    )
    second = PrimitiveCandidate(
        "second",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        observation_epoch=3,
    )
    program = PrimitiveProgram(
        "regrounded",
        (first, second),
        (PrimitiveTransition("first", TransitionEvent.SUCCEEDED, "second"),),
        "first",
    )

    resumed = PrimitiveProgramExecutor(program).start_at(
        "second", _observation(0.0, epoch=3)
    )

    assert resumed.status == ExecutorStatus.RUNNING
    assert resumed.current_id == "second"
    assert resumed.commands[0].command == ExecutionCommandType.START_MOTION


def test_truthy_guard_handles_vector_signals_without_ambiguous_bool_error():
    evaluator = GuardEvaluator()
    expression = guard_all(GuardCondition("contacts", Comparison.TRUTHY))

    assert evaluator.evaluate(
        expression,
        _observation(0.0, contacts=np.array([True, True])),
        scope="contacts",
    )
    assert not evaluator.evaluate(
        expression,
        _observation(0.1, contacts=np.array([True, False])),
        scope="contacts",
    )


def test_during_motion_action_is_issued_only_inside_inclusive_phase_window():
    action = ScheduledEndEffectorAction(
        EndEffectorCommand(EndEffectorCommandType.OPEN),
        ActionTiming.DURING_MOTION,
        phase_start=0.2,
        phase_end=0.4,
    )
    candidate = PrimitiveCandidate(
        "move",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        end_effector_actions=(action,),
        observation_epoch=2,
    )
    executor = PrimitiveProgramExecutor(PrimitiveProgram("move", (candidate,), (), "move"))
    executor.start(_observation(0.0))

    before_window = executor.update(
        PrimitiveFeedback(_observation(0.1), motion_phase=0.1)
    )
    assert before_window.commands == ()

    at_window_end = executor.update(
        PrimitiveFeedback(_observation(0.2), motion_phase=0.4)
    )
    assert [command.command for command in at_window_end.commands] == [
        ExecutionCommandType.END_EFFECTOR
    ]


def test_missed_during_motion_action_window_fails_closed_and_stops_motion():
    action = ScheduledEndEffectorAction(
        EndEffectorCommand(EndEffectorCommandType.OPEN),
        ActionTiming.DURING_MOTION,
        phase_start=0.1,
        phase_end=0.2,
    )
    candidate = PrimitiveCandidate(
        "move",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        end_effector_actions=(action,),
        observation_epoch=2,
    )
    executor = PrimitiveProgramExecutor(PrimitiveProgram("move", (candidate,), (), "move"))
    executor.start(_observation(0.0))

    missed = executor.update(
        PrimitiveFeedback(_observation(0.9), motion_phase=0.9)
    )

    assert missed.status == ExecutorStatus.FAILED
    assert missed.event == TransitionEvent.FAILED
    assert "action window elapsed" in missed.reason
    assert [command.command for command in missed.commands] == [
        ExecutionCommandType.STOP_MOTION
    ]


def test_executor_rejects_ambiguous_enabled_transitions():
    start = PrimitiveCandidate(
        "start",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        observation_epoch=2,
    )
    left = PrimitiveCandidate(
        "left",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        observation_epoch=2,
    )
    right = PrimitiveCandidate(
        "right",
        PrimitiveFamily.MINIMUM_JERK,
        EVIDENCE,
        motion=_path(),
        observation_epoch=2,
    )
    program = PrimitiveProgram(
        "ambiguous",
        (start, left, right),
        (
            PrimitiveTransition("start", TransitionEvent.SUCCEEDED, "left", label="left"),
            PrimitiveTransition("start", TransitionEvent.SUCCEEDED, "right", label="right"),
        ),
        "start",
    )
    executor = PrimitiveProgramExecutor(program)
    executor.start(_observation(0.0))

    with pytest.raises(PrimitiveExecutionError, match="multiple transitions enabled"):
        executor.update(
            PrimitiveFeedback(_observation(1.0), motion_phase=1.0, motion_complete=True)
        )
