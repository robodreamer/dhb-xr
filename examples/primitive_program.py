"""Run a deterministic pick-and-place program with explicit checks and recovery.

This example exercises DHB-XR's primitive contracts and executive with synthetic
feedback. It demonstrates control flow and authority boundaries; it is not a
simulation or real-robot success result.
"""

from __future__ import annotations

import numpy as np

from dhb_xr.primitives import (
    ActionTiming,
    Comparison,
    EndEffectorCommand,
    EndEffectorCommandType,
    EvidenceLevel,
    ExecutionObservation,
    ExecutorStatus,
    PrimitiveCandidate,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitiveFeedback,
    PrimitiveProgramExecutor,
    ScheduledEndEffectorAction,
    guard_all,
    program_from_sequence,
)
from dhb_xr.primitives.analytic import minimum_jerk_path
from dhb_xr.primitives.contracts import GuardCondition

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def _guard(signal: str, comparison: Comparison = Comparison.TRUTHY, **kwargs):
    return guard_all(GuardCondition(signal, comparison, **kwargs))


def build_program():
    """Build a planned-evidence program with official gripper actions."""

    evidence = PrimitiveEvidence(
        EvidenceLevel.PLANNED,
        "deterministic-example",
        source_revision="example-v1",
    )
    approach_path = minimum_jerk_path(
        np.array([0.4, 0.0, 0.3]),
        IDENTITY,
        np.array([0.55, 0.0, 0.12]),
        IDENTITY,
        samples=25,
        duration_s=1.0,
    )
    lift_path = minimum_jerk_path(
        approach_path.positions[-1],
        IDENTITY,
        np.array([0.55, 0.0, 0.30]),
        IDENTITY,
        samples=25,
        duration_s=1.0,
    )
    retreat_path = minimum_jerk_path(
        lift_path.positions[-1],
        IDENTITY,
        np.array([0.40, -0.1, 0.35]),
        IDENTITY,
        samples=25,
        duration_s=1.0,
    )
    preshape = ScheduledEndEffectorAction(
        EndEffectorCommand(
            EndEffectorCommandType.PRESHAPE,
            target_width_m=0.04,
            speed_m_s=0.03,
        ),
        ActionTiming.BEFORE_MOTION,
        completion=_guard(
            "gripper.width_m",
            Comparison.WITHIN,
            expected=0.04,
            tolerance=0.002,
        ),
    )
    approach = PrimitiveCandidate(
        "approach",
        PrimitiveFamily.MINIMUM_JERK,
        evidence,
        motion=approach_path,
        end_effector_actions=(preshape,),
        completion_guard=_guard("tcp.reached"),
        observation_epoch=0,
    )
    grasp = PrimitiveCandidate(
        "grasp",
        PrimitiveFamily.END_EFFECTOR_ONLY,
        evidence,
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
        completion_guard=_guard("gripper.contact"),
        observation_epoch=0,
        timeout_s=1.5,
    )
    lift = PrimitiveCandidate(
        "lift",
        PrimitiveFamily.MINIMUM_JERK,
        evidence,
        motion=lift_path,
        end_effector_actions=(
            ScheduledEndEffectorAction(
                EndEffectorCommand(EndEffectorCommandType.HOLD, max_force_n=20.0),
                ActionTiming.DURING_MOTION,
            ),
        ),
        completion_guard=_guard("object.retained"),
        observation_epoch=0,
    )
    release = PrimitiveCandidate(
        "release",
        PrimitiveFamily.END_EFFECTOR_ONLY,
        evidence,
        end_effector_actions=(
            ScheduledEndEffectorAction(
                EndEffectorCommand(EndEffectorCommandType.RELEASE),
                ActionTiming.BEFORE_MOTION,
            ),
        ),
        completion_guard=guard_all(
            GuardCondition("object.supported", Comparison.TRUTHY),
            GuardCondition(
                "gripper.width_m",
                Comparison.WITHIN,
                expected=0.08,
                tolerance=0.002,
            ),
        ),
        observation_epoch=0,
    )
    retreat = PrimitiveCandidate(
        "retreat",
        PrimitiveFamily.MINIMUM_JERK,
        evidence,
        motion=retreat_path,
        completion_guard=_guard("tcp.clear"),
        observation_epoch=0,
    )
    return program_from_sequence(
        "guarded-pick-place",
        (approach, grasp, lift, release, retreat),
        metadata={"evidence_boundary": "architecture_demo_only"},
    )


def _observation(timestamp_s: float, **signals) -> ExecutionObservation:
    return ExecutionObservation(0, timestamp_s, signals)


def _show(decision) -> None:
    commands = [command.command.value for command in decision.commands]
    print(
        f"status={decision.status.value:>8} node={str(decision.current_id):>8} "
        f"commands={commands} reason={decision.reason or '-'}"
    )


def run_example():
    """Execute the program against a deterministic stream of declared observations."""

    program = build_program()
    executor = PrimitiveProgramExecutor(program)
    decisions = []

    decisions.append(executor.start(_observation(0.0, gripper={"width_m": 0.08})))
    decisions.append(
        executor.update(
            PrimitiveFeedback(_observation(0.1, gripper={"width_m": 0.04}))
        )
    )
    decisions.append(
        executor.update(
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
    )
    decisions.append(
        executor.update(
            PrimitiveFeedback(_observation(1.2, gripper={"contact": True}))
        )
    )
    decisions.append(
        executor.update(
            PrimitiveFeedback(_observation(1.3), motion_phase=0.1)
        )
    )
    decisions.append(
        executor.update(
            PrimitiveFeedback(
                _observation(2.3, object={"retained": True}),
                motion_phase=1.0,
                motion_complete=True,
            )
        )
    )
    decisions.append(
        executor.update(
            PrimitiveFeedback(
                _observation(
                    2.4,
                    object={"supported": True},
                    gripper={"width_m": 0.08},
                )
            )
        )
    )
    decisions.append(
        executor.update(
            PrimitiveFeedback(
                _observation(3.4, tcp={"clear": True}),
                motion_phase=1.0,
                motion_complete=True,
            )
        )
    )

    print("DHB-XR primitive-program trace with checks and recovery")
    print("Evidence: deterministic architecture demonstration; no simulator or robot")
    for decision in decisions:
        _show(decision)
    assert decisions[-1].status == ExecutorStatus.SUCCEEDED
    return program, tuple(decisions)


def main():
    run_example()


if __name__ == "__main__":
    main()
