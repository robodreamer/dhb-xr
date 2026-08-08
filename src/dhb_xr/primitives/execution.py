"""Deterministic, guarded execution state machine for primitive programs.

This module coordinates declared motion and end-effector actions.  It does not
perform inverse kinematics, approve collision safety, infer task meaning, or
claim physical success beyond the observation predicates supplied to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np

from dhb_xr.primitives.contracts import (
    ActionTiming,
    Comparison,
    GuardCondition,
    GuardExpression,
    GuardMode,
    MotionPath,
    PrimitiveCandidate,
    PrimitiveContractError,
    PrimitiveProgram,
    ScheduledEndEffectorAction,
    TransitionEvent,
)


class PrimitiveExecutionError(RuntimeError):
    """Raised when runtime inputs or the transition graph are ambiguous."""


class ExecutorStatus(str, Enum):
    """Current lifecycle state of a primitive program executor."""

    IDLE = "idle"
    RUNNING = "running"
    REPLAN_REQUIRED = "replan_required"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class ExecutionCommandType(str, Enum):
    """Commands emitted to lower-level motion and end-effector adapters."""

    START_MOTION = "start_motion"
    STOP_MOTION = "stop_motion"
    END_EFFECTOR = "end_effector"
    REQUEST_REOBSERVATION = "request_reobservation"


@dataclass(frozen=True)
class ExecutionObservation:
    """Timestamped observation with an explicit scene-belief epoch."""

    epoch: int
    timestamp_s: float
    signals: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        epoch = int(self.epoch)
        timestamp = float(self.timestamp_s)
        if epoch < 0:
            raise PrimitiveExecutionError("observation epoch must be non-negative")
        if not np.isfinite(timestamp) or timestamp < 0.0:
            raise PrimitiveExecutionError("observation timestamp must be finite and non-negative")
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "signals", dict(self.signals))

    def resolve(self, signal: str) -> Any:
        """Resolve an exact or dotted signal from the observation mapping."""

        if signal in self.signals:
            return self.signals[signal]
        current: Any = self.signals
        for component in signal.split("."):
            if not isinstance(current, Mapping) or component not in current:
                raise KeyError(signal)
            current = current[component]
        return current


@dataclass(frozen=True)
class PrimitiveFeedback:
    """Feedback from the motion/controller layer for one executor update."""

    observation: ExecutionObservation
    motion_phase: float = 0.0
    motion_complete: bool = False
    motion_failed: bool = False
    cancelled: bool = False
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        phase = float(self.motion_phase)
        if not np.isfinite(phase) or not 0.0 <= phase <= 1.0:
            raise PrimitiveExecutionError("motion_phase must be finite and lie in [0, 1]")
        terminal_flags = sum((self.motion_complete, self.motion_failed, self.cancelled))
        if terminal_flags > 1:
            raise PrimitiveExecutionError("feedback terminal flags are mutually exclusive")
        object.__setattr__(self, "motion_phase", phase)


@dataclass(frozen=True)
class ExecutionCommand:
    """One typed instruction for a lower-level adapter."""

    command: ExecutionCommandType
    primitive_id: str
    motion: Optional[MotionPath] = None
    end_effector_action: Optional[ScheduledEndEffectorAction] = None
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        command = (
            self.command
            if isinstance(self.command, ExecutionCommandType)
            else ExecutionCommandType(self.command)
        )
        if command == ExecutionCommandType.START_MOTION and self.motion is None:
            raise PrimitiveExecutionError("start_motion command requires a motion path")
        if command == ExecutionCommandType.END_EFFECTOR and self.end_effector_action is None:
            raise PrimitiveExecutionError("end_effector command requires an action")
        object.__setattr__(self, "command", command)


@dataclass(frozen=True)
class ExecutionDecision:
    """Executor status and commands produced by one state transition."""

    status: ExecutorStatus
    current_id: Optional[str]
    commands: tuple[ExecutionCommand, ...] = ()
    event: Optional[TransitionEvent] = None
    next_id: Optional[str] = None
    reason: Optional[str] = None


class GuardEvaluator:
    """Evaluate serializable guards while tracking per-condition dwell time."""

    def __init__(self) -> None:
        self._true_since: dict[tuple[str, int], float] = {}

    def reset_scope(self, scope: str) -> None:
        for key in tuple(self._true_since):
            if key[0] == scope:
                del self._true_since[key]

    def evaluate(
        self,
        expression: Optional[GuardExpression],
        observation: ExecutionObservation,
        *,
        scope: str,
    ) -> bool:
        if expression is None:
            return True
        values = [
            self._evaluate_condition(condition, observation, scope, index)
            for index, condition in enumerate(expression.conditions)
        ]
        return all(values) if expression.mode == GuardMode.ALL else any(values)

    def _evaluate_condition(
        self,
        condition: GuardCondition,
        observation: ExecutionObservation,
        scope: str,
        index: int,
    ) -> bool:
        key = (scope, index)
        try:
            observed = observation.resolve(condition.signal)
        except KeyError:
            self._true_since.pop(key, None)
            return False
        raw_result = self._compare(condition, observed)
        if not raw_result:
            self._true_since.pop(key, None)
            return False
        if condition.dwell_s <= 0.0:
            return True
        true_since = self._true_since.setdefault(key, observation.timestamp_s)
        return observation.timestamp_s - true_since >= condition.dwell_s

    @staticmethod
    def _compare(condition: GuardCondition, observed: Any) -> bool:
        comparison = condition.comparison
        expected = condition.expected
        if comparison == Comparison.TRUTHY:
            return bool(np.all(np.asarray(observed)))
        if comparison == Comparison.FALSY:
            return not bool(np.all(np.asarray(observed)))
        if comparison == Comparison.EQ:
            return bool(np.array_equal(np.asarray(observed), np.asarray(expected)))
        if comparison == Comparison.NE:
            return not bool(np.array_equal(np.asarray(observed), np.asarray(expected)))
        if comparison == Comparison.WITHIN:
            try:
                error = np.asarray(observed, dtype=np.float64) - np.asarray(
                    expected, dtype=np.float64
                )
            except (TypeError, ValueError):
                return False
            if not np.all(np.isfinite(error)):
                return False
            assert condition.tolerance is not None
            return float(np.linalg.norm(error)) <= condition.tolerance
        try:
            observed_value = float(observed)
            expected_value = float(expected)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(observed_value) or not np.isfinite(expected_value):
            return False
        if comparison == Comparison.GT:
            return observed_value > expected_value
        if comparison == Comparison.GE:
            return observed_value >= expected_value
        if comparison == Comparison.LT:
            return observed_value < expected_value
        if comparison == Comparison.LE:
            return observed_value <= expected_value
        raise PrimitiveExecutionError(f"unsupported comparison {comparison.value}")


class PrimitiveProgramExecutor:
    """Execute one immutable primitive program through deterministic feedback steps."""

    def __init__(self, program: PrimitiveProgram, *, default_timeout_s: float = 10.0) -> None:
        if not isinstance(program, PrimitiveProgram):
            raise PrimitiveContractError("program must be PrimitiveProgram")
        timeout = float(default_timeout_s)
        if not np.isfinite(timeout) or timeout <= 0.0:
            raise PrimitiveExecutionError("default_timeout_s must be finite and positive")
        self.program = program
        self.default_timeout_s = timeout
        self.status = ExecutorStatus.IDLE
        self.current_id: Optional[str] = None
        self._active_started_s = 0.0
        self._last_timestamp_s = -np.inf
        self._motion_started = False
        self._motion_complete = False
        self._issued_actions: set[int] = set()
        self._traversals: dict[int, int] = {}
        self._guards = GuardEvaluator()

    @property
    def current(self) -> Optional[PrimitiveCandidate]:
        return None if self.current_id is None else self.program.primitive(self.current_id)

    def start(self, observation: ExecutionObservation) -> ExecutionDecision:
        if self.status != ExecutorStatus.IDLE:
            raise PrimitiveExecutionError("executor can only be started from idle")
        return self._activate(self.program.start_id, observation)

    def start_at(
        self, primitive_id: str, observation: ExecutionObservation
    ) -> ExecutionDecision:
        """Start a newly grounded program at a declared downstream node.

        Use this on a new executor after a prior executor returned
        ``REPLAN_REQUIRED``. It does not mutate stale candidates in place.
        """

        if self.status != ExecutorStatus.IDLE:
            raise PrimitiveExecutionError("executor can only be started from idle")
        return self._activate(primitive_id, observation)

    def update(self, feedback: PrimitiveFeedback) -> ExecutionDecision:
        if self.status != ExecutorStatus.RUNNING or self.current is None:
            raise PrimitiveExecutionError("executor update requires a running primitive")
        observation = feedback.observation
        self._check_monotonic_time(observation)
        candidate = self.current
        if observation.epoch != candidate.observation_epoch:
            return self._require_replan(
                observation,
                reason=(
                    f"stale observation epoch: candidate={candidate.observation_epoch}, "
                    f"observed={observation.epoch}"
                ),
            )
        if feedback.cancelled:
            return self._finish_primitive(
                TransitionEvent.CANCELLED,
                observation,
                reason=feedback.reason or "controller cancelled primitive",
            )
        if feedback.motion_failed:
            return self._finish_primitive(
                TransitionEvent.FAILED,
                observation,
                reason=feedback.reason or "motion controller reported failure",
            )
        if observation.timestamp_s - self._active_started_s > self._timeout(candidate):
            return self._finish_primitive(
                TransitionEvent.TIMED_OUT,
                observation,
                reason="primitive execution timeout",
            )

        commands: list[ExecutionCommand] = []
        if not self._motion_started:
            if self._actions_complete(ActionTiming.BEFORE_MOTION, observation):
                commands.extend(self._start_motion(candidate))
        if self._motion_started:
            commands.extend(
                self._issue_eligible_actions(
                    candidate,
                    ActionTiming.DURING_MOTION,
                    observation,
                    feedback.motion_phase,
                )
            )
            commands.extend(
                self._issue_eligible_actions(
                    candidate,
                    ActionTiming.ON_EVENT,
                    observation,
                    feedback.motion_phase,
                )
            )
            if feedback.motion_complete or candidate.motion is None:
                self._motion_complete = True
                commands.extend(
                    self._issue_eligible_actions(
                        candidate,
                        ActionTiming.AFTER_MOTION,
                        observation,
                        1.0,
                    )
                )

        if self._primitive_complete(candidate, observation):
            completed = self._finish_primitive(
                TransitionEvent.SUCCEEDED,
                observation,
                reason="declared completion and postconditions satisfied",
            )
            return ExecutionDecision(
                status=completed.status,
                current_id=completed.current_id,
                commands=tuple(commands) + completed.commands,
                event=completed.event,
                next_id=completed.next_id,
                reason=completed.reason,
            )
        return ExecutionDecision(
            status=self.status,
            current_id=self.current_id,
            commands=tuple(commands),
        )

    def cancel(
        self, observation: ExecutionObservation, reason: str = "external cancellation"
    ) -> ExecutionDecision:
        if self.status != ExecutorStatus.RUNNING or self.current is None:
            raise PrimitiveExecutionError("executor cancellation requires a running primitive")
        self._check_monotonic_time(observation)
        return self._finish_primitive(TransitionEvent.CANCELLED, observation, reason=reason)

    def _activate(
        self, primitive_id: str, observation: ExecutionObservation
    ) -> ExecutionDecision:
        candidate = self.program.primitive(primitive_id)
        self.current_id = primitive_id
        self._last_timestamp_s = observation.timestamp_s
        if observation.epoch != candidate.observation_epoch:
            return self._require_replan(
                observation,
                reason=(
                    f"cannot activate stale candidate: candidate={candidate.observation_epoch}, "
                    f"observed={observation.epoch}"
                ),
                next_id=primitive_id,
            )
        if not self._guards.evaluate(
            candidate.start_guard,
            observation,
            scope=f"{primitive_id}:start",
        ):
            self.status = ExecutorStatus.FAILED
            return ExecutionDecision(
                status=self.status,
                current_id=primitive_id,
                event=TransitionEvent.FAILED,
                reason="primitive start guard is not satisfied",
            )
        self.status = ExecutorStatus.RUNNING
        self._active_started_s = observation.timestamp_s
        self._motion_started = False
        self._motion_complete = candidate.motion is None
        self._issued_actions.clear()
        self._guards.reset_scope(f"{primitive_id}:completion")
        self._guards.reset_scope(f"{primitive_id}:postcondition")
        commands = self._issue_eligible_actions(
            candidate,
            ActionTiming.BEFORE_MOTION,
            observation,
            0.0,
        )
        if self._actions_complete(ActionTiming.BEFORE_MOTION, observation):
            commands.extend(self._start_motion(candidate))
        return ExecutionDecision(
            status=self.status,
            current_id=primitive_id,
            commands=tuple(commands),
        )

    def _start_motion(self, candidate: PrimitiveCandidate) -> list[ExecutionCommand]:
        self._motion_started = True
        if candidate.motion is None:
            self._motion_complete = True
            return []
        return [
            ExecutionCommand(
                ExecutionCommandType.START_MOTION,
                candidate.primitive_id,
                motion=candidate.motion,
            )
        ]

    def _issue_eligible_actions(
        self,
        candidate: PrimitiveCandidate,
        timing: ActionTiming,
        observation: ExecutionObservation,
        phase: float,
    ) -> list[ExecutionCommand]:
        commands = []
        for index, action in enumerate(candidate.end_effector_actions):
            if index in self._issued_actions or action.timing != timing:
                continue
            eligible = True
            if timing == ActionTiming.DURING_MOTION:
                eligible = phase >= action.phase_start
            elif timing == ActionTiming.ON_EVENT:
                eligible = self._guards.evaluate(
                    action.trigger,
                    observation,
                    scope=f"{candidate.primitive_id}:action:{index}:trigger",
                )
            if not eligible:
                continue
            self._issued_actions.add(index)
            commands.append(
                ExecutionCommand(
                    ExecutionCommandType.END_EFFECTOR,
                    candidate.primitive_id,
                    end_effector_action=action,
                )
            )
        return commands

    def _actions_complete(
        self, timing: ActionTiming, observation: ExecutionObservation
    ) -> bool:
        candidate = self.current
        assert candidate is not None
        relevant = [
            (index, action)
            for index, action in enumerate(candidate.end_effector_actions)
            if action.timing == timing
        ]
        for index, action in relevant:
            if index not in self._issued_actions:
                return False
            if not self._guards.evaluate(
                action.completion,
                observation,
                scope=f"{candidate.primitive_id}:action:{index}:completion",
            ):
                return False
        return True

    def _primitive_complete(
        self, candidate: PrimitiveCandidate, observation: ExecutionObservation
    ) -> bool:
        if not self._motion_complete:
            return False
        for timing in ActionTiming:
            if not self._actions_complete(timing, observation):
                return False
        if not self._guards.evaluate(
            candidate.completion_guard,
            observation,
            scope=f"{candidate.primitive_id}:completion",
        ):
            return False
        return self._guards.evaluate(
            candidate.postcondition,
            observation,
            scope=f"{candidate.primitive_id}:postcondition",
        )

    def _finish_primitive(
        self,
        event: TransitionEvent,
        observation: ExecutionObservation,
        *,
        reason: str,
    ) -> ExecutionDecision:
        candidate = self.current
        assert candidate is not None
        stop_commands: tuple[ExecutionCommand, ...] = ()
        if self._motion_started and not self._motion_complete:
            stop_commands = (
                ExecutionCommand(
                    ExecutionCommandType.STOP_MOTION,
                    candidate.primitive_id,
                    reason=reason,
                ),
            )
        enabled = []
        for index, transition in enumerate(self.program.transitions):
            if transition.source_id != candidate.primitive_id or transition.event != event:
                continue
            traversals = self._traversals.get(index, 0)
            if traversals >= transition.max_traversals:
                continue
            if self._guards.evaluate(
                transition.guard,
                observation,
                scope=f"transition:{index}",
            ):
                enabled.append((index, transition))
        if len(enabled) > 1:
            raise PrimitiveExecutionError(
                f"multiple transitions enabled for {candidate.primitive_id}:{event.value}"
            )
        if not enabled:
            terminal = {
                TransitionEvent.SUCCEEDED: ExecutorStatus.SUCCEEDED,
                TransitionEvent.FAILED: ExecutorStatus.FAILED,
                TransitionEvent.TIMED_OUT: ExecutorStatus.TIMED_OUT,
                TransitionEvent.CANCELLED: ExecutorStatus.CANCELLED,
            }[event]
            self.status = terminal
            return ExecutionDecision(
                status=terminal,
                current_id=candidate.primitive_id,
                commands=stop_commands,
                event=event,
                reason=reason,
            )
        index, transition = enabled[0]
        self._traversals[index] = self._traversals.get(index, 0) + 1
        if transition.target_id is None:
            terminal = {
                TransitionEvent.SUCCEEDED: ExecutorStatus.SUCCEEDED,
                TransitionEvent.FAILED: ExecutorStatus.FAILED,
                TransitionEvent.TIMED_OUT: ExecutorStatus.TIMED_OUT,
                TransitionEvent.CANCELLED: ExecutorStatus.CANCELLED,
            }[event]
            self.status = terminal
            return ExecutionDecision(
                status=terminal,
                current_id=candidate.primitive_id,
                commands=stop_commands,
                event=event,
                reason=reason,
            )
        if transition.requires_reobservation or transition.invalidates_downstream:
            return self._require_replan(
                observation,
                reason=transition.label or "transition requires reobservation",
                next_id=transition.target_id,
                commands=stop_commands,
                event=event,
            )
        activated = self._activate(transition.target_id, observation)
        return ExecutionDecision(
            status=activated.status,
            current_id=activated.current_id,
            commands=stop_commands + activated.commands,
            event=event,
            next_id=transition.target_id,
            reason=reason,
        )

    def _require_replan(
        self,
        observation: ExecutionObservation,
        *,
        reason: str,
        next_id: Optional[str] = None,
        commands: tuple[ExecutionCommand, ...] = (),
        event: Optional[TransitionEvent] = None,
    ) -> ExecutionDecision:
        current_id = self.current_id
        request = ExecutionCommand(
            ExecutionCommandType.REQUEST_REOBSERVATION,
            current_id or next_id or self.program.start_id,
            reason=reason,
        )
        stop: tuple[ExecutionCommand, ...] = ()
        already_stopped = any(
            command.command == ExecutionCommandType.STOP_MOTION for command in commands
        )
        if (
            self._motion_started
            and not self._motion_complete
            and current_id is not None
            and not already_stopped
        ):
            stop = (
                ExecutionCommand(
                    ExecutionCommandType.STOP_MOTION,
                    current_id,
                    reason=reason,
                ),
            )
        self.status = ExecutorStatus.REPLAN_REQUIRED
        self._last_timestamp_s = observation.timestamp_s
        return ExecutionDecision(
            status=self.status,
            current_id=current_id,
            commands=commands + stop + (request,),
            event=event,
            next_id=next_id,
            reason=reason,
        )

    def _timeout(self, candidate: PrimitiveCandidate) -> float:
        if candidate.timeout_s is not None:
            return candidate.timeout_s
        motion_duration = 0.0 if candidate.motion is None else candidate.motion.duration_s
        command_timeouts = [
            action.command.timeout_s
            for action in candidate.end_effector_actions
            if action.command.timeout_s is not None
        ]
        return max(self.default_timeout_s, motion_duration + sum(command_timeouts))

    def _check_monotonic_time(self, observation: ExecutionObservation) -> None:
        if observation.timestamp_s < self._last_timestamp_s:
            raise PrimitiveExecutionError("observation timestamps must be monotonic")
        self._last_timestamp_s = observation.timestamp_s
