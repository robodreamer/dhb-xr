"""Dependency-light contracts for reusable manipulation primitive programs.

The DHB descriptor remains a geometric representation.  These contracts bind a
geometric path to end-effector semantics, evidence, observation epochs, and
guarded transitions without assigning safety or task-success authority to the
motion source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np


class PrimitiveContractError(ValueError):
    """Raised when a primitive or program violates its executable contract."""


class PrimitiveFamily(str, Enum):
    """Provenance-aware motion primitive families."""

    HOLD = "hold"
    END_EFFECTOR_ONLY = "end_effector_only"
    MINIMUM_JERK = "minimum_jerk"
    ARC = "arc"
    SCREW = "screw"
    VIA_POINT = "via_point"
    CONSTRAINED_OCP = "constrained_ocp"
    DHB_MEMORY = "dhb_memory"
    LEARNED_FLOW = "learned_flow"
    LEARNED_DIFFUSION = "learned_diffusion"
    CUSTOM = "custom"


class EvidenceLevel(str, Enum):
    """Ordered evidence labels; ordering is exposed through ``rank``."""

    PLANNED = "planned"
    SIM_VERIFIED = "sim_verified"
    REAL_VERIFIED = "real_verified"

    @property
    def rank(self) -> int:
        return {
            EvidenceLevel.PLANNED: 0,
            EvidenceLevel.SIM_VERIFIED: 1,
            EvidenceLevel.REAL_VERIFIED: 2,
        }[self]


class ContactMode(str, Enum):
    """Declared contact regime for one primitive."""

    FREE_SPACE = "free_space"
    GUARDED = "guarded"
    SUSTAINED = "sustained"
    RELEASE = "release"


class EndEffectorCommandType(str, Enum):
    """Embodiment-neutral end-effector commands."""

    OPEN = "open"
    RELEASE = "release"
    PRESHAPE = "preshape"
    CLOSE_TO_WIDTH = "close_to_width"
    CLOSE_UNTIL_CONTACT = "close_until_contact"
    HOLD = "hold"
    MAINTAIN = "maintain"
    VACUUM_ON = "vacuum_on"
    VACUUM_OFF = "vacuum_off"


class ActionTiming(str, Enum):
    """How an end-effector action is synchronized with Cartesian motion."""

    BEFORE_MOTION = "before_motion"
    DURING_MOTION = "during_motion"
    AFTER_MOTION = "after_motion"
    ON_EVENT = "on_event"


class Comparison(str, Enum):
    """Serializable comparisons supported by the guarded executor."""

    TRUTHY = "truthy"
    FALSY = "falsy"
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GE = "ge"
    LT = "lt"
    LE = "le"
    WITHIN = "within"


class GuardMode(str, Enum):
    """Combination rule for guard conditions."""

    ALL = "all"
    ANY = "any"


class TransitionEvent(str, Enum):
    """Terminal event emitted by one primitive execution."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


def _as_enum(value: Any, enum_type: type[Enum], field_name: str) -> Enum:
    try:
        return value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        options = ", ".join(str(member.value) for member in enum_type)
        raise PrimitiveContractError(f"{field_name} must be one of: {options}") from exc


def _nonempty(value: str, field_name: str) -> str:
    result = str(value).strip()
    if not result:
        raise PrimitiveContractError(f"{field_name} must be non-empty")
    return result


def _optional_nonempty(value: Optional[str], field_name: str) -> Optional[str]:
    if value is None:
        return None
    return _nonempty(value, field_name)


def _finite_nonnegative(value: Optional[float], field_name: str) -> Optional[float]:
    if value is None:
        return None
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise PrimitiveContractError(f"{field_name} must be finite and non-negative")
    return result


def _json_safe(value: Any, path: str = "value") -> Any:
    """Return a detached JSON-compatible value or fail without stringifying it."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise PrimitiveContractError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PrimitiveContractError(f"{path} contains a non-string mapping key")
            result[key] = _json_safe(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise PrimitiveContractError(f"{path} contains unsupported type {type(value).__name__}")


def _metadata(value: Mapping[str, Any], field_name: str = "metadata") -> dict[str, Any]:
    safe = _json_safe(value, field_name)
    if not isinstance(safe, dict):
        raise PrimitiveContractError(f"{field_name} must be a mapping")
    return safe


@dataclass(frozen=True)
class MotionPath:
    """A finite, timestamped SE(3) path in scalar-first quaternion convention."""

    positions: np.ndarray
    quaternions_wxyz: np.ndarray
    timestamps_s: np.ndarray
    frame_id: str = "world"

    def __post_init__(self) -> None:
        positions = np.array(self.positions, dtype=np.float64, copy=True)
        quaternions = np.array(self.quaternions_wxyz, dtype=np.float64, copy=True)
        timestamps = np.array(self.timestamps_s, dtype=np.float64, copy=True)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 2:
            raise PrimitiveContractError("positions must have shape (N, 3) with N >= 2")
        if quaternions.shape != (len(positions), 4):
            raise PrimitiveContractError("quaternions_wxyz must have shape (N, 4)")
        if timestamps.shape != (len(positions),):
            raise PrimitiveContractError("timestamps_s must have shape (N,)")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(quaternions)):
            raise PrimitiveContractError("motion poses must be finite")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
            raise PrimitiveContractError("timestamps_s must be finite and strictly increasing")
        if timestamps[0] < 0.0:
            raise PrimitiveContractError("timestamps_s must begin at or after zero")
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(norms <= 1e-10):
            raise PrimitiveContractError("quaternions_wxyz must be non-zero")
        quaternions /= norms[:, None]
        for index in range(1, len(quaternions)):
            if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
                quaternions[index] *= -1.0
        positions.setflags(write=False)
        quaternions.setflags(write=False)
        timestamps.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "quaternions_wxyz", quaternions)
        object.__setattr__(self, "timestamps_s", timestamps)
        object.__setattr__(self, "frame_id", _nonempty(self.frame_id, "frame_id"))

    @property
    def samples(self) -> int:
        return len(self.positions)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1] - self.timestamps_s[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": self.positions.tolist(),
            "quaternions_wxyz": self.quaternions_wxyz.tolist(),
            "timestamps_s": self.timestamps_s.tolist(),
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MotionPath:
        return cls(
            positions=np.asarray(payload["positions"], dtype=np.float64),
            quaternions_wxyz=np.asarray(payload["quaternions_wxyz"], dtype=np.float64),
            timestamps_s=np.asarray(payload["timestamps_s"], dtype=np.float64),
            frame_id=str(payload.get("frame_id", "world")),
        )


@dataclass(frozen=True)
class GuardCondition:
    """One observation-based condition with optional tolerance and dwell time."""

    signal: str
    comparison: Comparison
    expected: Any = None
    tolerance: Optional[float] = None
    dwell_s: float = 0.0
    label: Optional[str] = None

    def __post_init__(self) -> None:
        comparison = _as_enum(self.comparison, Comparison, "comparison")
        tolerance = _finite_nonnegative(self.tolerance, "tolerance")
        dwell_s = _finite_nonnegative(self.dwell_s, "dwell_s")
        assert dwell_s is not None
        if comparison not in (Comparison.TRUTHY, Comparison.FALSY) and self.expected is None:
            raise PrimitiveContractError(f"comparison {comparison.value} requires expected")
        if comparison == Comparison.WITHIN and tolerance is None:
            raise PrimitiveContractError("within comparison requires tolerance")
        expected = _json_safe(self.expected, "expected")
        object.__setattr__(self, "signal", _nonempty(self.signal, "signal"))
        object.__setattr__(self, "comparison", comparison)
        object.__setattr__(self, "expected", expected)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "dwell_s", dwell_s)
        object.__setattr__(self, "label", _optional_nonempty(self.label, "label"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "comparison": self.comparison.value,
            "expected": self.expected,
            "tolerance": self.tolerance,
            "dwell_s": self.dwell_s,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GuardCondition:
        return cls(
            signal=str(payload["signal"]),
            comparison=Comparison(payload["comparison"]),
            expected=payload.get("expected"),
            tolerance=payload.get("tolerance"),
            dwell_s=float(payload.get("dwell_s", 0.0)),
            label=payload.get("label"),
        )


@dataclass(frozen=True)
class GuardExpression:
    """An all/any expression over serializable conditions."""

    conditions: tuple[GuardCondition, ...]
    mode: GuardMode = GuardMode.ALL

    def __post_init__(self) -> None:
        conditions = tuple(self.conditions)
        if not conditions or not all(isinstance(item, GuardCondition) for item in conditions):
            raise PrimitiveContractError("guard conditions must contain GuardCondition values")
        object.__setattr__(self, "conditions", conditions)
        object.__setattr__(self, "mode", _as_enum(self.mode, GuardMode, "guard mode"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "conditions": [condition.to_dict() for condition in self.conditions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GuardExpression:
        return cls(
            conditions=tuple(
                GuardCondition.from_dict(condition) for condition in payload["conditions"]
            ),
            mode=GuardMode(payload.get("mode", GuardMode.ALL.value)),
        )


@dataclass(frozen=True)
class EndEffectorCommand:
    """An embodiment-neutral gripper or tool command."""

    command: EndEffectorCommandType
    target_width_m: Optional[float] = None
    max_force_n: Optional[float] = None
    speed_m_s: Optional[float] = None
    timeout_s: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        command = _as_enum(self.command, EndEffectorCommandType, "end-effector command")
        width = _finite_nonnegative(self.target_width_m, "target_width_m")
        force = _finite_nonnegative(self.max_force_n, "max_force_n")
        speed = _finite_nonnegative(self.speed_m_s, "speed_m_s")
        timeout = _finite_nonnegative(self.timeout_s, "timeout_s")
        if command in (
            EndEffectorCommandType.PRESHAPE,
            EndEffectorCommandType.CLOSE_TO_WIDTH,
        ) and width is None:
            raise PrimitiveContractError(f"{command.value} requires target_width_m")
        if command == EndEffectorCommandType.CLOSE_UNTIL_CONTACT and (
            force is None or force <= 0.0
        ):
            raise PrimitiveContractError("close_until_contact requires positive max_force_n")
        if command in (EndEffectorCommandType.VACUUM_ON, EndEffectorCommandType.VACUUM_OFF) and (
            width is not None or speed is not None
        ):
            raise PrimitiveContractError("vacuum commands cannot specify finger width or speed")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "target_width_m", width)
        object.__setattr__(self, "max_force_n", force)
        object.__setattr__(self, "speed_m_s", speed)
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.value,
            "target_width_m": self.target_width_m,
            "max_force_n": self.max_force_n,
            "speed_m_s": self.speed_m_s,
            "timeout_s": self.timeout_s,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EndEffectorCommand:
        return cls(
            command=EndEffectorCommandType(payload["command"]),
            target_width_m=payload.get("target_width_m"),
            max_force_n=payload.get("max_force_n"),
            speed_m_s=payload.get("speed_m_s"),
            timeout_s=payload.get("timeout_s"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class ScheduledEndEffectorAction:
    """An end-effector action synchronized to motion phase or an event."""

    command: EndEffectorCommand
    timing: ActionTiming
    phase_start: float = 0.0
    phase_end: float = 1.0
    trigger: Optional[GuardExpression] = None
    completion: Optional[GuardExpression] = None

    def __post_init__(self) -> None:
        if not isinstance(self.command, EndEffectorCommand):
            raise PrimitiveContractError("command must be EndEffectorCommand")
        timing = _as_enum(self.timing, ActionTiming, "action timing")
        phase_start = float(self.phase_start)
        phase_end = float(self.phase_end)
        if (
            not np.isfinite(phase_start)
            or not np.isfinite(phase_end)
            or not 0.0 <= phase_start <= phase_end <= 1.0
        ):
            raise PrimitiveContractError("action phase must satisfy 0 <= start <= end <= 1")
        if timing == ActionTiming.ON_EVENT and self.trigger is None:
            raise PrimitiveContractError("on_event action requires a trigger guard")
        if self.trigger is not None and not isinstance(self.trigger, GuardExpression):
            raise PrimitiveContractError("trigger must be GuardExpression")
        if self.completion is not None and not isinstance(self.completion, GuardExpression):
            raise PrimitiveContractError("completion must be GuardExpression")
        object.__setattr__(self, "timing", timing)
        object.__setattr__(self, "phase_start", phase_start)
        object.__setattr__(self, "phase_end", phase_end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.to_dict(),
            "timing": self.timing.value,
            "phase_start": self.phase_start,
            "phase_end": self.phase_end,
            "trigger": None if self.trigger is None else self.trigger.to_dict(),
            "completion": None if self.completion is None else self.completion.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScheduledEndEffectorAction:
        trigger = payload.get("trigger")
        completion = payload.get("completion")
        return cls(
            command=EndEffectorCommand.from_dict(payload["command"]),
            timing=ActionTiming(payload["timing"]),
            phase_start=float(payload.get("phase_start", 0.0)),
            phase_end=float(payload.get("phase_end", 1.0)),
            trigger=None if trigger is None else GuardExpression.from_dict(trigger),
            completion=None if completion is None else GuardExpression.from_dict(completion),
        )


@dataclass(frozen=True)
class PrimitiveEvidence:
    """Evidence, provenance, and compatibility carried with a candidate."""

    level: EvidenceLevel
    provenance: str
    source_revision: Optional[str] = None
    artifact_hash: Optional[str] = None
    dataset: Optional[str] = None
    license_id: Optional[str] = None
    compatible_robot_controller_revisions: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        revisions = tuple(
            _nonempty(item, "compatible revision")
            for item in self.compatible_robot_controller_revisions
        )
        object.__setattr__(self, "level", _as_enum(self.level, EvidenceLevel, "evidence level"))
        object.__setattr__(self, "provenance", _nonempty(self.provenance, "provenance"))
        object.__setattr__(
            self, "source_revision", _optional_nonempty(self.source_revision, "source_revision")
        )
        object.__setattr__(
            self, "artifact_hash", _optional_nonempty(self.artifact_hash, "artifact_hash")
        )
        object.__setattr__(self, "dataset", _optional_nonempty(self.dataset, "dataset"))
        object.__setattr__(self, "license_id", _optional_nonempty(self.license_id, "license_id"))
        object.__setattr__(self, "compatible_robot_controller_revisions", revisions)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level.value,
            "provenance": self.provenance,
            "source_revision": self.source_revision,
            "artifact_hash": self.artifact_hash,
            "dataset": self.dataset,
            "license_id": self.license_id,
            "compatible_robot_controller_revisions": list(
                self.compatible_robot_controller_revisions
            ),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrimitiveEvidence:
        return cls(
            level=EvidenceLevel(payload["level"]),
            provenance=str(payload["provenance"]),
            source_revision=payload.get("source_revision"),
            artifact_hash=payload.get("artifact_hash"),
            dataset=payload.get("dataset"),
            license_id=payload.get("license_id"),
            compatible_robot_controller_revisions=tuple(
                payload.get("compatible_robot_controller_revisions", ())
            ),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class PrimitiveCandidate:
    """One unapproved primitive candidate and all context needed to gate it."""

    primitive_id: str
    family: PrimitiveFamily
    evidence: PrimitiveEvidence
    motion: Optional[MotionPath] = None
    end_effector_actions: tuple[ScheduledEndEffectorAction, ...] = ()
    contact_mode: ContactMode = ContactMode.FREE_SPACE
    reference_frames: Mapping[str, str] = field(default_factory=dict)
    start_guard: Optional[GuardExpression] = None
    completion_guard: Optional[GuardExpression] = None
    postcondition: Optional[GuardExpression] = None
    timeout_s: Optional[float] = None
    observation_epoch: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family = _as_enum(self.family, PrimitiveFamily, "primitive family")
        contact_mode = _as_enum(self.contact_mode, ContactMode, "contact mode")
        if not isinstance(self.evidence, PrimitiveEvidence):
            raise PrimitiveContractError("evidence must be PrimitiveEvidence")
        if self.motion is not None and not isinstance(self.motion, MotionPath):
            raise PrimitiveContractError("motion must be MotionPath or None")
        motion_optional = family in (PrimitiveFamily.HOLD, PrimitiveFamily.END_EFFECTOR_ONLY)
        if self.motion is None and not motion_optional:
            raise PrimitiveContractError(f"{family.value} requires a motion path")
        if self.motion is None and not self.end_effector_actions and family != PrimitiveFamily.HOLD:
            raise PrimitiveContractError("motionless primitive requires an end-effector action")
        actions = tuple(self.end_effector_actions)
        if not all(isinstance(item, ScheduledEndEffectorAction) for item in actions):
            raise PrimitiveContractError(
                "end_effector_actions must contain ScheduledEndEffectorAction values"
            )
        frames = {}
        for role, frame in self.reference_frames.items():
            frames[_nonempty(role, "reference frame role")] = _nonempty(
                frame, "reference frame id"
            )
        for field_name in ("start_guard", "completion_guard", "postcondition"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, GuardExpression):
                raise PrimitiveContractError(f"{field_name} must be GuardExpression")
        observation_epoch = int(self.observation_epoch)
        if observation_epoch < 0:
            raise PrimitiveContractError("observation_epoch must be non-negative")
        timeout_s = _finite_nonnegative(self.timeout_s, "timeout_s")
        if timeout_s == 0.0:
            raise PrimitiveContractError("timeout_s must be positive when specified")
        object.__setattr__(self, "primitive_id", _nonempty(self.primitive_id, "primitive_id"))
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "end_effector_actions", actions)
        object.__setattr__(self, "contact_mode", contact_mode)
        object.__setattr__(self, "reference_frames", frames)
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "observation_epoch", observation_epoch)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "family": self.family.value,
            "evidence": self.evidence.to_dict(),
            "motion": None if self.motion is None else self.motion.to_dict(),
            "end_effector_actions": [action.to_dict() for action in self.end_effector_actions],
            "contact_mode": self.contact_mode.value,
            "reference_frames": dict(self.reference_frames),
            "start_guard": None if self.start_guard is None else self.start_guard.to_dict(),
            "completion_guard": (
                None if self.completion_guard is None else self.completion_guard.to_dict()
            ),
            "postcondition": None if self.postcondition is None else self.postcondition.to_dict(),
            "timeout_s": self.timeout_s,
            "observation_epoch": self.observation_epoch,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrimitiveCandidate:
        motion = payload.get("motion")
        start_guard = payload.get("start_guard")
        completion_guard = payload.get("completion_guard")
        postcondition = payload.get("postcondition")
        return cls(
            primitive_id=str(payload["primitive_id"]),
            family=PrimitiveFamily(payload["family"]),
            evidence=PrimitiveEvidence.from_dict(payload["evidence"]),
            motion=None if motion is None else MotionPath.from_dict(motion),
            end_effector_actions=tuple(
                ScheduledEndEffectorAction.from_dict(action)
                for action in payload.get("end_effector_actions", ())
            ),
            contact_mode=ContactMode(payload.get("contact_mode", ContactMode.FREE_SPACE.value)),
            reference_frames=payload.get("reference_frames", {}),
            start_guard=None if start_guard is None else GuardExpression.from_dict(start_guard),
            completion_guard=(
                None
                if completion_guard is None
                else GuardExpression.from_dict(completion_guard)
            ),
            postcondition=(
                None if postcondition is None else GuardExpression.from_dict(postcondition)
            ),
            timeout_s=payload.get("timeout_s"),
            observation_epoch=int(payload.get("observation_epoch", 0)),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class PrimitiveTransition:
    """A bounded transition between primitive nodes."""

    source_id: str
    event: TransitionEvent
    target_id: Optional[str] = None
    guard: Optional[GuardExpression] = None
    requires_reobservation: bool = False
    invalidates_downstream: bool = False
    max_traversals: int = 1
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if self.guard is not None and not isinstance(self.guard, GuardExpression):
            raise PrimitiveContractError("transition guard must be GuardExpression")
        max_traversals = int(self.max_traversals)
        if max_traversals < 1:
            raise PrimitiveContractError("max_traversals must be at least one")
        object.__setattr__(self, "source_id", _nonempty(self.source_id, "source_id"))
        object.__setattr__(self, "event", _as_enum(self.event, TransitionEvent, "event"))
        object.__setattr__(self, "target_id", _optional_nonempty(self.target_id, "target_id"))
        object.__setattr__(self, "requires_reobservation", bool(self.requires_reobservation))
        object.__setattr__(self, "invalidates_downstream", bool(self.invalidates_downstream))
        object.__setattr__(self, "max_traversals", max_traversals)
        object.__setattr__(self, "label", _optional_nonempty(self.label, "label"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "event": self.event.value,
            "target_id": self.target_id,
            "guard": None if self.guard is None else self.guard.to_dict(),
            "requires_reobservation": self.requires_reobservation,
            "invalidates_downstream": self.invalidates_downstream,
            "max_traversals": self.max_traversals,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrimitiveTransition:
        guard = payload.get("guard")
        return cls(
            source_id=str(payload["source_id"]),
            event=TransitionEvent(payload["event"]),
            target_id=payload.get("target_id"),
            guard=None if guard is None else GuardExpression.from_dict(guard),
            requires_reobservation=bool(payload.get("requires_reobservation", False)),
            invalidates_downstream=bool(payload.get("invalidates_downstream", False)),
            max_traversals=int(payload.get("max_traversals", 1)),
            label=payload.get("label"),
        )


@dataclass(frozen=True)
class PrimitiveProgram:
    """A validated, serializable graph of guarded manipulation primitives."""

    program_id: str
    primitives: tuple[PrimitiveCandidate, ...]
    transitions: tuple[PrimitiveTransition, ...]
    start_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        primitives = tuple(self.primitives)
        transitions = tuple(self.transitions)
        if not primitives or not all(isinstance(item, PrimitiveCandidate) for item in primitives):
            raise PrimitiveContractError("program must contain PrimitiveCandidate values")
        if not all(isinstance(item, PrimitiveTransition) for item in transitions):
            raise PrimitiveContractError("transitions must contain PrimitiveTransition values")
        ids = [primitive.primitive_id for primitive in primitives]
        if len(ids) != len(set(ids)):
            raise PrimitiveContractError("primitive ids must be unique within a program")
        start_id = _nonempty(self.start_id, "start_id")
        if start_id not in ids:
            raise PrimitiveContractError(f"start_id {start_id!r} is not a program primitive")
        id_set = set(ids)
        transition_keys = set()
        adjacency: dict[str, set[str]] = {primitive_id: set() for primitive_id in ids}
        for transition in transitions:
            if transition.source_id not in id_set:
                raise PrimitiveContractError(
                    f"transition source {transition.source_id!r} is not a program primitive"
                )
            if transition.target_id is not None and transition.target_id not in id_set:
                raise PrimitiveContractError(
                    f"transition target {transition.target_id!r} is not a program primitive"
                )
            key = (transition.source_id, transition.event, transition.label)
            if key in transition_keys:
                raise PrimitiveContractError(
                    "transitions with the same source, event, and label must be unique"
                )
            transition_keys.add(key)
            if transition.target_id is not None:
                adjacency[transition.source_id].add(transition.target_id)

        reachable = {start_id}
        frontier = [start_id]
        while frontier:
            current = frontier.pop()
            for target in adjacency[current]:
                if target not in reachable:
                    reachable.add(target)
                    frontier.append(target)
        unreachable = sorted(id_set - reachable)
        if unreachable:
            raise PrimitiveContractError(
                "program contains unreachable primitives: " + ", ".join(unreachable)
            )

        object.__setattr__(self, "program_id", _nonempty(self.program_id, "program_id"))
        object.__setattr__(self, "primitives", primitives)
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "start_id", start_id)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def primitive(self, primitive_id: str) -> PrimitiveCandidate:
        for candidate in self.primitives:
            if candidate.primitive_id == primitive_id:
                return candidate
        raise KeyError(primitive_id)

    def outgoing(
        self, primitive_id: str, event: TransitionEvent
    ) -> tuple[PrimitiveTransition, ...]:
        return tuple(
            transition
            for transition in self.transitions
            if transition.source_id == primitive_id and transition.event == event
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "program_id": self.program_id,
            "primitives": [primitive.to_dict() for primitive in self.primitives],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "start_id": self.start_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrimitiveProgram:
        return cls(
            program_id=str(payload["program_id"]),
            primitives=tuple(
                PrimitiveCandidate.from_dict(primitive) for primitive in payload["primitives"]
            ),
            transitions=tuple(
                PrimitiveTransition.from_dict(transition)
                for transition in payload.get("transitions", ())
            ),
            start_id=str(payload["start_id"]),
            metadata=payload.get("metadata", {}),
        )


def guard_all(*conditions: GuardCondition) -> GuardExpression:
    """Convenience constructor for an all-conditions guard."""

    return GuardExpression(tuple(conditions), mode=GuardMode.ALL)


def guard_any(*conditions: GuardCondition) -> GuardExpression:
    """Convenience constructor for an any-condition guard."""

    return GuardExpression(tuple(conditions), mode=GuardMode.ANY)


def program_from_sequence(
    program_id: str,
    primitives: Sequence[PrimitiveCandidate],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> PrimitiveProgram:
    """Build a success-only sequence while retaining the explicit graph contract."""

    nodes = tuple(primitives)
    if not nodes:
        raise PrimitiveContractError("sequence must contain at least one primitive")
    transitions = tuple(
        PrimitiveTransition(
            source_id=current.primitive_id,
            event=TransitionEvent.SUCCEEDED,
            target_id=following.primitive_id,
        )
        for current, following in zip(nodes[:-1], nodes[1:])
    )
    return PrimitiveProgram(
        program_id=program_id,
        primitives=nodes,
        transitions=transitions,
        start_id=nodes[0].primitive_id,
        metadata={} if metadata is None else metadata,
    )
