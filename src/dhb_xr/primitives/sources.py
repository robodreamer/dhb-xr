"""Uniform candidate-source interface for analytic and future learned backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

import numpy as np

from dhb_xr.primitives.analytic import (
    arc_path,
    minimum_jerk_path,
    screw_path,
    via_point_path,
)
from dhb_xr.primitives.contracts import (
    ContactMode,
    GuardExpression,
    MotionPath,
    PrimitiveCandidate,
    PrimitiveContractError,
    PrimitiveEvidence,
    PrimitiveFamily,
    ScheduledEndEffectorAction,
)


@dataclass(frozen=True)
class RelationalPoseAnchor:
    """One scene-grounded intermediate pose constraint.

    ``phase`` is normalized path progress, not wall-clock time.  Endpoints are
    carried by :class:`PrimitiveRequest`; anchors therefore live strictly
    inside ``(0, 1)`` and must name the relation that grounded them.
    """

    phase: float
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    relation_id: str

    def __post_init__(self) -> None:
        phase = float(self.phase)
        relation_id = str(self.relation_id).strip()
        if not np.isfinite(phase) or not 0.0 < phase < 1.0:
            raise PrimitiveContractError("anchor phase must be finite and strictly inside (0, 1)")
        if not relation_id:
            raise PrimitiveContractError("anchor relation_id must be non-empty")
        pose = MotionPath(
            np.stack([self.position, self.position]),
            np.stack([self.quaternion_wxyz, self.quaternion_wxyz]),
            np.array([0.0, 1.0]),
        )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "position", pose.positions[0])
        object.__setattr__(self, "quaternion_wxyz", pose.quaternions_wxyz[0])
        object.__setattr__(self, "relation_id", relation_id)


@dataclass(frozen=True)
class PrimitiveRequest:
    """A grounded request shared by interchangeable candidate sources."""

    primitive_id: str
    start_position: np.ndarray
    start_quaternion_wxyz: np.ndarray
    goal_position: np.ndarray
    goal_quaternion_wxyz: np.ndarray
    samples: int = 50
    duration_s: float = 2.0
    frame_id: str = "world"
    observation_epoch: int = 0
    reference_frames: Mapping[str, str] = field(default_factory=dict)
    relational_anchors: tuple[RelationalPoseAnchor, ...] = ()
    end_effector_actions: tuple[ScheduledEndEffectorAction, ...] = ()
    contact_mode: ContactMode = ContactMode.FREE_SPACE
    start_guard: Optional[GuardExpression] = None
    completion_guard: Optional[GuardExpression] = None
    postcondition: Optional[GuardExpression] = None
    timeout_s: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        primitive_id = str(self.primitive_id).strip()
        if not primitive_id:
            raise PrimitiveContractError("primitive_id must be non-empty")
        samples = int(self.samples)
        if samples < 2:
            raise PrimitiveContractError("samples must be at least two")
        duration_s = float(self.duration_s)
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise PrimitiveContractError("duration_s must be finite and positive")
        observation_epoch = int(self.observation_epoch)
        if observation_epoch < 0:
            raise PrimitiveContractError("observation_epoch must be non-negative")
        boundary = MotionPath(
            np.stack([self.start_position, self.goal_position]),
            np.stack([self.start_quaternion_wxyz, self.goal_quaternion_wxyz]),
            np.array([0.0, duration_s]),
            frame_id=self.frame_id,
        )
        actions = tuple(self.end_effector_actions)
        if not all(isinstance(action, ScheduledEndEffectorAction) for action in actions):
            raise PrimitiveContractError(
                "end_effector_actions must contain ScheduledEndEffectorAction values"
            )
        contact_mode = (
            self.contact_mode
            if isinstance(self.contact_mode, ContactMode)
            else ContactMode(self.contact_mode)
        )
        frames = {}
        for role, frame in self.reference_frames.items():
            role_value = str(role).strip()
            frame_value = str(frame).strip()
            if not role_value or not frame_value:
                raise PrimitiveContractError("reference frame roles and ids must be non-empty")
            frames[role_value] = frame_value
        anchors = tuple(self.relational_anchors)
        if not all(isinstance(anchor, RelationalPoseAnchor) for anchor in anchors):
            raise PrimitiveContractError(
                "relational_anchors must contain RelationalPoseAnchor values"
            )
        phases = [anchor.phase for anchor in anchors]
        if phases != sorted(phases) or len(phases) != len(set(phases)):
            raise PrimitiveContractError(
                "relational anchor phases must be unique and strictly increasing"
            )
        for field_name in ("start_guard", "completion_guard", "postcondition"):
            guard = getattr(self, field_name)
            if guard is not None and not isinstance(guard, GuardExpression):
                raise PrimitiveContractError(f"{field_name} must be GuardExpression")
        timeout_s = None if self.timeout_s is None else float(self.timeout_s)
        if timeout_s is not None and (not np.isfinite(timeout_s) or timeout_s <= 0.0):
            raise PrimitiveContractError("timeout_s must be finite and positive")
        object.__setattr__(self, "primitive_id", primitive_id)
        object.__setattr__(self, "start_position", boundary.positions[0])
        object.__setattr__(self, "start_quaternion_wxyz", boundary.quaternions_wxyz[0])
        object.__setattr__(self, "goal_position", boundary.positions[1])
        object.__setattr__(self, "goal_quaternion_wxyz", boundary.quaternions_wxyz[1])
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "duration_s", duration_s)
        object.__setattr__(self, "frame_id", boundary.frame_id)
        object.__setattr__(self, "observation_epoch", observation_epoch)
        object.__setattr__(self, "reference_frames", frames)
        object.__setattr__(self, "relational_anchors", anchors)
        object.__setattr__(self, "end_effector_actions", actions)
        object.__setattr__(self, "contact_mode", contact_mode)
        object.__setattr__(self, "timeout_s", timeout_s)
        object.__setattr__(self, "metadata", dict(self.metadata))


class PrimitiveSource(Protocol):
    """A proposal source; returned candidates are not safety-approved."""

    @property
    def source_id(self) -> str:
        """Stable source identifier for provenance and experiment attribution."""

    def propose(self, request: PrimitiveRequest) -> tuple[PrimitiveCandidate, ...]:
        """Return zero or more unapproved candidates."""


@dataclass(frozen=True)
class AnalyticPrimitiveSource:
    """Configured source for the built-in analytic primitive families."""

    family: PrimitiveFamily
    evidence: PrimitiveEvidence
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        family = (
            self.family
            if isinstance(self.family, PrimitiveFamily)
            else PrimitiveFamily(self.family)
        )
        supported = {
            PrimitiveFamily.MINIMUM_JERK,
            PrimitiveFamily.ARC,
            PrimitiveFamily.SCREW,
            PrimitiveFamily.VIA_POINT,
        }
        if family not in supported:
            raise PrimitiveContractError(
                f"analytic source does not support primitive family {family.value}"
            )
        if not isinstance(self.evidence, PrimitiveEvidence):
            raise PrimitiveContractError("evidence must be PrimitiveEvidence")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "parameters", dict(self.parameters))

    @property
    def source_id(self) -> str:
        return f"analytic:{self.family.value}"

    def propose(self, request: PrimitiveRequest) -> tuple[PrimitiveCandidate, ...]:
        path = self._generate(request)
        metadata = dict(request.metadata)
        metadata.update({"candidate_source": self.source_id})
        return (
            PrimitiveCandidate(
                primitive_id=request.primitive_id,
                family=self.family,
                motion=path,
                end_effector_actions=request.end_effector_actions,
                contact_mode=request.contact_mode,
                reference_frames=request.reference_frames,
                start_guard=request.start_guard,
                completion_guard=request.completion_guard,
                postcondition=request.postcondition,
                timeout_s=request.timeout_s,
                observation_epoch=request.observation_epoch,
                evidence=self.evidence,
                metadata=metadata,
            ),
        )

    def _generate(self, request: PrimitiveRequest) -> MotionPath:
        if self.family == PrimitiveFamily.MINIMUM_JERK:
            return minimum_jerk_path(
                request.start_position,
                request.start_quaternion_wxyz,
                request.goal_position,
                request.goal_quaternion_wxyz,
                samples=request.samples,
                duration_s=request.duration_s,
                frame_id=request.frame_id,
            )
        if self.family == PrimitiveFamily.ARC:
            return arc_path(
                request.start_position,
                request.start_quaternion_wxyz,
                goal_quaternion_wxyz=request.goal_quaternion_wxyz,
                center=np.asarray(self.parameters["center"]),
                axis=np.asarray(self.parameters["axis"]),
                angle_rad=float(self.parameters["angle_rad"]),
                rotate_tool_with_arc=bool(self.parameters.get("rotate_tool_with_arc", True)),
                samples=request.samples,
                duration_s=request.duration_s,
                frame_id=request.frame_id,
            )
        if self.family == PrimitiveFamily.SCREW:
            return screw_path(
                request.start_position,
                request.start_quaternion_wxyz,
                axis_point=np.asarray(self.parameters["axis_point"]),
                axis=np.asarray(self.parameters["axis"]),
                angle_rad=float(self.parameters["angle_rad"]),
                translation_m=float(self.parameters["translation_m"]),
                rotate_tool=bool(self.parameters.get("rotate_tool", True)),
                samples=request.samples,
                duration_s=request.duration_s,
                frame_id=request.frame_id,
            )
        if self.family == PrimitiveFamily.VIA_POINT:
            positions = [request.start_position]
            positions.extend(np.asarray(value) for value in self.parameters.get("positions", ()))
            positions.append(request.goal_position)
            quaternions = [request.start_quaternion_wxyz]
            quaternions.extend(
                np.asarray(value) for value in self.parameters.get("quaternions_wxyz", ())
            )
            quaternions.append(request.goal_quaternion_wxyz)
            return via_point_path(
                positions,
                quaternions,
                samples_per_segment=int(
                    self.parameters.get("samples_per_segment", request.samples)
                ),
                duration_s=request.duration_s,
                frame_id=request.frame_id,
            )
        raise AssertionError(f"unhandled analytic family {self.family.value}")
