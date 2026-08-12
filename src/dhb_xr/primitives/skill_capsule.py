"""Object-centric, provenance-preserving capsules for reusable SE(3) skills.

A capsule is deliberately richer than a DHB descriptor.  It retains the raw
world trajectories, the TCP trajectory expressed in the moving object frame,
sparse interaction relations, timing events, end-effector actions, and source
evidence.  DHB descriptors are derived indexes: they can be regenerated and
must never become the only copy of an observed motion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from dhb_xr.core import geometry as geom
from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.primitives.contracts import (
    MotionPath,
    PrimitiveCandidate,
    PrimitiveContractError,
    PrimitiveEvidence,
    ScheduledEndEffectorAction,
)
from dhb_xr.primitives.sources import PrimitiveRequest

CAPSULE_SCHEMA_VERSION = 1
_REQUIRED_PROVENANCE = (
    "source_uri",
    "source_revision",
    "action_convention",
    "license_id",
)


@dataclass(frozen=True)
class CapsuleRelation:
    """One sparse TCP-in-object relation or interaction event."""

    phase: float
    relation_id: str
    position: np.ndarray
    quaternion_wxyz: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        phase = float(self.phase)
        relation_id = str(self.relation_id).strip()
        if not np.isfinite(phase) or not 0.0 <= phase <= 1.0:
            raise PrimitiveContractError("capsule relation phase must lie in [0, 1]")
        if not relation_id:
            raise PrimitiveContractError("capsule relation_id must be non-empty")
        pose = MotionPath(
            np.stack([self.position, self.position]),
            np.stack([self.quaternion_wxyz, self.quaternion_wxyz]),
            np.array([0.0, 1.0]),
            frame_id="object",
        )
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "relation_id", relation_id)
        object.__setattr__(self, "position", pose.positions[0])
        object.__setattr__(self, "quaternion_wxyz", pose.quaternions_wxyz[0])
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "relation metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "relation_id": self.relation_id,
            "position": self.position.tolist(),
            "quaternion_wxyz": self.quaternion_wxyz.tolist(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CapsuleRelation:
        return cls(
            phase=float(payload["phase"]),
            relation_id=str(payload["relation_id"]),
            position=np.asarray(payload["position"], dtype=np.float64),
            quaternion_wxyz=np.asarray(payload["quaternion_wxyz"], dtype=np.float64),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class DhbDescriptorWindow:
    """DHB-DR descriptor over one declared phase interval."""

    scale_samples: int
    start_phase: float
    end_phase: float
    invariants: np.ndarray

    def __post_init__(self) -> None:
        scale = int(self.scale_samples)
        start = float(self.start_phase)
        end = float(self.end_phase)
        values = np.array(self.invariants, dtype=np.float64, copy=True)
        if scale < 3:
            raise PrimitiveContractError("descriptor scale_samples must be at least three")
        if not 0.0 <= start < end <= 1.0:
            raise PrimitiveContractError("descriptor phases must satisfy 0 <= start < end <= 1")
        if values.ndim != 2 or values.shape[1] != 8 or not np.all(np.isfinite(values)):
            raise PrimitiveContractError("descriptor invariants must be finite with shape (N, 8)")
        values.setflags(write=False)
        object.__setattr__(self, "scale_samples", scale)
        object.__setattr__(self, "start_phase", start)
        object.__setattr__(self, "end_phase", end)
        object.__setattr__(self, "invariants", values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale_samples": self.scale_samples,
            "start_phase": self.start_phase,
            "end_phase": self.end_phase,
            "invariants": self.invariants.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DhbDescriptorWindow:
        return cls(
            scale_samples=int(payload["scale_samples"]),
            start_phase=float(payload["start_phase"]),
            end_phase=float(payload["end_phase"]),
            invariants=np.asarray(payload["invariants"], dtype=np.float64),
        )


@dataclass(frozen=True)
class MultiscaleDhbDescriptor:
    """Local-to-global DHB windows derived from one object-motion path."""

    windows: tuple[DhbDescriptorWindow, ...]
    frame_id: str
    method: str = "double_reflection_position"

    def __post_init__(self) -> None:
        windows = tuple(self.windows)
        if not windows or not all(isinstance(item, DhbDescriptorWindow) for item in windows):
            raise PrimitiveContractError("multiscale descriptor requires DHB windows")
        frame_id = str(self.frame_id).strip()
        if not frame_id:
            raise PrimitiveContractError("descriptor frame_id must be non-empty")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "frame_id", frame_id)

    @property
    def scales(self) -> tuple[int, ...]:
        return tuple(sorted({window.scale_samples for window in self.windows}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "method": self.method,
            "windows": [window.to_dict() for window in self.windows],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MultiscaleDhbDescriptor:
        return cls(
            windows=tuple(
                DhbDescriptorWindow.from_dict(item) for item in payload["windows"]
            ),
            frame_id=str(payload["frame_id"]),
            method=str(payload.get("method", "double_reflection_position")),
        )


@dataclass(frozen=True)
class SparseMotionCompletionRequest:
    """Sparse relational completion query shared by interchangeable sources."""

    capsule_id: str
    primitive_request: PrimitiveRequest
    observed_relation_ids: tuple[str, ...]
    completion_frame_id: str

    def __post_init__(self) -> None:
        capsule_id = str(self.capsule_id).strip()
        frame = str(self.completion_frame_id).strip()
        relation_ids = tuple(str(value).strip() for value in self.observed_relation_ids)
        if not capsule_id or not frame:
            raise PrimitiveContractError("capsule_id and completion_frame_id must be non-empty")
        if not isinstance(self.primitive_request, PrimitiveRequest):
            raise PrimitiveContractError("primitive_request must be PrimitiveRequest")
        if any(not value for value in relation_ids) or len(set(relation_ids)) != len(relation_ids):
            raise PrimitiveContractError("observed_relation_ids must be unique and non-empty")
        declared = {anchor.relation_id for anchor in self.primitive_request.relational_anchors}
        if set(relation_ids) != declared:
            raise PrimitiveContractError(
                "observed_relation_ids must exactly attribute the request's relational anchors"
            )
        object.__setattr__(self, "capsule_id", capsule_id)
        object.__setattr__(self, "observed_relation_ids", relation_ids)
        object.__setattr__(self, "completion_frame_id", frame)


class SparseMotionCompletionSource(Protocol):
    """Proposal-only interface for completing a path from sparse relations."""

    @property
    def source_id(self) -> str:
        """Stable source identifier used for attribution."""

    def complete(
        self, request: SparseMotionCompletionRequest
    ) -> tuple[PrimitiveCandidate, ...]:
        """Return unapproved candidates; downstream gates retain authority."""


@dataclass(frozen=True)
class ObjectCentricSkillCapsule:
    """Raw and derived motion evidence for reusable object-centric behavior."""

    capsule_id: str
    semantic_label: str
    tcp_world: MotionPath
    object_world: MotionPath
    tcp_in_object: MotionPath
    descriptor: MultiscaleDhbDescriptor
    relations: tuple[CapsuleRelation, ...]
    end_effector_actions: tuple[ScheduledEndEffectorAction, ...]
    evidence: PrimitiveEvidence
    provenance: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CAPSULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        capsule_id = str(self.capsule_id).strip()
        label = str(self.semantic_label).strip()
        if not capsule_id or not label:
            raise PrimitiveContractError("capsule_id and semantic_label must be non-empty")
        if self.schema_version != CAPSULE_SCHEMA_VERSION:
            raise PrimitiveContractError(
                f"unsupported skill capsule schema {self.schema_version!r}"
            )
        paths = (self.tcp_world, self.object_world, self.tcp_in_object)
        if not all(isinstance(path, MotionPath) for path in paths):
            raise PrimitiveContractError("capsule trajectories must be MotionPath values")
        if len({path.samples for path in paths}) != 1:
            raise PrimitiveContractError("capsule trajectories must have the same sample count")
        if not all(np.allclose(path.timestamps_s, self.tcp_world.timestamps_s) for path in paths):
            raise PrimitiveContractError("capsule trajectories must share timestamps")
        if not isinstance(self.descriptor, MultiscaleDhbDescriptor):
            raise PrimitiveContractError("descriptor must be MultiscaleDhbDescriptor")
        relations = tuple(self.relations)
        phases = [relation.phase for relation in relations]
        if not all(isinstance(item, CapsuleRelation) for item in relations):
            raise PrimitiveContractError("relations must contain CapsuleRelation values")
        if phases != sorted(phases) or len(phases) != len(set(phases)):
            raise PrimitiveContractError("capsule relation phases must be unique and increasing")
        actions = tuple(self.end_effector_actions)
        if not all(isinstance(item, ScheduledEndEffectorAction) for item in actions):
            raise PrimitiveContractError(
                "end_effector_actions must contain ScheduledEndEffectorAction values"
            )
        if not isinstance(self.evidence, PrimitiveEvidence):
            raise PrimitiveContractError("evidence must be PrimitiveEvidence")
        provenance = _json_mapping(self.provenance, "capsule provenance")
        missing = [key for key in _REQUIRED_PROVENANCE if not str(provenance.get(key, "")).strip()]
        if missing:
            raise PrimitiveContractError(
                "capsule provenance is missing required fields: " + ", ".join(missing)
            )
        object.__setattr__(self, "capsule_id", capsule_id)
        object.__setattr__(self, "semantic_label", label)
        object.__setattr__(self, "relations", relations)
        object.__setattr__(self, "end_effector_actions", actions)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "capsule metadata"))

    @classmethod
    def capture(
        cls,
        *,
        capsule_id: str,
        semantic_label: str,
        tcp_world: MotionPath,
        object_world: MotionPath,
        relation_events: Mapping[str, float],
        end_effector_actions: tuple[ScheduledEndEffectorAction, ...],
        evidence: PrimitiveEvidence,
        provenance: Mapping[str, Any],
        descriptor_scales: tuple[int, ...] = (8, 16, 32),
        descriptor_stride_fraction: float = 0.5,
        metadata: Mapping[str, Any] | None = None,
    ) -> ObjectCentricSkillCapsule:
        relative = object_relative_motion(tcp_world, object_world)
        relations = []
        for relation_id, phase_value in sorted(
            relation_events.items(), key=lambda item: float(item[1])
        ):
            phase = float(phase_value)
            index = int(round(phase * (relative.samples - 1)))
            relations.append(
                CapsuleRelation(
                    phase,
                    relation_id,
                    relative.positions[index],
                    relative.quaternions_wxyz[index],
                    metadata={"sample_index": index, "frame": relative.frame_id},
                )
            )
        return cls(
            capsule_id=capsule_id,
            semantic_label=semantic_label,
            tcp_world=tcp_world,
            object_world=object_world,
            tcp_in_object=relative,
            # The reusable geometry is the object's task-space path expressed
            # in its initial pose. TCP-in-object is retained separately as the
            # grasp/tool relation; encoding that often-nearly-constant path
            # would erase the carry geometry that retrieval and completion need.
            descriptor=build_multiscale_dhb_descriptor(
                motion_in_initial_frame(object_world),
                scales=descriptor_scales,
                stride_fraction=descriptor_stride_fraction,
            ),
            relations=tuple(relations),
            end_effector_actions=end_effector_actions,
            evidence=evidence,
            provenance=provenance,
            metadata={} if metadata is None else metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capsule_id": self.capsule_id,
            "semantic_label": self.semantic_label,
            "tcp_world": self.tcp_world.to_dict(),
            "object_world": self.object_world.to_dict(),
            "tcp_in_object": self.tcp_in_object.to_dict(),
            "descriptor": self.descriptor.to_dict(),
            "relations": [relation.to_dict() for relation in self.relations],
            "end_effector_actions": [action.to_dict() for action in self.end_effector_actions],
            "evidence": self.evidence.to_dict(),
            "provenance": dict(self.provenance),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ObjectCentricSkillCapsule:
        return cls(
            schema_version=int(payload["schema_version"]),
            capsule_id=str(payload["capsule_id"]),
            semantic_label=str(payload["semantic_label"]),
            tcp_world=MotionPath.from_dict(payload["tcp_world"]),
            object_world=MotionPath.from_dict(payload["object_world"]),
            tcp_in_object=MotionPath.from_dict(payload["tcp_in_object"]),
            descriptor=MultiscaleDhbDescriptor.from_dict(payload["descriptor"]),
            relations=tuple(CapsuleRelation.from_dict(item) for item in payload["relations"]),
            end_effector_actions=tuple(
                ScheduledEndEffectorAction.from_dict(item)
                for item in payload.get("end_effector_actions", ())
            ),
            evidence=PrimitiveEvidence.from_dict(payload["evidence"]),
            provenance=payload["provenance"],
            metadata=payload.get("metadata", {}),
        )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> ObjectCentricSkillCapsule:
        return cls.from_dict(json.loads(Path(path).read_text()))


def object_relative_motion(tcp_world: MotionPath, object_world: MotionPath) -> MotionPath:
    """Express a synchronized world-frame TCP path in the moving object frame."""

    if tcp_world.samples != object_world.samples or not np.allclose(
        tcp_world.timestamps_s, object_world.timestamps_s
    ):
        raise PrimitiveContractError("TCP and object trajectories must be synchronized")
    positions = np.empty_like(tcp_world.positions)
    quaternions = np.empty_like(tcp_world.quaternions_wxyz)
    for index, (tcp_position, tcp_quaternion, object_position, object_quaternion) in enumerate(
        zip(
            tcp_world.positions,
            tcp_world.quaternions_wxyz,
            object_world.positions,
            object_world.quaternions_wxyz,
            strict=True,
        )
    ):
        object_inverse = geom.quat_inv(object_quaternion)
        positions[index] = geom.quat_to_rot(object_inverse) @ (tcp_position - object_position)
        quaternions[index] = geom.quat_multiply(object_inverse, tcp_quaternion)
    return MotionPath(
        positions,
        quaternions,
        tcp_world.timestamps_s,
        frame_id=f"{tcp_world.frame_id}_tcp_in_{object_world.frame_id}_object",
    )


def motion_in_initial_frame(motion: MotionPath) -> MotionPath:
    """Canonicalize a path to its initial pose before descriptor encoding."""

    initial_position = motion.positions[0]
    initial_inverse = geom.quat_inv(motion.quaternions_wxyz[0])
    initial_rotation = geom.quat_to_rot(initial_inverse)
    positions = (motion.positions - initial_position) @ initial_rotation.T
    quaternions = np.stack(
        [
            geom.quat_multiply(initial_inverse, quaternion)
            for quaternion in motion.quaternions_wxyz
        ]
    )
    return MotionPath(
        positions,
        quaternions,
        motion.timestamps_s,
        frame_id=f"{motion.frame_id}_relative_to_initial_object_pose",
    )


def build_multiscale_dhb_descriptor(
    motion: MotionPath,
    *,
    scales: tuple[int, ...] = (8, 16, 32),
    stride_fraction: float = 0.5,
) -> MultiscaleDhbDescriptor:
    """Encode overlapping local windows plus a mandatory whole-path level."""

    if not 0.0 < float(stride_fraction) <= 1.0:
        raise PrimitiveContractError("descriptor stride_fraction must lie in (0, 1]")
    requested = sorted({min(int(scale), motion.samples) for scale in scales} | {motion.samples})
    if not requested or requested[0] < 3:
        raise PrimitiveContractError("descriptor scales must contain at least three samples")
    windows = []
    for scale in requested:
        stride = max(1, int(round(scale * float(stride_fraction))))
        starts = list(range(0, motion.samples - scale + 1, stride))
        final_start = motion.samples - scale
        if starts[-1] != final_start:
            starts.append(final_start)
        for start in starts:
            stop = start + scale
            encoded = encode_dhb_dr(
                motion.positions[start:stop],
                motion.quaternions_wxyz[start:stop],
                method=EncodingMethod.POSITION,
                use_default_initial_frames=True,
                dhb_method=DHBMethod.DOUBLE_REFLECTION,
                robust_mode=True,
            )
            invariants = np.concatenate(
                [
                    encoded["linear_motion_invariants"],
                    encoded["angular_motion_invariants"],
                ],
                axis=1,
            )
            windows.append(
                DhbDescriptorWindow(
                    scale,
                    start / max(motion.samples - 1, 1),
                    (stop - 1) / max(motion.samples - 1, 1),
                    invariants,
                )
            )
    return MultiscaleDhbDescriptor(tuple(windows), frame_id=motion.frame_id)


def _json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    try:
        serialized = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PrimitiveContractError(f"{field_name} must be JSON serializable") from exc
    restored = json.loads(serialized)
    if not isinstance(restored, dict):
        raise PrimitiveContractError(f"{field_name} must be a mapping")
    return restored
