"""Canonical trace ingestion for evidence-preserving primitive mining pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

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
from dhb_xr.primitives.segmentation import (
    propose_primitive_boundaries,
    split_motion_at_indices,
)


@dataclass(frozen=True)
class DemonstrationTrace:
    """A normalized raw demonstration trace with declared provenance."""

    motion: MotionPath
    gripper_commands: np.ndarray
    source_uri: str
    dataset: str
    action_convention: str
    license_id: str
    contacts: Optional[np.ndarray] = None
    object_positions: Optional[np.ndarray] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.motion, MotionPath):
            raise PrimitiveContractError("motion must be MotionPath")
        gripper = np.array(self.gripper_commands, dtype=np.float64, copy=True)
        if gripper.shape != (self.motion.samples,) or not np.all(np.isfinite(gripper)):
            raise PrimitiveContractError(
                "gripper_commands must be finite with one value per motion sample"
            )
        contacts = None
        if self.contacts is not None:
            contacts = np.array(self.contacts, dtype=bool, copy=True)
            if contacts.shape != (self.motion.samples,):
                raise PrimitiveContractError(
                    "contacts must contain one value per motion sample"
                )
            contacts.setflags(write=False)
        object_positions = None
        if self.object_positions is not None:
            object_positions = np.array(
                self.object_positions, dtype=np.float64, copy=True
            )
            if object_positions.shape != (self.motion.samples, 3) or not np.all(
                np.isfinite(object_positions)
            ):
                raise PrimitiveContractError(
                    "object_positions must be finite with shape (N, 3)"
                )
            object_positions.setflags(write=False)
        fields = {
            "source_uri": self.source_uri,
            "dataset": self.dataset,
            "action_convention": self.action_convention,
            "license_id": self.license_id,
        }
        for name, value in fields.items():
            if not str(value).strip():
                raise PrimitiveContractError(f"{name} must be non-empty")
            object.__setattr__(self, name, str(value).strip())
        gripper.setflags(write=False)
        object.__setattr__(self, "gripper_commands", gripper)
        object.__setattr__(self, "contacts", contacts)
        object.__setattr__(self, "object_positions", object_positions)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True)
class MinedPrimitiveSegment:
    """A proposed segment; it is not executable until semantics are supplied."""

    start_index: int
    end_index: int
    motion: MotionPath
    gripper_commands: np.ndarray
    start_reasons: tuple[str, ...]
    end_reasons: tuple[str, ...]
    source_uri: str
    dataset: str
    action_convention: str
    license_id: str
    contacts: Optional[np.ndarray] = None
    object_positions: Optional[np.ndarray] = None
    metadata: Optional[Mapping[str, Any]] = None


def mine_demonstration_segments(
    trace: DemonstrationTrace,
    *,
    object_motion_threshold_m: float = 0.002,
    closed_threshold: float = 0.3,
    open_threshold: float = 0.7,
    min_dwell_samples: int = 2,
) -> tuple[MinedPrimitiveSegment, ...]:
    """Propose shared-endpoint segments from measured trace events."""

    if not isinstance(trace, DemonstrationTrace):
        raise PrimitiveContractError("trace must be DemonstrationTrace")
    proposals = propose_primitive_boundaries(
        trace.gripper_commands,
        contacts=trace.contacts,
        object_positions=trace.object_positions,
        object_motion_threshold_m=object_motion_threshold_m,
        closed_threshold=closed_threshold,
        open_threshold=open_threshold,
        min_dwell_samples=min_dwell_samples,
    )
    indices = tuple(proposal.index for proposal in proposals)
    reasons = {proposal.index: proposal.reasons for proposal in proposals}
    pieces = split_motion_at_indices(trace.motion, indices)
    segments = []
    for piece, start, end in zip(pieces, indices[:-1], indices[1:]):
        contacts = None if trace.contacts is None else trace.contacts[start : end + 1]
        object_positions = (
            None
            if trace.object_positions is None
            else trace.object_positions[start : end + 1]
        )
        segments.append(
            MinedPrimitiveSegment(
                start_index=start,
                end_index=end,
                motion=piece,
                gripper_commands=trace.gripper_commands[start : end + 1],
                start_reasons=reasons[start],
                end_reasons=reasons[end],
                source_uri=trace.source_uri,
                dataset=trace.dataset,
                action_convention=trace.action_convention,
                license_id=trace.license_id,
                contacts=contacts,
                object_positions=object_positions,
                metadata=trace.metadata,
            )
        )
    return tuple(segments)


def segment_to_candidate(
    segment: MinedPrimitiveSegment,
    *,
    primitive_id: str,
    semantic_label: str,
    evidence: PrimitiveEvidence,
    family: PrimitiveFamily = PrimitiveFamily.DHB_MEMORY,
    end_effector_actions: tuple[ScheduledEndEffectorAction, ...] = (),
    contact_mode: ContactMode = ContactMode.FREE_SPACE,
    reference_frames: Optional[Mapping[str, str]] = None,
    start_guard: Optional[GuardExpression] = None,
    completion_guard: Optional[GuardExpression] = None,
    postcondition: Optional[GuardExpression] = None,
    observation_epoch: int = 0,
) -> PrimitiveCandidate:
    """Promote a proposed segment only after explicit semantics are supplied."""

    label = str(semantic_label).strip()
    if not label:
        raise PrimitiveContractError(
            "semantic_label is required; ingestion does not infer executable skill meaning"
        )
    metadata = dict(segment.metadata or {})
    metadata.update(
        {
            "semantic_label": label,
            "source_uri": segment.source_uri,
            "dataset": segment.dataset,
            "action_convention": segment.action_convention,
            "license_id": segment.license_id,
            "source_indices": [segment.start_index, segment.end_index],
            "start_boundary_reasons": list(segment.start_reasons),
            "end_boundary_reasons": list(segment.end_reasons),
            "ingestion_status": "boundary_proposal_with_explicit_semantics",
        }
    )
    return PrimitiveCandidate(
        primitive_id=primitive_id,
        family=family,
        evidence=evidence,
        motion=segment.motion,
        end_effector_actions=end_effector_actions,
        contact_mode=contact_mode,
        reference_frames={} if reference_frames is None else reference_frames,
        start_guard=start_guard,
        completion_guard=completion_guard,
        postcondition=postcondition,
        observation_epoch=observation_epoch,
        metadata=metadata,
    )
