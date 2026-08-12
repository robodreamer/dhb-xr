"""Evidence-preserving memory for typed manipulation primitives."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.database.similarity import invariant_distance
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.primitives.contracts import (
    EvidenceLevel,
    PrimitiveCandidate,
    PrimitiveContractError,
)

MEMORY_SCHEMA_VERSION = 1


class VerificationOutcome(str, Enum):
    """Result of one declared replay, simulation, or real verification."""

    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class VerificationRecord:
    """Immutable result row used to justify or reject evidence promotion."""

    outcome: VerificationOutcome
    level: EvidenceLevel
    protocol_id: str
    artifact_hash: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        outcome = (
            self.outcome
            if isinstance(self.outcome, VerificationOutcome)
            else VerificationOutcome(self.outcome)
        )
        level = self.level if isinstance(self.level, EvidenceLevel) else EvidenceLevel(self.level)
        protocol_id = str(self.protocol_id).strip()
        if not protocol_id:
            raise PrimitiveContractError("verification protocol_id must be non-empty")
        metadata = _json_mapping(self.metadata, "verification metadata")
        object.__setattr__(self, "outcome", outcome)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "level": self.level.value,
            "protocol_id": self.protocol_id,
            "artifact_hash": self.artifact_hash,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationRecord:
        return cls(
            outcome=VerificationOutcome(payload["outcome"]),
            level=EvidenceLevel(payload["level"]),
            protocol_id=str(payload["protocol_id"]),
            artifact_hash=payload.get("artifact_hash"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True)
class PrimitiveMemoryRecord:
    """One candidate, its DHB geometry, and its complete verification history."""

    entry_id: str
    candidate: PrimitiveCandidate
    invariants: Optional[np.ndarray]
    interaction_signature: tuple[str, ...]
    verification_history: tuple[VerificationRecord, ...] = ()
    source_uri: Optional[str] = None
    source_action_convention: Optional[str] = None

    def __post_init__(self) -> None:
        entry_id = str(self.entry_id).strip()
        if not entry_id:
            raise PrimitiveContractError("memory entry_id must be non-empty")
        if not isinstance(self.candidate, PrimitiveCandidate):
            raise PrimitiveContractError("memory candidate must be PrimitiveCandidate")
        invariants = self.invariants
        if invariants is not None:
            invariants = np.array(invariants, dtype=np.float64, copy=True)
            if invariants.ndim != 2 or invariants.shape[1] != 8:
                raise PrimitiveContractError("DHB-DR memory invariants must have shape (N, 8)")
            if not np.all(np.isfinite(invariants)):
                raise PrimitiveContractError("memory invariants must be finite")
            invariants.setflags(write=False)
        signature = tuple(str(item) for item in self.interaction_signature)
        history = tuple(self.verification_history)
        if not all(isinstance(item, VerificationRecord) for item in history):
            raise PrimitiveContractError("verification_history contains an invalid record")
        object.__setattr__(self, "entry_id", entry_id)
        object.__setattr__(self, "invariants", invariants)
        object.__setattr__(self, "interaction_signature", signature)
        object.__setattr__(self, "verification_history", history)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "candidate": self.candidate.to_dict(),
            "invariants": None if self.invariants is None else self.invariants.tolist(),
            "interaction_signature": list(self.interaction_signature),
            "verification_history": [record.to_dict() for record in self.verification_history],
            "source_uri": self.source_uri,
            "source_action_convention": self.source_action_convention,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PrimitiveMemoryRecord:
        invariants = payload.get("invariants")
        return cls(
            entry_id=str(payload["entry_id"]),
            candidate=PrimitiveCandidate.from_dict(payload["candidate"]),
            invariants=None if invariants is None else np.asarray(invariants, dtype=np.float64),
            interaction_signature=tuple(payload.get("interaction_signature", ())),
            verification_history=tuple(
                VerificationRecord.from_dict(record)
                for record in payload.get("verification_history", ())
            ),
            source_uri=payload.get("source_uri"),
            source_action_convention=payload.get("source_action_convention"),
        )


@dataclass(frozen=True)
class PrimitiveMemoryMatch:
    """A memory record and its attributed composite retrieval distance."""

    record: PrimitiveMemoryRecord
    geometric_distance: float
    semantic_penalty: float

    @property
    def distance(self) -> float:
        return self.geometric_distance + self.semantic_penalty


class PrimitiveMemory:
    """Small canonical memory with semantic filtering before geometric ranking."""

    def __init__(self) -> None:
        self._records: dict[str, PrimitiveMemoryRecord] = {}

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> tuple[PrimitiveMemoryRecord, ...]:
        return tuple(self._records.values())

    def add(
        self,
        candidate: PrimitiveCandidate,
        *,
        entry_id: Optional[str] = None,
        source_uri: Optional[str] = None,
        source_action_convention: Optional[str] = None,
    ) -> PrimitiveMemoryRecord:
        if not isinstance(candidate, PrimitiveCandidate):
            raise PrimitiveContractError("candidate must be PrimitiveCandidate")
        resolved_id = entry_id or _candidate_fingerprint(candidate)
        if resolved_id in self._records:
            raise PrimitiveContractError(f"memory entry {resolved_id!r} already exists")
        record = PrimitiveMemoryRecord(
            entry_id=resolved_id,
            candidate=candidate,
            invariants=_candidate_invariants(candidate),
            interaction_signature=_interaction_signature(candidate),
            source_uri=source_uri,
            source_action_convention=source_action_convention,
        )
        self._records[record.entry_id] = record
        return record

    def get(self, entry_id: str) -> PrimitiveMemoryRecord:
        return self._records[entry_id]

    def record_verification(
        self, entry_id: str, verification: VerificationRecord
    ) -> PrimitiveMemoryRecord:
        if not isinstance(verification, VerificationRecord):
            raise PrimitiveContractError("verification must be VerificationRecord")
        current = self._records[entry_id]
        candidate = current.candidate
        if (
            verification.outcome == VerificationOutcome.PASSED
            and verification.level.rank > candidate.evidence.level.rank
        ):
            evidence = replace(
                candidate.evidence,
                level=verification.level,
                artifact_hash=verification.artifact_hash or candidate.evidence.artifact_hash,
            )
            candidate = replace(candidate, evidence=evidence)
        updated = replace(
            current,
            candidate=candidate,
            verification_history=current.verification_history + (verification,),
        )
        self._records[entry_id] = updated
        return updated

    def retrieve(
        self,
        query: PrimitiveCandidate,
        *,
        k: int = 5,
        minimum_evidence: EvidenceLevel = EvidenceLevel.PLANNED,
        require_interaction_match: bool = True,
        semantic_mismatch_penalty: float = 1_000.0,
        compatible_robot_controller_revision: Optional[str] = None,
    ) -> tuple[PrimitiveMemoryMatch, ...]:
        if int(k) < 1:
            raise PrimitiveContractError("k must be at least one")
        minimum = (
            minimum_evidence
            if isinstance(minimum_evidence, EvidenceLevel)
            else EvidenceLevel(minimum_evidence)
        )
        query_invariants = _candidate_invariants(query)
        query_signature = _interaction_signature(query)
        matches = []
        for record in self._records.values():
            if record.candidate.evidence.level.rank < minimum.rank:
                continue
            compatible = record.candidate.evidence.compatible_robot_controller_revisions
            if (
                compatible_robot_controller_revision is not None
                and compatible
                and compatible_robot_controller_revision not in compatible
            ):
                continue
            signature_matches = record.interaction_signature == query_signature
            if require_interaction_match and not signature_matches:
                continue
            if query_invariants is None or record.invariants is None:
                if query_invariants is not None or record.invariants is not None:
                    continue
                geometric_distance = 0.0
            else:
                geometric_distance = invariant_distance(query_invariants, record.invariants)
            penalty = 0.0 if signature_matches else float(semantic_mismatch_penalty)
            matches.append(PrimitiveMemoryMatch(record, geometric_distance, penalty))
        matches.sort(key=lambda match: (match.distance, match.record.entry_id))
        return tuple(matches[: int(k)])

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "records": [record.to_dict() for record in self.records],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> PrimitiveMemory:
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            raise PrimitiveContractError(
                f"unsupported primitive memory schema {payload.get('schema_version')!r}"
            )
        memory = cls()
        for item in payload.get("records", ()):
            record = PrimitiveMemoryRecord.from_dict(item)
            if record.entry_id in memory._records:
                raise PrimitiveContractError(f"duplicate memory entry {record.entry_id!r}")
            memory._records[record.entry_id] = record
        return memory


def _candidate_invariants(candidate: PrimitiveCandidate) -> Optional[np.ndarray]:
    if candidate.motion is None:
        return None
    encoded = encode_dhb_dr(
        candidate.motion.positions,
        candidate.motion.quaternions_wxyz,
        method=EncodingMethod.POSITION.value,
        use_default_initial_frames=True,
        dhb_method=DHBMethod.DOUBLE_REFLECTION,
        robust_mode=True,
    )
    invariants: np.ndarray = np.asarray(
        np.concatenate(
            [
                encoded["linear_motion_invariants"],
                encoded["angular_motion_invariants"],
            ],
            axis=1,
        ),
        dtype=np.float64,
    )
    return invariants


def _interaction_signature(candidate: PrimitiveCandidate) -> tuple[str, ...]:
    return (candidate.contact_mode.value,) + tuple(
        f"{action.timing.value}:{action.command.command.value}"
        for action in candidate.end_effector_actions
    )


def _candidate_fingerprint(candidate: PrimitiveCandidate) -> str:
    canonical = json.dumps(candidate.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_mapping(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    try:
        serialized = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PrimitiveContractError(f"{field_name} must be JSON serializable") from exc
    restored = json.loads(serialized)
    if not isinstance(restored, dict):
        raise PrimitiveContractError(f"{field_name} must be a mapping")
    return restored
