"""Adapters from existing DHB/OCP and learned backends to primitive candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from dhb_xr.primitives.contracts import (
    MotionPath,
    PrimitiveCandidate,
    PrimitiveContractError,
    PrimitiveEvidence,
    PrimitiveFamily,
)
from dhb_xr.primitives.sources import PrimitiveRequest

BackendResult = Mapping[str, Any]
BackendCallable = Callable[[PrimitiveRequest], BackendResult | Sequence[BackendResult]]


def motion_path_from_backend_result(
    result: BackendResult,
    *,
    duration_s: float,
    frame_id: str,
) -> MotionPath:
    """Normalize common backend array keys into the strict motion contract."""

    if "positions" not in result:
        raise PrimitiveContractError("backend result is missing positions")
    quaternion_key = (
        "quaternions_wxyz" if "quaternions_wxyz" in result else "quaternions"
    )
    if quaternion_key not in result:
        raise PrimitiveContractError("backend result is missing quaternions")
    positions = np.asarray(result["positions"], dtype=np.float64)
    quaternions = np.asarray(result[quaternion_key], dtype=np.float64)
    timestamps = result.get("timestamps_s")
    if timestamps is None:
        timestamps = np.linspace(0.0, float(duration_s), len(positions))
    return MotionPath(positions, quaternions, np.asarray(timestamps), frame_id=frame_id)


def candidate_from_backend_result(
    request: PrimitiveRequest,
    result: BackendResult,
    *,
    family: PrimitiveFamily,
    evidence: PrimitiveEvidence,
    source_id: str,
    rank: Optional[int] = None,
) -> PrimitiveCandidate:
    """Bind backend geometry to the request's interaction and authority context."""

    primitive_id = request.primitive_id if rank is None else f"{request.primitive_id}:rank-{rank}"
    backend_metadata = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "positions",
            "quaternions",
            "quaternions_wxyz",
            "timestamps_s",
        }
        and _is_json_value(value)
    }
    metadata = dict(request.metadata)
    metadata.update(
        {
            "candidate_source": source_id,
            "backend": backend_metadata,
        }
    )
    return PrimitiveCandidate(
        primitive_id=primitive_id,
        family=family,
        evidence=evidence,
        motion=motion_path_from_backend_result(
            result,
            duration_s=request.duration_s,
            frame_id=request.frame_id,
        ),
        end_effector_actions=request.end_effector_actions,
        contact_mode=request.contact_mode,
        reference_frames=request.reference_frames,
        start_guard=request.start_guard,
        completion_guard=request.completion_guard,
        postcondition=request.postcondition,
        timeout_s=request.timeout_s,
        observation_epoch=request.observation_epoch,
        metadata=metadata,
    )


@dataclass(frozen=True)
class BackendPrimitiveSource:
    """Wrap a learned or optimization callable behind the common source API."""

    source_id: str
    family: PrimitiveFamily
    evidence: PrimitiveEvidence
    backend: BackendCallable = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        source_id = str(self.source_id).strip()
        if not source_id:
            raise PrimitiveContractError("source_id must be non-empty")
        family = (
            self.family
            if isinstance(self.family, PrimitiveFamily)
            else PrimitiveFamily(self.family)
        )
        if family not in {
            PrimitiveFamily.CONSTRAINED_OCP,
            PrimitiveFamily.LEARNED_FLOW,
            PrimitiveFamily.LEARNED_DIFFUSION,
            PrimitiveFamily.CUSTOM,
        }:
            raise PrimitiveContractError(
                "BackendPrimitiveSource must declare OCP, learned, or custom provenance"
            )
        if not callable(self.backend):
            raise PrimitiveContractError("backend must be callable")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "family", family)

    def propose(self, request: PrimitiveRequest) -> tuple[PrimitiveCandidate, ...]:
        result = self.backend(request)
        rows = [result] if isinstance(result, Mapping) else list(result)
        if not rows:
            return ()
        if not all(isinstance(row, Mapping) for row in rows):
            raise PrimitiveContractError("backend must return a mapping or sequence of mappings")
        multiple = len(rows) > 1
        return tuple(
            candidate_from_backend_result(
                request,
                row,
                family=self.family,
                evidence=self.evidence,
                source_id=self.source_id,
                rank=index if multiple else None,
            )
            for index, row in enumerate(rows)
        )


@dataclass(frozen=True)
class DHBRetargetSource:
    """Adapt one stored full-SE(3) reference through DHB-XR's public optimizer."""

    reference_positions: np.ndarray
    reference_quaternions_wxyz: np.ndarray
    evidence: PrimitiveEvidence
    backend: str = "auto"
    allow_fallback: bool = False
    generator: Optional[Callable[..., BackendResult]] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        positions = np.asarray(self.reference_positions, dtype=np.float64)
        quaternions = np.asarray(self.reference_quaternions_wxyz, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 2:
            raise PrimitiveContractError(
                "reference_positions must have shape (N, 3) with N >= 2"
            )
        if quaternions.shape != (len(positions), 4):
            raise PrimitiveContractError(
                "reference_quaternions_wxyz must have shape (N, 4)"
            )
        validated = MotionPath(
            positions,
            quaternions,
            np.linspace(0.0, 1.0, len(positions)),
        )
        backend = str(self.backend).strip()
        if not backend:
            raise PrimitiveContractError("backend must be non-empty")
        if not isinstance(self.evidence, PrimitiveEvidence):
            raise PrimitiveContractError("evidence must be PrimitiveEvidence")
        if self.generator is not None and not callable(self.generator):
            raise PrimitiveContractError("generator must be callable")
        object.__setattr__(self, "reference_positions", validated.positions)
        object.__setattr__(
            self, "reference_quaternions_wxyz", validated.quaternions_wxyz
        )
        object.__setattr__(self, "backend", backend)

    @property
    def source_id(self) -> str:
        return f"dhb-retarget:{self.backend}"

    def propose(self, request: PrimitiveRequest) -> tuple[PrimitiveCandidate, ...]:
        generator = self.generator
        if generator is None:
            from dhb_xr.optimization import generate_trajectory

            generator = generate_trajectory
        result = generator(
            pos_data=np.asarray(self.reference_positions, dtype=np.float64),
            quat_data=np.asarray(self.reference_quaternions_wxyz, dtype=np.float64),
            pose_target_init={
                "position": np.asarray(request.start_position, dtype=np.float64),
                "quaternion": np.asarray(request.start_quaternion_wxyz, dtype=np.float64),
            },
            pose_target_final={
                "position": np.asarray(request.goal_position, dtype=np.float64),
                "quaternion": np.asarray(request.goal_quaternion_wxyz, dtype=np.float64),
            },
            traj_length=request.samples,
            backend=self.backend,
            allow_fallback=self.allow_fallback,
        )
        return (
            candidate_from_backend_result(
                request,
                result,
                family=PrimitiveFamily.DHB_MEMORY,
                evidence=self.evidence,
                source_id=self.source_id,
            ),
        )


def _is_json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int, float, np.generic)):
        return bool(np.all(np.isfinite(value))) if isinstance(value, (float, np.floating)) else True
    if isinstance(value, np.ndarray):
        return bool(np.all(np.isfinite(value)))
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    return False
