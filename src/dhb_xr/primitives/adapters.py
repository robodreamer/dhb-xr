"""Adapters from existing DHB/OCP and learned backends to primitive candidates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from dhb_xr.primitives.analytic import minimum_jerk_blend
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
    quaternion_key = "quaternions_wxyz" if "quaternions_wxyz" in result else "quaternions"
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

    if "success" in result and not bool(result["success"]):
        raise PrimitiveContractError(
            f"backend source {source_id!r} reported success=False; refusing candidate promotion"
        )
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
class EndpointCorrectedReferenceSource:
    """Retarget a raw SE(3) reference with smooth endpoint residuals.

    This is an intentionally strong, dependency-light non-DHB reuse control.
    It preserves the sampled Cartesian reference while distributing independent
    start and goal residuals with a minimum-jerk phase law.  It does not solve
    constraints or certify collision safety.
    """

    reference_positions: np.ndarray
    reference_quaternions_wxyz: np.ndarray
    evidence: PrimitiveEvidence

    def __post_init__(self) -> None:
        positions = np.asarray(self.reference_positions, dtype=np.float64)
        quaternions = np.asarray(self.reference_quaternions_wxyz, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 2:
            raise PrimitiveContractError("reference_positions must have shape (N, 3) with N >= 2")
        if quaternions.shape != (len(positions), 4):
            raise PrimitiveContractError("reference_quaternions_wxyz must have shape (N, 4)")
        validated = MotionPath(
            positions,
            quaternions,
            np.linspace(0.0, 1.0, len(positions)),
        )
        if not isinstance(self.evidence, PrimitiveEvidence):
            raise PrimitiveContractError("evidence must be PrimitiveEvidence")
        object.__setattr__(self, "reference_positions", validated.positions)
        object.__setattr__(self, "reference_quaternions_wxyz", validated.quaternions_wxyz)

    @property
    def source_id(self) -> str:
        return "reference:endpoint_corrected_se3"

    def propose(self, request: PrimitiveRequest) -> tuple[PrimitiveCandidate, ...]:
        positions, quaternions = _resample_reference(
            self.reference_positions,
            self.reference_quaternions_wxyz,
            request.samples,
        )
        phase = minimum_jerk_blend(np.linspace(0.0, 1.0, request.samples))
        start_delta = request.start_position - positions[0]
        goal_delta = request.goal_position - positions[-1]
        positions = positions + (1.0 - phase[:, None]) * start_delta + phase[:, None] * goal_delta

        start_residual = _quaternion_multiply(
            request.start_quaternion_wxyz,
            _quaternion_inverse(quaternions[0]),
        )
        goal_residual = _quaternion_multiply(
            request.goal_quaternion_wxyz,
            _quaternion_inverse(quaternions[-1]),
        )
        corrected = np.empty_like(quaternions)
        for index, value in enumerate(phase):
            residual = _quaternion_slerp(start_residual, goal_residual, float(value))
            corrected[index] = _quaternion_multiply(residual, quaternions[index])
        positions[0] = request.start_position
        positions[-1] = request.goal_position
        corrected[0] = request.start_quaternion_wxyz
        corrected[-1] = request.goal_quaternion_wxyz

        candidate = PrimitiveCandidate(
            primitive_id=request.primitive_id,
            family=PrimitiveFamily.CUSTOM,
            evidence=self.evidence,
            motion=MotionPath(
                positions,
                corrected,
                np.linspace(0.0, request.duration_s, request.samples),
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
            metadata={
                **request.metadata,
                "candidate_source": self.source_id,
                "retargeting": "raw_se3_minimum_jerk_endpoint_residual",
            },
        )
        return (candidate,)


@dataclass(frozen=True)
class RelationalAnchorReferenceSource:
    """Strong raw-SE(3) control using smooth piecewise anchor residuals."""

    reference_positions: np.ndarray
    reference_quaternions_wxyz: np.ndarray
    evidence: PrimitiveEvidence

    def __post_init__(self) -> None:
        endpoint_source = EndpointCorrectedReferenceSource(
            self.reference_positions,
            self.reference_quaternions_wxyz,
            self.evidence,
        )
        object.__setattr__(self, "reference_positions", endpoint_source.reference_positions)
        object.__setattr__(
            self,
            "reference_quaternions_wxyz",
            endpoint_source.reference_quaternions_wxyz,
        )

    @property
    def source_id(self) -> str:
        return "reference:relational_anchor_se3"

    def propose(self, request: PrimitiveRequest) -> tuple[PrimitiveCandidate, ...]:
        endpoint_source = EndpointCorrectedReferenceSource(
            self.reference_positions,
            self.reference_quaternions_wxyz,
            self.evidence,
        )
        (base,) = endpoint_source.propose(request)
        from dhb_xr.optimization.relational_fatrop import (
            _anchored_position_guess,
            _anchored_quaternion_reference,
        )

        positions = _anchored_position_guess(
            base.motion.positions,
            request.relational_anchors,
        )
        quaternions = _anchored_quaternion_reference(
            base.motion.quaternions_wxyz,
            request.relational_anchors,
        )
        positions[0] = request.start_position
        positions[-1] = request.goal_position
        quaternions[0] = request.start_quaternion_wxyz
        quaternions[-1] = request.goal_quaternion_wxyz
        return (
            PrimitiveCandidate(
                primitive_id=request.primitive_id,
                family=PrimitiveFamily.CUSTOM,
                evidence=self.evidence,
                motion=MotionPath(
                    positions,
                    quaternions,
                    np.linspace(0.0, request.duration_s, request.samples),
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
                metadata={
                    **request.metadata,
                    "candidate_source": self.source_id,
                    "retargeting": "raw_se3_piecewise_minimum_jerk_anchor_residual",
                },
            ),
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
            raise PrimitiveContractError("reference_positions must have shape (N, 3) with N >= 2")
        if quaternions.shape != (len(positions), 4):
            raise PrimitiveContractError("reference_quaternions_wxyz must have shape (N, 4)")
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
        object.__setattr__(self, "reference_quaternions_wxyz", validated.quaternions_wxyz)
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


class PersistentRelationalFatropSource:
    """Expose one persistent matched FATROP retargeter as a primitive source."""

    def __init__(self, retargeter: Any, evidence: PrimitiveEvidence) -> None:
        if not hasattr(retargeter, "generate") or not hasattr(retargeter, "source_id"):
            raise PrimitiveContractError("retargeter must provide source_id and generate(...)")
        if not isinstance(evidence, PrimitiveEvidence):
            raise PrimitiveContractError("evidence must be PrimitiveEvidence")
        self.retargeter = retargeter
        self.evidence = evidence

    @property
    def source_id(self) -> str:
        return str(self.retargeter.source_id)

    def propose(self, request: PrimitiveRequest) -> tuple[PrimitiveCandidate, ...]:
        if request.samples != self.retargeter.samples:
            raise PrimitiveContractError(
                "request samples must match the persistent FATROP graph horizon"
            )
        result = self.retargeter.generate(
            start_position=request.start_position,
            start_quaternion_wxyz=request.start_quaternion_wxyz,
            goal_position=request.goal_position,
            goal_quaternion_wxyz=request.goal_quaternion_wxyz,
            relational_anchors=request.relational_anchors,
        )
        family = (
            PrimitiveFamily.DHB_MEMORY
            if getattr(self.retargeter, "regularization", None) == "dhb"
            else PrimitiveFamily.CONSTRAINED_OCP
        )
        return (
            candidate_from_backend_result(
                request,
                result,
                family=family,
                evidence=self.evidence,
                source_id=self.source_id,
            ),
        )


def _resample_reference(
    positions: np.ndarray,
    quaternions: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    source_phase = np.linspace(0.0, 1.0, len(positions))
    target_phase = np.linspace(0.0, 1.0, samples)
    resampled_positions = np.column_stack(
        [np.interp(target_phase, source_phase, positions[:, axis]) for axis in range(3)]
    )
    resampled_quaternions = np.empty((samples, 4), dtype=np.float64)
    for index, value in enumerate(target_phase):
        upper = int(np.searchsorted(source_phase, value, side="right"))
        upper = min(max(upper, 1), len(source_phase) - 1)
        lower = upper - 1
        span = source_phase[upper] - source_phase[lower]
        local = 0.0 if span <= 0.0 else (value - source_phase[lower]) / span
        resampled_quaternions[index] = _quaternion_slerp(
            quaternions[lower], quaternions[upper], float(local)
        )
    return resampled_positions, resampled_quaternions


def _quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
        dtype=np.float64,
    ) / float(np.dot(quaternion, quaternion))


def _quaternion_multiply(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(first, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(second, dtype=np.float64)
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    return result / np.linalg.norm(result)


def _quaternion_slerp(first: np.ndarray, second: np.ndarray, phase: float) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.linalg.norm(first)
    second = second / np.linalg.norm(second)
    alignment = float(np.dot(first, second))
    if alignment < 0.0:
        second = -second
        alignment = -alignment
    alignment = float(np.clip(alignment, -1.0, 1.0))
    if alignment > 0.9995:
        value = (1.0 - phase) * first + phase * second
        return value / np.linalg.norm(value)
    angle = np.arccos(alignment)
    return (np.sin((1.0 - phase) * angle) * first + np.sin(phase * angle) * second) / np.sin(angle)


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
