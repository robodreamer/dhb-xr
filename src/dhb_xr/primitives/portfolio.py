"""Common proposal and gate harness for matched primitive-source comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from dhb_xr.primitives.contracts import PrimitiveCandidate, PrimitiveContractError
from dhb_xr.primitives.sources import PrimitiveRequest, PrimitiveSource


@dataclass(frozen=True)
class GateResult:
    """One attributed admission decision."""

    gate_id: str
    accepted: bool
    reason: str
    metrics: Mapping[str, Any] = field(default_factory=dict)


class CandidateGate(Protocol):
    """A gate shared by every candidate source in one comparison."""

    @property
    def gate_id(self) -> str:
        """Stable gate identifier."""

    def evaluate(self, candidate: PrimitiveCandidate, request: PrimitiveRequest) -> GateResult:
        """Evaluate a candidate without modifying it."""


@dataclass(frozen=True)
class ObservationEpochGate:
    """Reject proposals grounded against a different scene-belief epoch."""

    gate_id: str = "observation_epoch"

    def evaluate(self, candidate: PrimitiveCandidate, request: PrimitiveRequest) -> GateResult:
        accepted = candidate.observation_epoch == request.observation_epoch
        return GateResult(
            self.gate_id,
            accepted,
            "epoch matches" if accepted else "candidate is stale",
            {
                "candidate_epoch": candidate.observation_epoch,
                "request_epoch": request.observation_epoch,
            },
        )


@dataclass(frozen=True)
class EndpointGate:
    """Check Cartesian boundary agreement without claiming robot feasibility."""

    position_tolerance_m: float = 0.008
    orientation_tolerance_rad: float = 0.03
    gate_id: str = "cartesian_endpoints"

    def __post_init__(self) -> None:
        if self.position_tolerance_m <= 0.0 or self.orientation_tolerance_rad <= 0.0:
            raise PrimitiveContractError("endpoint tolerances must be positive")

    def evaluate(self, candidate: PrimitiveCandidate, request: PrimitiveRequest) -> GateResult:
        if candidate.motion is None:
            return GateResult(self.gate_id, False, "candidate has no Cartesian motion")
        start_position_error = float(
            np.linalg.norm(candidate.motion.positions[0] - request.start_position)
        )
        goal_position_error = float(
            np.linalg.norm(candidate.motion.positions[-1] - request.goal_position)
        )
        start_orientation_error = _quaternion_distance(
            candidate.motion.quaternions_wxyz[0], request.start_quaternion_wxyz
        )
        goal_orientation_error = _quaternion_distance(
            candidate.motion.quaternions_wxyz[-1], request.goal_quaternion_wxyz
        )
        accepted = (
            max(start_position_error, goal_position_error) <= self.position_tolerance_m
            and max(start_orientation_error, goal_orientation_error)
            <= self.orientation_tolerance_rad
        )
        return GateResult(
            self.gate_id,
            accepted,
            "endpoints pass" if accepted else "Cartesian endpoint tolerance exceeded",
            {
                "start_position_error_m": start_position_error,
                "goal_position_error_m": goal_position_error,
                "start_orientation_error_rad": start_orientation_error,
                "goal_orientation_error_rad": goal_orientation_error,
            },
        )


@dataclass(frozen=True)
class RelationalAnchorGate:
    """Check every scene-grounded intermediate pose at its declared phase."""

    position_tolerance_m: float = 0.008
    orientation_tolerance_rad: float = 0.03
    gate_id: str = "relational_pose_anchors"

    def __post_init__(self) -> None:
        if self.position_tolerance_m <= 0.0 or self.orientation_tolerance_rad <= 0.0:
            raise PrimitiveContractError("anchor tolerances must be positive")

    def evaluate(self, candidate: PrimitiveCandidate, request: PrimitiveRequest) -> GateResult:
        if candidate.motion is None:
            return GateResult(self.gate_id, False, "candidate has no Cartesian motion")
        position_errors = []
        orientation_errors = []
        relation_ids = []
        for anchor in request.relational_anchors:
            index = int(round(anchor.phase * (candidate.motion.samples - 1)))
            position_errors.append(
                float(np.linalg.norm(candidate.motion.positions[index] - anchor.position))
            )
            orientation_errors.append(
                _quaternion_distance(
                    candidate.motion.quaternions_wxyz[index],
                    anchor.quaternion_wxyz,
                )
            )
            relation_ids.append(anchor.relation_id)
        max_position = max(position_errors, default=0.0)
        max_orientation = max(orientation_errors, default=0.0)
        accepted = (
            max_position <= self.position_tolerance_m
            and max_orientation <= self.orientation_tolerance_rad
        )
        return GateResult(
            self.gate_id,
            accepted,
            "relational anchors pass" if accepted else "relational anchor tolerance exceeded",
            {
                "relation_ids": relation_ids,
                "position_errors_m": position_errors,
                "orientation_errors_rad": orientation_errors,
                "max_position_error_m": max_position,
                "max_orientation_error_rad": max_orientation,
            },
        )


@dataclass(frozen=True)
class SphereClearanceGate:
    """Dense TCP-point sphere clearance check, not link or swept-volume safety."""

    centers: np.ndarray
    radii_m: np.ndarray
    margin_m: float = 0.0
    gate_id: str = "tcp_sphere_clearance"

    def __post_init__(self) -> None:
        centers = np.array(self.centers, dtype=np.float64, copy=True)
        radii = np.array(self.radii_m, dtype=np.float64, copy=True)
        margin = float(self.margin_m)
        if centers.ndim != 2 or centers.shape[1:] != (3,) or not len(centers):
            raise PrimitiveContractError("sphere centers must have shape (M, 3), M >= 1")
        if radii.shape != (len(centers),) or np.any(radii <= 0.0):
            raise PrimitiveContractError("sphere radii must be positive with shape (M,)")
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
            raise PrimitiveContractError("sphere geometry must be finite")
        if not np.isfinite(margin) or margin < 0.0:
            raise PrimitiveContractError("margin_m must be finite and non-negative")
        centers.setflags(write=False)
        radii.setflags(write=False)
        object.__setattr__(self, "centers", centers)
        object.__setattr__(self, "radii_m", radii)
        object.__setattr__(self, "margin_m", margin)

    def evaluate(self, candidate: PrimitiveCandidate, request: PrimitiveRequest) -> GateResult:
        del request
        if candidate.motion is None:
            return GateResult(self.gate_id, False, "candidate has no Cartesian motion")
        distances = np.linalg.norm(
            candidate.motion.positions[:, None, :] - self.centers[None, :, :],
            axis=2,
        )
        clearances = distances - self.radii_m[None, :] - self.margin_m
        minimum = float(np.min(clearances))
        accepted = minimum >= 0.0
        return GateResult(
            self.gate_id,
            accepted,
            "TCP samples clear spheres" if accepted else "TCP sample intersects sphere",
            {
                "minimum_tcp_clearance_m": minimum,
                "scope": "sampled_tcp_points_only",
            },
        )


@dataclass(frozen=True)
class OrientedPayloadAabbClearanceGate:
    """Check a carried oriented box against fixture AABBs at dense path samples.

    The test uses the separating-axis theorem for the full payload volume at
    interpolated poses.  It remains a dense discrete geometric gate: it does
    not include robot links, contact dynamics, or continuous-time certification.
    """

    payload_half_extents_m: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_half_extents_m: np.ndarray
    payload_center_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    margin_m: float = 0.0
    temporal_substeps: int = 4
    gate_id: str = "oriented_payload_aabb_clearance"

    def __post_init__(self) -> None:
        payload_half_extents = np.array(self.payload_half_extents_m, dtype=np.float64, copy=True)
        payload_offset = np.array(self.payload_center_offset_m, dtype=np.float64, copy=True)
        centers = np.array(self.obstacle_centers, dtype=np.float64, copy=True)
        obstacle_half_extents = np.array(self.obstacle_half_extents_m, dtype=np.float64, copy=True)
        margin = float(self.margin_m)
        temporal_substeps = int(self.temporal_substeps)
        if payload_half_extents.shape != (3,) or np.any(payload_half_extents <= 0.0):
            raise PrimitiveContractError("payload_half_extents_m must be positive with shape (3,)")
        if payload_offset.shape != (3,):
            raise PrimitiveContractError("payload_center_offset_m must have shape (3,)")
        if centers.ndim != 2 or centers.shape[1:] != (3,) or not len(centers):
            raise PrimitiveContractError("obstacle_centers must have shape (M, 3), M >= 1")
        if obstacle_half_extents.shape != centers.shape or np.any(obstacle_half_extents <= 0.0):
            raise PrimitiveContractError(
                "obstacle_half_extents_m must be positive and match obstacle_centers"
            )
        if not all(
            np.all(np.isfinite(value))
            for value in (
                payload_half_extents,
                payload_offset,
                centers,
                obstacle_half_extents,
            )
        ):
            raise PrimitiveContractError("payload and obstacle geometry must be finite")
        if not np.isfinite(margin) or margin < 0.0:
            raise PrimitiveContractError("margin_m must be finite and non-negative")
        if temporal_substeps < 1:
            raise PrimitiveContractError("temporal_substeps must be at least one")
        for value in (
            payload_half_extents,
            payload_offset,
            centers,
            obstacle_half_extents,
        ):
            value.setflags(write=False)
        object.__setattr__(self, "payload_half_extents_m", payload_half_extents)
        object.__setattr__(self, "payload_center_offset_m", payload_offset)
        object.__setattr__(self, "obstacle_centers", centers)
        object.__setattr__(self, "obstacle_half_extents_m", obstacle_half_extents)
        object.__setattr__(self, "margin_m", margin)
        object.__setattr__(self, "temporal_substeps", temporal_substeps)

    def evaluate(self, candidate: PrimitiveCandidate, request: PrimitiveRequest) -> GateResult:
        del request
        if candidate.motion is None:
            return GateResult(self.gate_id, False, "candidate has no Cartesian motion")
        positions, quaternions = _densify_motion(
            candidate.motion.positions,
            candidate.motion.quaternions_wxyz,
            self.temporal_substeps,
        )
        rotations = _quaternion_rotation_matrices(quaternions)
        payload_centers = positions + np.einsum(
            "sij,j->si", rotations, self.payload_center_offset_m
        )
        separations = _obb_aabb_signed_separations(
            payload_centers,
            rotations,
            self.payload_half_extents_m,
            self.obstacle_centers,
            self.obstacle_half_extents_m,
            self.margin_m,
        )
        closest_sample, closest_obstacle = np.unravel_index(
            int(np.argmin(separations)), separations.shape
        )
        minimum_signed_separation = float(separations[closest_sample, closest_obstacle])
        accepted = bool(minimum_signed_separation >= 0.0)
        return GateResult(
            self.gate_id,
            accepted,
            (
                "oriented payload clears fixture boxes"
                if accepted
                else "oriented payload intersects a fixture box"
            ),
            {
                "minimum_sat_separation_m": float(minimum_signed_separation),
                "closest_dense_sample": closest_sample,
                "closest_obstacle": closest_obstacle,
                "dense_pose_samples": len(positions),
                "temporal_substeps": self.temporal_substeps,
                "scope": "dense_discrete_payload_obb_vs_fixture_aabb",
            },
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate evaluated by the same declared gate sequence."""

    source_id: str
    candidate: PrimitiveCandidate
    generation_ms: float
    gates: tuple[GateResult, ...]

    @property
    def accepted(self) -> bool:
        return all(gate.accepted for gate in self.gates)


@dataclass(frozen=True)
class SourceEvaluation:
    """Generation outcome for one source, including concise expected failures."""

    source_id: str
    generation_ms: float
    candidate_count: int
    error: str | None = None


@dataclass(frozen=True)
class PortfolioEvaluation:
    """Complete row-level output of one matched portfolio request."""

    sources: tuple[SourceEvaluation, ...]
    candidates: tuple[CandidateEvaluation, ...]

    @property
    def accepted(self) -> tuple[CandidateEvaluation, ...]:
        return tuple(candidate for candidate in self.candidates if candidate.accepted)


class PrimitivePortfolio:
    """Run interchangeable sources and admit them through identical gates."""

    def __init__(
        self,
        sources: Sequence[PrimitiveSource],
        gates: Sequence[CandidateGate],
    ) -> None:
        self.sources = tuple(sources)
        self.gates = tuple(gates)
        if not self.sources:
            raise PrimitiveContractError("portfolio requires at least one source")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise PrimitiveContractError("portfolio source ids must be unique")
        gate_ids = [gate.gate_id for gate in self.gates]
        if len(gate_ids) != len(set(gate_ids)):
            raise PrimitiveContractError("portfolio gate ids must be unique")

    def evaluate(self, request: PrimitiveRequest) -> PortfolioEvaluation:
        source_rows = []
        candidate_rows = []
        for source in self.sources:
            started = perf_counter()
            try:
                candidates = source.propose(request)
            except Exception as exc:
                elapsed_ms = (perf_counter() - started) * 1_000.0
                source_rows.append(
                    SourceEvaluation(
                        source.source_id,
                        elapsed_ms,
                        0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            elapsed_ms = (perf_counter() - started) * 1_000.0
            source_rows.append(SourceEvaluation(source.source_id, elapsed_ms, len(candidates)))
            for candidate in candidates:
                gate_results = []
                for gate in self.gates:
                    try:
                        gate_results.append(gate.evaluate(candidate, request))
                    except Exception as exc:
                        gate_results.append(
                            GateResult(
                                gate.gate_id,
                                False,
                                f"gate error: {type(exc).__name__}: {exc}",
                            )
                        )
                candidate_rows.append(
                    CandidateEvaluation(
                        source.source_id,
                        candidate,
                        elapsed_ms,
                        tuple(gate_results),
                    )
                )
        return PortfolioEvaluation(tuple(source_rows), tuple(candidate_rows))


def _quaternion_distance(first: np.ndarray, second: np.ndarray) -> float:
    alignment = abs(float(np.dot(first, second)))
    return float(2.0 * np.arccos(np.clip(alignment, 0.0, 1.0)))


def _densify_motion(
    positions: np.ndarray,
    quaternions: np.ndarray,
    substeps: int,
) -> tuple[np.ndarray, np.ndarray]:
    dense_positions = []
    dense_quaternions = []
    for index in range(len(positions) - 1):
        for substep in range(substeps):
            phase = substep / substeps
            dense_positions.append((1.0 - phase) * positions[index] + phase * positions[index + 1])
            dense_quaternions.append(
                _quaternion_slerp(quaternions[index], quaternions[index + 1], phase)
            )
    dense_positions.append(positions[-1])
    dense_quaternions.append(quaternions[-1])
    return np.asarray(dense_positions), np.asarray(dense_quaternions)


def _quaternion_slerp(first: np.ndarray, second: np.ndarray, phase: float) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
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


def _quaternion_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _quaternion_rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    """Vectorized counterpart of :func:`_quaternion_rotation_matrix`."""

    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("quaternions must have shape (N, 4)")
    w, x, y, z = values.T
    rotations = np.empty((len(values), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - z * w)
    rotations[:, 0, 2] = 2.0 * (x * z + y * w)
    rotations[:, 1, 0] = 2.0 * (x * y + z * w)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - x * w)
    rotations[:, 2, 0] = 2.0 * (x * z - y * w)
    rotations[:, 2, 1] = 2.0 * (y * z + x * w)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def _obb_aabb_signed_separations(
    payload_centers: np.ndarray,
    payload_rotations: np.ndarray,
    payload_half_extents: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_half_extents: np.ndarray,
    margin_m: float,
) -> np.ndarray:
    """Evaluate every payload-pose/obstacle SAT pair without Python loops."""

    sample_count = len(payload_centers)
    payload_axes = np.transpose(payload_rotations, (0, 2, 1))
    world_axes = np.broadcast_to(np.eye(3), (sample_count, 3, 3))
    cross_axes = np.cross(
        payload_axes[:, :, None, :],
        np.eye(3)[None, None, :, :],
    ).reshape(sample_count, 9, 3)
    axes = np.concatenate((payload_axes, world_axes, cross_axes), axis=1)
    norms = np.linalg.norm(axes, axis=2)
    valid = norms > 1.0e-10
    directions = np.divide(
        axes,
        norms[:, :, None],
        out=np.zeros_like(axes),
        where=valid[:, :, None],
    )

    center_delta = obstacle_centers[None, :, :] - payload_centers[:, None, :]
    projected_distance = np.abs(np.einsum("sod,sad->soa", center_delta, directions))
    payload_local_axes = np.einsum("skd,sad->sak", payload_axes, directions)
    payload_radius = np.einsum("k,sak->sa", payload_half_extents, np.abs(payload_local_axes))
    obstacle_radius = np.einsum("od,sad->soa", obstacle_half_extents, np.abs(directions))
    gaps = projected_distance - payload_radius[:, None, :] - obstacle_radius - float(margin_m)
    gaps = np.where(valid[:, None, :], gaps, -np.inf)
    return np.max(gaps, axis=2)


def _obb_aabb_signed_separation(
    payload_center: np.ndarray,
    payload_rotation: np.ndarray,
    payload_half_extents: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_half_extents: np.ndarray,
    margin_m: float,
) -> float:
    """Return a separating-axis clearance proxy; negative means intersection."""

    axes = [payload_rotation[:, index] for index in range(3)]
    axes.extend(np.eye(3)[index] for index in range(3))
    axes.extend(
        np.cross(payload_rotation[:, payload_axis], np.eye(3)[world_axis])
        for payload_axis in range(3)
        for world_axis in range(3)
    )
    delta = np.asarray(obstacle_center) - np.asarray(payload_center)
    gaps = []
    for axis in axes:
        norm = float(np.linalg.norm(axis))
        if norm <= 1.0e-10:
            continue
        direction = axis / norm
        payload_radius = float(
            np.sum(payload_half_extents * np.abs(payload_rotation.T @ direction))
        )
        obstacle_radius = float(np.sum(obstacle_half_extents * np.abs(direction)))
        gaps.append(
            abs(float(np.dot(delta, direction))) - payload_radius - obstacle_radius - margin_m
        )
    return float(max(gaps))
