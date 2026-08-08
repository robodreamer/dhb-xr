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

    def evaluate(
        self, candidate: PrimitiveCandidate, request: PrimitiveRequest
    ) -> GateResult:
        """Evaluate a candidate without modifying it."""


@dataclass(frozen=True)
class ObservationEpochGate:
    """Reject proposals grounded against a different scene-belief epoch."""

    gate_id: str = "observation_epoch"

    def evaluate(
        self, candidate: PrimitiveCandidate, request: PrimitiveRequest
    ) -> GateResult:
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

    def evaluate(
        self, candidate: PrimitiveCandidate, request: PrimitiveRequest
    ) -> GateResult:
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

    def evaluate(
        self, candidate: PrimitiveCandidate, request: PrimitiveRequest
    ) -> GateResult:
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
            source_rows.append(
                SourceEvaluation(source.source_id, elapsed_ms, len(candidates))
            )
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
