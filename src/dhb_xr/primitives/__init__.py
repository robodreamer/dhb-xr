"""Typed manipulation primitives and guarded primitive programs.

Exports are lazy so users of the dependency-light contracts and state machine
do not import optional analytic, memory, optimization, or learning stacks.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS = {
    "dhb_xr.primitives.contracts": (
        "ActionTiming",
        "Comparison",
        "ContactMode",
        "EndEffectorCommand",
        "EndEffectorCommandType",
        "EvidenceLevel",
        "GuardCondition",
        "GuardExpression",
        "GuardMode",
        "MotionPath",
        "PrimitiveCandidate",
        "PrimitiveContractError",
        "PrimitiveEvidence",
        "PrimitiveFamily",
        "PrimitiveProgram",
        "PrimitiveTransition",
        "ScheduledEndEffectorAction",
        "TransitionEvent",
        "guard_all",
        "guard_any",
        "program_from_sequence",
    ),
    "dhb_xr.primitives.analytic": (
        "affordance_normal_path",
        "arc_path",
        "minimum_jerk_blend",
        "minimum_jerk_path",
        "screw_path",
        "via_point_path",
    ),
    "dhb_xr.primitives.sources": (
        "AnalyticPrimitiveSource",
        "PrimitiveRequest",
        "PrimitiveSource",
    ),
    "dhb_xr.primitives.execution": (
        "ExecutionCommand",
        "ExecutionCommandType",
        "ExecutionDecision",
        "ExecutionObservation",
        "ExecutorStatus",
        "GuardEvaluator",
        "PrimitiveExecutionError",
        "PrimitiveFeedback",
        "PrimitiveProgramExecutor",
    ),
    "dhb_xr.primitives.adapters": (
        "BackendPrimitiveSource",
        "DHBRetargetSource",
        "candidate_from_backend_result",
        "motion_path_from_backend_result",
    ),
    "dhb_xr.primitives.memory": (
        "PrimitiveMemory",
        "PrimitiveMemoryMatch",
        "PrimitiveMemoryRecord",
        "VerificationOutcome",
        "VerificationRecord",
    ),
    "dhb_xr.primitives.ingestion": (
        "DemonstrationTrace",
        "MinedPrimitiveSegment",
        "mine_demonstration_segments",
        "segment_to_candidate",
    ),
    "dhb_xr.primitives.segmentation": (
        "BoundaryProposal",
        "exact_gripper_run_control_indices",
        "hysteretic_gripper_control_indices",
        "propose_primitive_boundaries",
        "split_motion_at_indices",
        "stitch_motion_paths",
    ),
    "dhb_xr.primitives.portfolio": (
        "CandidateEvaluation",
        "CandidateGate",
        "EndpointGate",
        "GateResult",
        "ObservationEpochGate",
        "PortfolioEvaluation",
        "PrimitivePortfolio",
        "SourceEvaluation",
        "SphereClearanceGate",
    ),
}
_LAZY_EXPORTS = {
    name: module_name for module_name, names in _EXPORT_GROUPS.items() for name in names
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module 'dhb_xr.primitives' has no attribute {name!r}") from error
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = list(_LAZY_EXPORTS)
