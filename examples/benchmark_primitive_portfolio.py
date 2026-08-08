"""Run a 20-trial matched primitive-source gate benchmark.

The benchmark checks sampled TCP geometry only. It is an architecture/component
benchmark, not link-level collision certification or task success.
"""

from __future__ import annotations

import json

import numpy as np

from dhb_xr.primitives import (
    AnalyticPrimitiveSource,
    EndpointGate,
    EvidenceLevel,
    ObservationEpochGate,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitivePortfolio,
    PrimitiveRequest,
    SphereClearanceGate,
)


def run_benchmark(trials: int = 20):
    """Compare a direct path with a shared-waypoint detour for 20 trials."""

    evidence = PrimitiveEvidence(EvidenceLevel.PLANNED, "analytic-benchmark-v1")
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    sources = (
        AnalyticPrimitiveSource(PrimitiveFamily.MINIMUM_JERK, evidence),
        AnalyticPrimitiveSource(
            PrimitiveFamily.VIA_POINT,
            evidence,
            parameters={
                "positions": [[0.5, 0.35, 0.0]],
                "quaternions_wxyz": [identity],
                "samples_per_segment": 26,
            },
        ),
    )
    portfolio = PrimitivePortfolio(
        sources,
        (
            ObservationEpochGate(),
            EndpointGate(),
            SphereClearanceGate(
                centers=np.array([[0.5, 0.0, 0.0]]),
                radii_m=np.array([0.15]),
                margin_m=0.02,
            ),
        ),
    )
    request = PrimitiveRequest(
        "obstacle-detour",
        np.zeros(3),
        identity,
        np.array([1.0, 0.0, 0.0]),
        identity,
        samples=51,
        duration_s=1.0,
        observation_epoch=1,
    )
    rows = []
    for trial in range(int(trials)):
        result = portfolio.evaluate(request)
        for candidate in result.candidates:
            clearance = next(
                gate.metrics["minimum_tcp_clearance_m"]
                for gate in candidate.gates
                if gate.gate_id == "tcp_sphere_clearance"
            )
            rows.append(
                {
                    "trial": trial,
                    "source": candidate.source_id,
                    "accepted": candidate.accepted,
                    "generation_ms": candidate.generation_ms,
                    "minimum_tcp_clearance_m": clearance,
                }
            )
    summary = {
        "evidence_boundary": "sampled_tcp_component_benchmark_only",
        "trials": int(trials),
        "sources": {},
    }
    for source in (item.source_id for item in sources):
        source_rows = [row for row in rows if row["source"] == source]
        summary["sources"][source] = {
            "accepted": sum(row["accepted"] for row in source_rows),
            "attempted": len(source_rows),
            "median_generation_ms": float(
                np.median([row["generation_ms"] for row in source_rows])
            ),
            "minimum_clearance_m": float(
                min(row["minimum_tcp_clearance_m"] for row in source_rows)
            ),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary, tuple(rows)


if __name__ == "__main__":
    run_benchmark()
