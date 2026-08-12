"""Run a 20-trial matched primitive-source gate benchmark.

The benchmark checks sampled TCP geometry only. It is an architecture/component
benchmark, not link-level collision certification or task success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", type=Path, help="Write the machine-readable JSON report")
    parser.add_argument("--figure", type=Path, help="Write a PNG or SVG dashboard")
    return parser.parse_args()


def run_benchmark(trials: int = 20, *, output: Path | None = None, figure: Path | None = None):
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
    illustration_paths = {}
    for trial in range(int(trials)):
        result = portfolio.evaluate(request)
        for candidate in result.candidates:
            if not illustration_paths and candidate.candidate.motion is None:
                raise RuntimeError("benchmark candidates must contain motion")
            if candidate.candidate.motion is not None:
                illustration_paths.setdefault(
                    candidate.source_id, candidate.candidate.motion.positions.tolist()
                )
            clearance = next(
                gate.metrics["minimum_tcp_clearance_m"]
                for gate in candidate.gates
                if gate.gate_id == "tcp_sphere_clearance"
            )
            rows.append(
                {
                    "condition": "sphere_detour",
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
            "median_generation_ms": float(np.median([row["generation_ms"] for row in source_rows])),
            "minimum_clearance_m": float(
                min(row["minimum_tcp_clearance_m"] for row in source_rows)
            ),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    report = {
        "schema_version": 1,
        "maturity": "sampled_tcp_component_benchmark",
        "seed": None,
        "claim_boundary": [
            "This benchmark checks sampled TCP geometry and common proposal gates.",
            "It does not certify robot links, continuous collision, contacts, or task success.",
        ],
        "summary": summary,
        "rows": rows,
        "illustration": {
            "paths": illustration_paths,
            "sphere_centers": [[0.5, 0.0, 0.0]],
            "sphere_radii_m": [0.17],
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {output}")
    if figure is not None:
        try:
            from examples.plot_primitive_benchmark import plot_report
        except ModuleNotFoundError:
            from plot_primitive_benchmark import plot_report

        plot_report(report, figure)
        print(f"Wrote {figure}")
    return summary, tuple(rows)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.trials < 1:
        raise ValueError("--trials must be positive")
    run_benchmark(arguments.trials, output=arguments.output, figure=arguments.figure)
