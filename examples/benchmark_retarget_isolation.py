#!/usr/bin/env python3
"""Exploratory matched-source screen for nontrivial full-SE(3) retargeting.

The same oracle reference and boundary request are given to every reuse source.
The hard condition checks an oriented carried box against two offset apertures;
the easy condition is an open-space negative control. This is a geometric
component screen, not robot feasibility or physical task success.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dhb_xr.primitives import (
    AnalyticPrimitiveSource,
    DHBRetargetSource,
    EndpointCorrectedReferenceSource,
    EndpointGate,
    EvidenceLevel,
    ObservationEpochGate,
    OrientedPayloadAabbClearanceGate,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitivePortfolio,
    PrimitiveRequest,
    via_point_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=260809)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--dhb-backend", default="fatrop")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def roll_quaternion(angle_rad: float) -> np.ndarray:
    return np.array(
        [np.cos(angle_rad / 2.0), np.sin(angle_rad / 2.0), 0.0, 0.0],
        dtype=np.float64,
    )


def reference_motion(samples: int):
    return via_point_path(
        positions=(
            np.array([0.12, -0.18, 0.52]),
            np.array([0.30, -0.05, 0.50]),
            np.array([0.42, -0.05, 0.50]),
            np.array([0.54, -0.05, 0.50]),
            np.array([0.64, 0.06, 0.53]),
            np.array([0.76, 0.06, 0.53]),
            np.array([0.88, 0.06, 0.53]),
            np.array([1.05, 0.16, 0.50]),
        ),
        quaternions_wxyz=(
            roll_quaternion(0.0),
            roll_quaternion(0.0),
            roll_quaternion(0.0),
            roll_quaternion(0.0),
            roll_quaternion(np.pi / 2.0),
            roll_quaternion(np.pi / 2.0),
            roll_quaternion(np.pi / 2.0),
            roll_quaternion(np.pi / 2.0),
        ),
        samples_per_segment=max(4, samples // 7),
        duration_s=3.0,
    )


def aperture_fixture_boxes() -> tuple[np.ndarray, np.ndarray]:
    centers = []
    half_extents = []

    def add_aperture(x: float, y: float, z: float, half_y: float, half_z: float):
        outer_y = 0.22
        outer_z = 0.16
        slab_half_x = 0.012
        side_half_y = (outer_y - half_y) / 2.0
        vertical_half_z = (outer_z - half_z) / 2.0
        centers.extend(
            [
                [x, y - (outer_y + half_y) / 2.0, z],
                [x, y + (outer_y + half_y) / 2.0, z],
                [x, y, z - (outer_z + half_z) / 2.0],
                [x, y, z + (outer_z + half_z) / 2.0],
            ]
        )
        half_extents.extend(
            [
                [slab_half_x, side_half_y, outer_z],
                [slab_half_x, side_half_y, outer_z],
                [slab_half_x, half_y, vertical_half_z],
                [slab_half_x, half_y, vertical_half_z],
            ]
        )

    # The first aperture is wide and short; the second is narrow and tall.
    # The asymmetric payload must roll approximately 90 degrees between them.
    add_aperture(0.42, -0.05, 0.50, half_y=0.070, half_z=0.036)
    add_aperture(0.76, 0.06, 0.53, half_y=0.036, half_z=0.070)
    return np.asarray(centers, dtype=np.float64), np.asarray(half_extents, dtype=np.float64)


def make_request(
    reference,
    rng: np.random.Generator,
    trial: int,
    samples: int,
    condition: str,
) -> PrimitiveRequest:
    if condition == "oriented_apertures":
        start_jitter = rng.uniform([-0.008, -0.012, -0.010], [0.008, 0.012, 0.010])
        goal_jitter = rng.uniform([-0.008, -0.012, -0.010], [0.008, 0.012, 0.010])
        roll_jitter = np.deg2rad(5.0)
    else:
        start_jitter = rng.uniform(-0.02, 0.02, size=3)
        goal_jitter = rng.uniform(-0.02, 0.02, size=3)
        roll_jitter = np.deg2rad(10.0)
    start_roll = rng.uniform(-roll_jitter, roll_jitter)
    goal_roll = np.pi / 2.0 + rng.uniform(-roll_jitter, roll_jitter)
    return PrimitiveRequest(
        primitive_id=f"{condition}-{trial:03d}",
        start_position=reference.positions[0] + start_jitter,
        start_quaternion_wxyz=roll_quaternion(start_roll),
        goal_position=reference.positions[-1] + goal_jitter,
        goal_quaternion_wxyz=roll_quaternion(goal_roll),
        samples=samples,
        duration_s=3.0,
        observation_epoch=trial + 1,
        reference_frames={
            "payload": "oracle_payload_frame",
            "fixture": "oracle_fixture_frame",
        },
        metadata={
            "condition": condition,
            "oracle_reference": True,
            "scene_geometry_available_to_retargeter": False,
        },
    )


def gate_row(evaluation) -> dict:
    return {
        gate.gate_id: {
            "accepted": gate.accepted,
            "reason": gate.reason,
            "metrics": dict(gate.metrics),
        }
        for gate in evaluation.gates
    }


def main() -> None:
    args = parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")
    if args.samples < 8:
        raise ValueError("--samples must be at least eight")
    rng = np.random.default_rng(args.seed)
    reference = reference_motion(args.samples)
    evidence = PrimitiveEvidence(
        EvidenceLevel.PLANNED,
        "synthetic_oracle_reference_for_matched_component_screen",
    )
    sources = (
        AnalyticPrimitiveSource(PrimitiveFamily.MINIMUM_JERK, evidence),
        EndpointCorrectedReferenceSource(reference.positions, reference.quaternions_wxyz, evidence),
        DHBRetargetSource(
            reference.positions,
            reference.quaternions_wxyz,
            evidence,
            backend=args.dhb_backend,
            allow_fallback=False,
        ),
    )
    centers, half_extents = aperture_fixture_boxes()
    rows = []
    for condition in ("oriented_apertures", "open_space_control"):
        gates = [
            ObservationEpochGate(),
            EndpointGate(
                position_tolerance_m=1.0e-6,
                orientation_tolerance_rad=1.0e-6,
            ),
        ]
        if condition == "oriented_apertures":
            gates.append(
                OrientedPayloadAabbClearanceGate(
                    payload_half_extents_m=np.array([0.10, 0.055, 0.020]),
                    obstacle_centers=centers,
                    obstacle_half_extents_m=half_extents,
                    margin_m=0.002,
                    temporal_substeps=8,
                )
            )
        portfolio = PrimitivePortfolio(sources, gates)
        for trial in range(args.trials):
            request = make_request(reference, rng, trial, args.samples, condition)
            result = portfolio.evaluate(request)
            source_timing = {source.source_id: source for source in result.sources}
            candidates = {candidate.source_id: candidate for candidate in result.candidates}
            for source in result.sources:
                candidate = candidates.get(source.source_id)
                rows.append(
                    {
                        "condition": condition,
                        "trial": trial,
                        "source": source.source_id,
                        "generation_ms": source.generation_ms,
                        "source_error": source.error,
                        "accepted": candidate.accepted if candidate is not None else False,
                        "gates": gate_row(candidate) if candidate is not None else {},
                    }
                )
            accepted = [
                source_id for source_id, candidate in candidates.items() if candidate.accepted
            ]
            timings = ", ".join(
                f"{source_id}={source_timing[source_id].generation_ms:.1f}ms"
                for source_id in source_timing
            )
            print(
                f"[{condition} {trial + 1:02d}/{args.trials:02d}] "
                f"accepted={accepted or ['none']} ({timings})",
                flush=True,
            )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["source"])].append(row)
    summary = {}
    for (condition, source), values in sorted(grouped.items()):
        summary[f"{condition}/{source}"] = {
            "condition": condition,
            "source": source,
            "trials": len(values),
            "accepted": sum(bool(value["accepted"]) for value in values),
            "accepted_rate": float(np.mean([bool(value["accepted"]) for value in values])),
            "generation_ms_median": float(np.median([value["generation_ms"] for value in values])),
            "generation_ms_p95": float(
                np.percentile([value["generation_ms"] for value in values], 95)
            ),
            "errors": sorted({value["source_error"] for value in values if value["source_error"]}),
        }
    report = {
        "schema_version": 1,
        "maturity": "exploratory_geometric_component_screen",
        "seed": args.seed,
        "trials_per_condition": args.trials,
        "samples": args.samples,
        "dhb_backend": args.dhb_backend,
        "oracle_reference": True,
        "claim_boundary": [
            "The screen compares retargeting geometry under common gates.",
            (
                "It does not measure robot feasibility, MuJoCo contacts, task "
                "success, retrieval, or estimated perception."
            ),
            (
                "The payload gate is dense discrete OBB-vs-AABB geometry, not "
                "continuous collision certification."
            ),
            (
                "Results are exploratory until a separate protocol is frozen "
                "before confirmatory trials."
            ),
        ],
        "summary": summary,
        "rows": rows,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
