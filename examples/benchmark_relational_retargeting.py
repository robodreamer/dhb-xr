#!/usr/bin/env python3
"""Matched persistent-FATROP screen with independently moved pose anchors.

The benchmark changes two thick aperture poses independently while holding the
reference memory fixed.  Both FATROP variants share dynamics, anchors, solver,
initialization, and gates; only DHB-invariant versus sampled-Cartesian
regularization differs.  Orientation-anchor interpolation is shared, so this
screen isolates the *linear* representation effect and is not robot execution.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dhb_xr.optimization.relational_fatrop import PersistentRelationalFatropRetargeter
from dhb_xr.primitives import (
    AnalyticPrimitiveSource,
    EndpointCorrectedReferenceSource,
    EndpointGate,
    EvidenceLevel,
    ObservationEpochGate,
    OrientedPayloadAabbClearanceGate,
    PersistentRelationalFatropSource,
    PrimitiveEvidence,
    PrimitiveFamily,
    PrimitivePortfolio,
    PrimitiveRequest,
    RelationalAnchorGate,
    RelationalAnchorReferenceSource,
    RelationalPoseAnchor,
    via_point_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-trials", type=int, default=60)
    parser.add_argument("--easy-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=260810)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--solver", choices=("fatrop", "ipopt"), default="fatrop")
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
            np.array([0.10, -0.18, 0.52]),
            np.array([0.24, -0.05, 0.50]),
            np.array([0.38, -0.05, 0.50]),
            np.array([0.52, -0.05, 0.50]),
            np.array([0.68, 0.06, 0.53]),
            np.array([0.84, 0.06, 0.53]),
            np.array([1.00, 0.06, 0.53]),
            np.array([1.14, 0.16, 0.50]),
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
        samples_per_segment=max(5, samples // 7),
        duration_s=3.0,
    )


def fixture_boxes(anchors: tuple[RelationalPoseAnchor, ...]) -> tuple[np.ndarray, np.ndarray]:
    centers = []
    half_extents = []

    def add_aperture(position: np.ndarray, half_y: float, half_z: float) -> None:
        x, y, z = position
        outer_y = 0.22
        outer_z = 0.16
        slab_half_x = 0.045
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

    add_aperture(anchors[1].position, half_y=0.078, half_z=0.045)
    add_aperture(anchors[4].position, half_y=0.045, half_z=0.078)
    return np.asarray(centers), np.asarray(half_extents)


def make_request(
    reference,
    rng: np.random.Generator,
    trial: int,
    samples: int,
    *,
    hard: bool,
) -> PrimitiveRequest:
    scale = 1.0 if hard else 0.15
    first_offset = scale * rng.uniform([-0.006, -0.035, -0.018], [0.006, 0.035, 0.018])
    second_offset = scale * rng.uniform([-0.006, -0.035, -0.018], [0.006, 0.035, 0.018])
    anchors = tuple(
        RelationalPoseAnchor(phase, base + offset, quaternion, relation)
        for phase, base, offset, quaternion, relation in (
            (
                1.0 / 7.0,
                np.array([0.24, -0.05, 0.50]),
                first_offset,
                roll_quaternion(0.0),
                "wide_short_entry",
            ),
            (
                2.0 / 7.0,
                np.array([0.38, -0.05, 0.50]),
                first_offset,
                roll_quaternion(0.0),
                "wide_short_center",
            ),
            (
                3.0 / 7.0,
                np.array([0.52, -0.05, 0.50]),
                first_offset,
                roll_quaternion(0.0),
                "wide_short_exit",
            ),
            (
                4.0 / 7.0,
                np.array([0.68, 0.06, 0.53]),
                second_offset,
                roll_quaternion(np.pi / 2.0),
                "narrow_tall_entry",
            ),
            (
                5.0 / 7.0,
                np.array([0.84, 0.06, 0.53]),
                second_offset,
                roll_quaternion(np.pi / 2.0),
                "narrow_tall_center",
            ),
            (
                6.0 / 7.0,
                np.array([1.00, 0.06, 0.53]),
                second_offset,
                roll_quaternion(np.pi / 2.0),
                "narrow_tall_exit",
            ),
        )
    )
    boundary_scale = 0.006 if hard else 0.001
    start = reference.positions[0] + rng.uniform(-boundary_scale, boundary_scale, 3)
    goal = reference.positions[-1] + rng.uniform(-boundary_scale, boundary_scale, 3)
    return PrimitiveRequest(
        primitive_id=f"{'hard' if hard else 'easy'}-{trial:03d}",
        start_position=start,
        start_quaternion_wxyz=roll_quaternion(0.0),
        goal_position=goal,
        goal_quaternion_wxyz=roll_quaternion(np.pi / 2.0),
        samples=samples,
        duration_s=3.0,
        observation_epoch=trial + 1,
        reference_frames={"fixture": "oracle_fixture_frame"},
        relational_anchors=anchors,
        metadata={
            "oracle_relational_anchors": True,
            "comparison_scope": "linear_representation_only",
        },
    )


def main() -> None:
    args = parse_args()
    if args.hard_trials < 1 or args.easy_trials < 1:
        raise ValueError("--hard-trials and --easy-trials must be positive")
    reference = reference_motion(args.samples)
    evidence = PrimitiveEvidence(EvidenceLevel.PLANNED, "matched_relational_anchor_screen")
    use_fatrop = args.solver == "fatrop"
    dhb_retargeter = PersistentRelationalFatropRetargeter(
        reference.positions,
        reference.quaternions_wxyz,
        samples=args.samples,
        regularization="dhb",
        use_fatrop=use_fatrop,
    )
    cartesian_retargeter = PersistentRelationalFatropRetargeter(
        reference.positions,
        reference.quaternions_wxyz,
        samples=args.samples,
        regularization="cartesian",
        use_fatrop=use_fatrop,
    )
    sources = (
        AnalyticPrimitiveSource(PrimitiveFamily.MINIMUM_JERK, evidence),
        EndpointCorrectedReferenceSource(reference.positions, reference.quaternions_wxyz, evidence),
        RelationalAnchorReferenceSource(reference.positions, reference.quaternions_wxyz, evidence),
        PersistentRelationalFatropSource(cartesian_retargeter, evidence),
        PersistentRelationalFatropSource(dhb_retargeter, evidence),
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    conditions = (
        ("hard_relational_shift", True, args.hard_trials),
        ("easy_control", False, args.easy_trials),
    )
    for condition, hard, trial_count in conditions:
        for trial in range(trial_count):
            request = make_request(reference, rng, trial, args.samples, hard=hard)
            centers, extents = fixture_boxes(request.relational_anchors)
            gates = (
                ObservationEpochGate(),
                EndpointGate(1.0e-6, 1.0e-6),
                RelationalAnchorGate(0.008, np.deg2rad(1.0)),
                OrientedPayloadAabbClearanceGate(
                    payload_half_extents_m=np.array([0.10, 0.055, 0.020]),
                    obstacle_centers=centers,
                    obstacle_half_extents_m=extents,
                    margin_m=0.001,
                    temporal_substeps=8,
                ),
            )
            result = PrimitivePortfolio(sources, gates).evaluate(request)
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
                        "accepted": candidate.accepted if candidate else False,
                        "gates": {
                            gate.gate_id: {
                                "accepted": gate.accepted,
                                "reason": gate.reason,
                                "metrics": dict(gate.metrics),
                            }
                            for gate in candidate.gates
                        }
                        if candidate
                        else {},
                        "backend": dict(candidate.candidate.metadata.get("backend", {}))
                        if candidate
                        else {},
                    }
                )
            accepted = [candidate.source_id for candidate in result.accepted]
            errors = [source.error for source in result.sources if source.error]
            print(
                f"[{condition} {trial + 1:02d}/{trial_count:02d}] "
                f"accepted={accepted or ['none']} errors={errors or ['none']}",
                flush=True,
            )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["source"])].append(row)
    summary = {}
    for (condition, source), values in sorted(grouped.items()):
        timings = [row["generation_ms"] for row in values]
        summary[f"{condition}/{source}"] = {
            "condition": condition,
            "source": source,
            "trials": len(values),
            "accepted": sum(bool(row["accepted"]) for row in values),
            "accepted_rate": float(np.mean([bool(row["accepted"]) for row in values])),
            "generation_ms_median": float(np.median(timings)),
            "generation_ms_p95": float(np.percentile(timings, 95)),
            "errors": sorted({row["source_error"] for row in values if row["source_error"]}),
        }
    report = {
        "schema_version": 1,
        "maturity": "exploratory_relational_representation_screen",
        "seed": args.seed,
        "trial_counts": {
            "hard_relational_shift": args.hard_trials,
            "easy_control": args.easy_trials,
        },
        "samples": args.samples,
        "solver": args.solver,
        "frozen_before_result": {
            "primary_metric": "paired hard-condition all-gate acceptance",
            "superiority_margin_percentage_points": 15.0,
            "easy_control_regression_tolerance_percentage_points": 5.0,
            "anchor_position_tolerance_m": 0.008,
            "anchor_orientation_tolerance_deg": 1.0,
        },
        "claim_boundary": [
            (
                "Oracle relational anchors isolate motion representation from "
                "perception and retrieval."
            ),
            (
                "Both FATROP variants share graph structure, anchors, solver, "
                "initialization, and gates."
            ),
            "Shared quaternion correction limits the comparison to linear-path representation.",
            "Dense payload geometry is not robot-link feasibility or physical task success.",
        ],
        "persistent_solver": {
            "dhb_build_ms": dhb_retargeter.build_ms,
            "dhb_solve_count": dhb_retargeter.solve_count,
            "cartesian_build_ms": cartesian_retargeter.build_ms,
            "cartesian_solve_count": cartesian_retargeter.solve_count,
        },
        "summary": summary,
        "rows": rows,
    }
    report["claim_gate"] = _claim_gate(rows)
    print(json.dumps(summary, indent=2), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=_json_default) + "\n")
        print(f"Wrote {args.output}", flush=True)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _claim_gate(rows: list[dict]) -> dict:
    hard = [row for row in rows if row["condition"] == "hard_relational_shift"]
    by_source = defaultdict(dict)
    for row in hard:
        by_source[row["source"]][row["trial"]] = bool(row["accepted"])
    dhb = by_source["dhb-regularized:fatrop_relational"]
    raw = by_source["reference:relational_anchor_se3"]
    common = sorted(set(dhb) & set(raw))
    differences = np.asarray([float(dhb[index]) - float(raw[index]) for index in common])
    observed = float(np.mean(differences)) if len(differences) else float("nan")
    if len(differences):
        rng = np.random.default_rng(26081001)
        sampled = rng.choice(differences, size=(10_000, len(differences)), replace=True)
        lower, upper = np.percentile(np.mean(sampled, axis=1), [2.5, 97.5])
    else:
        lower = upper = float("nan")
    passed = bool(observed >= 0.15 and lower > 0.0)
    return {
        "passed": passed,
        "comparison": ("dhb-regularized:fatrop_relational minus reference:relational_anchor_se3"),
        "paired_trials": len(common),
        "acceptance_delta": observed,
        "paired_bootstrap_95_interval": [float(lower), float(upper)],
        "required_delta": 0.15,
        "required_lower_bound": 0.0,
        "bootstrap_resamples": 10_000,
        "bootstrap_seed": 26081001,
    }


if __name__ == "__main__":
    main()
