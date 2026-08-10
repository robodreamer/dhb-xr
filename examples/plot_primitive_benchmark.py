#!/usr/bin/env python3
"""Render a reviewable dashboard from a primitive benchmark JSON report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import fill
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Benchmark JSON report")
    parser.add_argument("--output", type=Path, required=True, help="PNG or SVG output path")
    return parser.parse_args()


def plot_report(report: dict[str, Any], output: Path) -> None:
    """Write a three-panel geometry, acceptance, and latency dashboard."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    rows = report.get("rows", [])
    if not rows:
        raise ValueError("report must contain at least one benchmark row")
    output = Path(output)
    if output.suffix.lower() not in {".png", ".svg"}:
        raise ValueError("figure output must end in .png or .svg")

    conditions = list(dict.fromkeys(str(row.get("condition", "benchmark")) for row in rows))
    sources = list(dict.fromkeys(str(row["source"]) for row in rows))
    colors = plt.colormaps["tab10"](np.linspace(0.0, 0.9, max(len(sources), 2)))
    source_colors = dict(zip(sources, colors))

    fig = plt.figure(figsize=(15.5, 5.4), constrained_layout=True)
    axes = fig.subplots(1, 3)
    fig.suptitle(_report_title(report), fontsize=16, fontweight="bold")

    illustration = report.get("illustration", {})
    paths = illustration.get("paths", {})
    if paths:
        for source, positions in paths.items():
            positions = np.asarray(positions, dtype=np.float64)
            axes[0].plot(
                positions[:, 0],
                positions[:, 1],
                linewidth=2.3,
                color=source_colors.get(source),
                label=_short_source(source),
            )
        for center, radius in zip(
            illustration.get("sphere_centers", ()), illustration.get("sphere_radii_m", ())
        ):
            axes[0].add_patch(
                Circle(
                    np.asarray(center)[:2],
                    float(radius),
                    facecolor="#d95f5f33",
                    edgecolor="#a32727",
                    linewidth=1.5,
                )
            )
        for center, half_extent in zip(
            illustration.get("fixture_centers", ()),
            illustration.get("fixture_half_extents_m", ()),
        ):
            center = np.asarray(center, dtype=np.float64)
            half_extent = np.asarray(half_extent, dtype=np.float64)
            axes[0].add_patch(
                Rectangle(
                    center[:2] - half_extent[:2],
                    2.0 * half_extent[0],
                    2.0 * half_extent[1],
                    facecolor="#80808025",
                    edgecolor="#555555",
                    linewidth=0.8,
                )
            )
        axes[0].set_aspect("equal", adjustable="datalim")
        axes[0].set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        axes[0].set_title("Representative path geometry")
        axes[0].legend(fontsize=8, loc="best")
    else:
        boundary = report.get("claim_boundary", [report.get("maturity", "exploratory")])
        axes[0].text(
            0.04,
            0.96,
            "\n\n".join(fill(str(item), 42) for item in boundary[:4]),
            va="top",
            transform=axes[0].transAxes,
            fontsize=9,
        )
        axes[0].set_title("Claim boundary")
        axes[0].axis("off")

    width = 0.8 / max(len(sources), 1)
    x_values = np.arange(len(conditions), dtype=np.float64)
    for source_index, source in enumerate(sources):
        rates = []
        for condition in conditions:
            selected = [
                bool(row["accepted"])
                for row in rows
                if str(row.get("condition", "benchmark")) == condition
                and str(row["source"]) == source
            ]
            rates.append(100.0 * float(np.mean(selected)) if selected else np.nan)
        offset = (source_index - (len(sources) - 1) / 2.0) * width
        axes[1].bar(
            x_values + offset,
            rates,
            width=width,
            label=_short_source(source),
            color=source_colors[source],
        )
    axes[1].set_xticks(x_values, [_short_condition(item) for item in conditions], rotation=12)
    axes[1].set_ylim(0.0, 105.0)
    axes[1].set_ylabel("all-gate acceptance (%)")
    axes[1].set_title("Matched gate outcome")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=7, loc="best")

    timing_values = [
        [float(row["generation_ms"]) for row in rows if str(row["source"]) == source]
        for source in sources
    ]
    box = axes[2].boxplot(timing_values, patch_artist=True, tick_labels=["" for _ in sources])
    for patch, source in zip(box["boxes"], sources):
        patch.set_facecolor(source_colors[source])
        patch.set_alpha(0.75)
    axes[2].set_xticks(
        np.arange(1, len(sources) + 1),
        [_short_source(item) for item in sources],
        rotation=25,
        ha="right",
    )
    axes[2].set_ylabel("generation latency (ms)")
    axes[2].set_title("Steady-state request latency")
    axes[2].grid(axis="y", alpha=0.25)
    positive = [value for values in timing_values for value in values if value > 0.0]
    if positive and max(positive) / min(positive) > 100.0:
        axes[2].set_yscale("log")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _report_title(report: dict[str, Any]) -> str:
    maturity = str(report.get("maturity", "primitive benchmark")).replace("_", " ")
    seed = report.get("seed")
    return f"DHB-XR {maturity}" + (f" · seed {seed}" if seed is not None else "")


def _short_source(value: str) -> str:
    return (
        value.replace("analytic:", "")
        .replace("reference:endpoint_corrected_se3", "endpoint corrected")
        .replace("reference:relational_anchor_se3", "relational replay")
        .replace(":fatrop_relational", " FATROP")
        .replace(":ipopt_relational", " IPOPT")
    )


def _short_condition(value: str) -> str:
    return value.replace("_", " ")


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text())
    plot_report(report, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
