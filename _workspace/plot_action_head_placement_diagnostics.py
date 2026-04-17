#!/usr/bin/env python3
"""Plot final placement bias for the smoke diagnostic DHB action-head run."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 160,
    }
)


ROOT = Path("/home/andypark/Projects/repos/dhb-xr")
WORK = ROOT / "_workspace"
ASSETS = ROOT / "docs/paper/assets"


def main() -> None:
    with open(WORK / "dhb_head_e2e_fix_smoke_diag_sim.json") as f:
        sim = json.load(f)
    with open(WORK / "dhb_head_e2e_fix_smoke_diag_ik.json") as f:
        ik = json.load(f)

    runs = []
    for run in sim["runs"]:
        if run["attempt"] != "default":
            continue
        final = run["final_snapshot"]["ee_minus_plate"]
        runs.append(
            {
                "demo": run["demo_key"],
                "offset": run["offset_cm"],
                "success": run["success"],
                "y_mm": final[1] * 1000,
                "z_mm": final[2] * 1000,
            }
        )

    ik_means = [r["ik_mean_err_mm"] for r in ik["runs"]]

    demos = sorted({r["demo"] for r in runs})
    colors = {"demo_0": "#2563eb", "demo_1": "#dc2626", "demo_10": "#7c3aed"}
    markers = {0.0: "o", 3.0: "D"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.4, 1.0]})

    ax = axes[0]
    for r in runs:
        ax.scatter(
            r["y_mm"],
            r["z_mm"],
            s=130 if r["success"] else 95,
            marker=markers[r["offset"]],
            color=colors[r["demo"]],
            edgecolor="#111827" if r["success"] else "white",
            linewidth=1.2,
            alpha=0.95,
        )
        ax.text(r["y_mm"] + 0.8, r["z_mm"] + 0.3, f"{r['demo']} @ {int(r['offset'])}cm", fontsize=8)

    ax.axvline(-20.0, color="#94a3b8", linestyle="--", linewidth=1)
    ax.axvline(-45.0, color="#cbd5e1", linestyle=":", linewidth=1)
    ax.set_xlabel("Final EE minus plate in Y (mm)")
    ax.set_ylabel("Final EE minus plate in Z (mm)")
    ax.set_title("(a) Final placement bias clusters by demo", fontweight="bold")
    ax.grid(True, alpha=0.25)

    ax2 = axes[1]
    bars = ax2.bar(
        ["IK mean", "IK max-ish range"],
        [float(np.mean(ik_means)), float(np.max(ik_means) - np.min(ik_means))],
        color=["#059669", "#93c5fd"],
        width=0.6,
    )
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{bar.get_height():.3f}", ha="center", fontsize=9)
    ax2.set_ylabel("Error (mm)")
    ax2.set_title("(b) IK stays small and stable", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.text(
        0.5,
        0.94,
        "Failures are too large to be explained by IK",
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#475569",
    )

    fig.suptitle("Smoke Diagnostic: Placement Error Is Structured, Not Random", fontsize=14, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Successful demo_0 runs finish closer to the plate center in Y. The harder demos cluster further behind the target with a smaller positive Z bias, while IK remains near 0.47 mm.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    plt.tight_layout()

    out = ASSETS / "action_head_placement_diagnostics.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
