#!/usr/bin/env python3
"""Plot current SmolVLA + DHB action-head checkpoint status from summary JSONs."""

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


def load_json(name: str) -> dict:
    with open(WORK / name) as f:
        return json.load(f)


def main() -> None:
    runs = [
        ("Smoke (10 ep)", 0.9409537017345428, load_json("dhb_head_e2e_fix_smoke_diag_sim.json")),
        ("Full (50 ep)", 0.6991323006153106, load_json("dhb_head_e2e_fix_full_sweep_sim.json")),
        ("Longer (200 ep)", 0.06337, load_json("dhb_head_e2e_fix_200ep_sweep_sim.json")),
    ]
    offsets = ["0.0cm", "3.0cm"]
    offset_labels = ["0 cm", "+3 cm"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1.0]})

    x = np.arange(len(runs))
    width = 0.34
    colors = ["#2563eb", "#059669"]

    ax = axes[0]
    for i, (offset_key, offset_label) in enumerate(zip(offsets, offset_labels)):
        vals = [run[2]["summary"][offset_key]["rate"] for run in runs]
        bars = ax.bar(x + (i - 0.5) * width, vals, width=width, color=colors[i], label=offset_label)
        for bar, run in zip(bars, runs):
            summary = run[2]["summary"][offset_key]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{summary['successes']}/{summary['total']}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in runs])
    ax.set_ylabel("Closed-loop success rate (%)")
    ax.set_ylim(0, 80)
    ax.set_title("(a) Executable checkpoint comparison", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    ax2 = axes[1]
    losses = [r[1] for r in runs]
    ax2.plot(x, losses, "o-", color="#7c3aed", linewidth=2.5, markersize=8)
    for xi, loss in zip(x, losses):
        ax2.text(xi, loss * 1.08, f"{loss:.3f}" if loss >= 0.1 else f"{loss:.4f}", ha="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r[0].split()[0] for r in runs])
    ax2.set_ylabel("Best supervised loss")
    ax2.set_yscale("log")
    ax2.set_title("(b) Lower train loss did not improve robustness", fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.25, which="both")

    fig.suptitle("Current SmolVLA + DHB Action Head Status", fontsize=14, fontweight="bold", y=1.03)
    fig.text(
        0.5,
        -0.02,
        "Smoke remains the best executable baseline. Training fit improves sharply from 10 to 200 epochs, but closed-loop success stays flat on the harder demos.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    plt.tight_layout()

    out = ASSETS / "action_head_checkpoint_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
