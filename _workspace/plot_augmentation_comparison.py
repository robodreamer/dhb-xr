#!/usr/bin/env python3
"""Plot preliminary augmentation comparison from the n=50 degradation JSONs."""

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


def get_rates(data: dict, offsets_cm: list[float]) -> list[float]:
    out = []
    for offset in offsets_cm:
        key = f"{offset:.1f}cm"
        out.append(data["summary"][key]["rate"] * 100.0)
    return out


def main() -> None:
    offsets = [0.0, 3.0, 5.0]
    base = load_json("degradation_base_n50.json")
    orig = load_json("degradation_orig_n50.json")
    dhb = load_json("degradation_dhb_n50.json")
    naive = load_json("degradation_naive_n50.json")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))

    series = [
        ("Base SmolVLA", get_rates(base, offsets), "#6b7280", "o--"),
        ("Original FT", get_rates(orig, offsets), "#2563eb", "s-"),
        ("DHB-aug FT", get_rates(dhb, offsets), "#059669", "D-"),
        ("Naive-aug FT", get_rates(naive, offsets), "#dc2626", "^-"),
    ]
    for label, rates, color, style in series:
        ax.plot(offsets, rates, style, color=color, linewidth=2.4, markersize=8, label=label)

    # Call out the main boundary result at +3 cm.
    dhb_3 = get_rates(dhb, offsets)[1]
    naive_3 = get_rates(naive, offsets)[1]
    ax.annotate(
        f"{dhb_3:.0f}% vs {naive_3:.0f}%",
        xy=(3.0, dhb_3),
        xytext=(3.55, max(dhb_3, naive_3) + 13),
        fontsize=10,
        fontweight="bold",
        color="#059669",
        arrowprops=dict(arrowstyle="->", color="#059669", linewidth=1.5),
    )

    ax.set_xlabel("Workspace offset (Y-axis, cm)")
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("Preliminary Augmentation Comparison (n=50)", fontweight="bold")
    ax.set_ylim(-5, 100)
    ax.set_xlim(-0.3, 5.5)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(
        0.02,
        0.02,
        "Use this as directional evidence only: geometry-aware augmentation helps at the boundary,\n"
        "but this is weaker than the main invariant-tokenization and open-loop adaptation results.",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#475569",
        va="bottom",
    )

    plt.tight_layout()
    out = ASSETS / "augmentation_comparison.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
