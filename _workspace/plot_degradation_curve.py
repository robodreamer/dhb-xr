#!/usr/bin/env python3
"""
Plot SmolVLA degradation curve from eval results.

Usage:
    python _workspace/plot_degradation_curve.py \
        --input _workspace/degradation_curve_results.json \
        --output _workspace/degradation_curve.png
"""
import json, argparse
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="_workspace/degradation_curve_results.json")
    parser.add_argument("--output", default="_workspace/degradation_curve.png")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    meta = data["metadata"]
    summary = data["summary"]

    offsets = []
    rates = []
    ci_low = []
    ci_high = []

    for offset_cm in meta["offsets_cm"]:
        s = summary[f"{offset_cm}cm"]
        n = s["total"]
        k = s["successes"]
        p = s["rate"]
        offsets.append(offset_cm)
        rates.append(p * 100)

        # Wilson score 95% confidence interval
        z = 1.96
        denom = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
        ci_low.append(max(0, center - margin) * 100)
        ci_high.append(min(1, center + margin) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot with confidence band
    ax.fill_between(offsets, ci_low, ci_high, alpha=0.2, color='#E74C3C')
    ax.plot(offsets, rates, 'o-', color='#E74C3C', linewidth=2.5, markersize=8,
            label=f'SmolVLA (500M)', zorder=5)

    # Annotate each point
    for x, y, n_succ, n_tot in zip(offsets, rates,
                                     [summary[f"{o}cm"]["successes"] for o in meta["offsets_cm"]],
                                     [summary[f"{o}cm"]["total"] for o in meta["offsets_cm"]]):
        ax.annotate(f'{n_succ}/{n_tot}', (x, y), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9, color='#555')

    ax.set_xlabel(f'Workspace Offset ({meta["axis"]}-axis, cm)')
    ax.set_ylabel('Task Success Rate (%)')
    ax.set_title(f'SmolVLA Spatial Degradation\n'
                 f'LIBERO-Spatial Task {meta["task_id"]}: {meta["task_name"][:50]}')

    ax.set_ylim(-5, 105)
    ax.set_xlim(min(offsets) - 0.5, max(offsets) + 0.5)
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='50% threshold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Add subtitle with experiment details
    ax.text(0.02, 0.02, f'{meta["seeds_per_offset"]} trials per offset | {meta["timestamp"]}',
            transform=ax.transAxes, fontsize=8, color='gray', va='bottom')

    plt.tight_layout()
    plt.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"Saved: {args.output}")

    # Also save high-res for project page
    page_path = args.output.replace('.png', '_hires.png')
    plt.savefig(page_path, dpi=300, bbox_inches='tight')
    print(f"Saved high-res: {page_path}")
    plt.close()


if __name__ == "__main__":
    main()
