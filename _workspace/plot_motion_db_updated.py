#!/usr/bin/env python3
"""
Plot updated motion DB retrieval comparison from quick_motion_db_results.json.

Creates:
  1. Within-task retrieval at 0cm (bar chart, all 5 methods)
  2. Self-retrieval under SE(3) offset (line chart)
  3. Within-task retrieval vs offset (line chart)

Usage:
    pixi run python _workspace/plot_motion_db_updated.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

results_path = "_workspace/quick_motion_db_results.json"
with open(results_path) as f:
    data = json.load(f)

# ── Colors ────────────────────
colors = {
    "dhb_dr_full": "#93c5fd",   # light blue
    "dhb_dr_mag":  "#2563eb",   # blue
    "dhb_ti_full": "#c4b5fd",   # light purple
    "dhb_ti_mag":  "#7c3aed",   # purple
    "position":    "#f87171",   # red
}
labels = {
    "dhb_dr_full": "DHB-DR (8D)",
    "dhb_dr_mag":  "DHB-DR mag (2D)",
    "dhb_ti_full": "DHB-TI (8D)",
    "dhb_ti_mag":  "DHB-TI mag (2D)",
    "position":    "Position-based",
}
methods = ["dhb_dr_full", "dhb_dr_mag", "dhb_ti_full", "dhb_ti_mag", "position"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ── Panel 1: Within-task retrieval at 0cm ──
ax = axes[0]
l2 = data["within_task_l2_0cm"]
x = np.arange(len(methods))
top1 = [l2[m].get("top1", 0) * 100 for m in methods]
top5 = [l2[m].get("top5", 0) * 100 for m in methods]
w = 0.35
bars1 = ax.bar(x - w/2, top1, w, label="Top-1", color=[colors[m] for m in methods], edgecolor="white")
bars2 = ax.bar(x + w/2, top5, w, label="Top-5", color=[colors[m] for m in methods], alpha=0.5, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([labels[m] for m in methods], rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Same-task retrieval (%)")
ax.set_title("Within-task Retrieval @ 0 cm", fontweight="bold")
ax.set_ylim(0, 105)
ax.legend(fontsize=8)
for i, v in enumerate(top1):
    ax.text(i - w/2, v + 1, f"{v:.0f}", ha="center", fontsize=7, fontweight="bold")

# ── Panel 2: Self-retrieval under SE(3) offset ──
ax = axes[1]
sr = data["self_retrieval_under_offset"]
offsets = [0, 3, 5, 10]
for m in methods:
    vals = [sr[m].get(f"{o}cm", 0) * 100 for o in offsets]
    ax.plot(offsets, vals, "o-", color=colors[m], label=labels[m], linewidth=2, markersize=6)
ax.set_xlabel("Workspace offset (cm)")
ax.set_ylabel("Self-retrieval accuracy (%)")
ax.set_title("Self-Retrieval Under SE(3) Offset", fontweight="bold")
ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc="lower left")
ax.axhline(y=100, color="#e5e7eb", linestyle="--", linewidth=0.5)

# ── Panel 3: Within-task retrieval at offsets ──
ax = axes[2]
co = data["within_task_at_offsets"]
for m in methods:
    vals = [co[m].get(f"{o}cm", {}).get("top1", 0) * 100 for o in offsets]
    ax.plot(offsets, vals, "o-", color=colors[m], label=labels[m], linewidth=2, markersize=6)
ax.set_xlabel("Workspace offset (cm)")
ax.set_ylabel("Same-task top-1 accuracy (%)")
ax.set_title("Within-task Retrieval vs Offset", fontweight="bold")
ax.set_ylim(0, 105)
ax.legend(fontsize=7, loc="lower left")

plt.tight_layout()
out = "docs/paper/assets/motion_db_retrieval_comparison.png"
plt.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved: {out}")

# Also save a simpler 2-panel version focused on the key story
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: Self-retrieval (the strongest result)
for m in methods:
    vals = [sr[m].get(f"{o}cm", 0) * 100 for o in offsets]
    ls = "-" if "dhb" in m else "--"
    ax1.plot(offsets, vals, "o" + ls, color=colors[m], label=labels[m],
             linewidth=2.5 if "mag" in m or m == "position" else 1.5, markersize=7)
ax1.set_xlabel("Workspace offset (cm)", fontsize=11)
ax1.set_ylabel("Self-retrieval accuracy (%)", fontsize=11)
ax1.set_title("(a) Can you find the same trajectory\nafter workspace shift?", fontweight="bold")
ax1.set_ylim(50, 105)
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel B: Within-task retrieval at offsets
for m in ["dhb_dr_mag", "dhb_ti_mag", "position"]:
    vals = [co[m].get(f"{o}cm", {}).get("top1", 0) * 100 for o in offsets]
    ls = "-" if "dhb" in m else "--"
    ax2.plot(offsets, vals, "o" + ls, color=colors[m], label=labels[m],
             linewidth=2.5, markersize=7)
ax2.set_xlabel("Workspace offset (cm)", fontsize=11)
ax2.set_ylabel("Same-task top-1 accuracy (%)", fontsize=11)
ax2.set_title("(b) Same-task retrieval accuracy\nacross workspace offsets", fontweight="bold")
ax2.set_ylim(0, 105)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out2 = "docs/paper/assets/motion_db_results.png"
plt.savefig(out2, dpi=200, bbox_inches="tight")
print(f"Saved: {out2}")
