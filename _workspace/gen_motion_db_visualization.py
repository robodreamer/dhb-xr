#!/usr/bin/env python3
"""
Motion database visualization for DHB-XR project page.

Panel 1: Retrieval accuracy vs offset — DHB invariant (flat) vs position-based (degrading)
Panel 2: Self-retrieval under random SE(3) — bar chart showing DHB 100% vs Position 11%
"""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.dpi': 150,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#333333',
    'axes.linewidth': 0.8,
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#222222',
})

# Load data
with open('/home/andypark/Projects/repos/dhb_xr/_workspace/libero_motion_db/retrieval_contrast.json') as f:
    data = json.load(f)

offsets = [0, 3, 5, 10]
offset_labels = ['0', '3', '5', '10']

# DHB retrieval (flat)
dhb_top1 = [data['cross_task_retrieval']['dhb'][f'{o}cm']['top1'] * 100 for o in offsets]
dhb_top5 = [data['cross_task_retrieval']['dhb'][f'{o}cm']['top5'] * 100 for o in offsets]

# Position-based retrieval (degrading)
pos_top1 = [data['cross_task_retrieval']['position'][f'{o}cm']['top1'] * 100 for o in offsets]
pos_top5 = [data['cross_task_retrieval']['position'][f'{o}cm']['top5'] * 100 for o in offsets]

# Self-retrieval
dhb_self = data['self_retrieval']['dhb_self_retrieval'] * 100
pos_self = data['self_retrieval']['position_self_retrieval'] * 100

# Colors
C_DHB_1 = '#2563eb'  # blue
C_DHB_5 = '#60a5fa'  # light blue
C_POS_1 = '#dc2626'  # red
C_POS_5 = '#f97316'  # orange
C_DHB_BAR = '#2563eb'
C_POS_BAR = '#dc2626'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                gridspec_kw={'width_ratios': [2.2, 1]})

# ============================================================
# Panel 1: Retrieval accuracy vs offset
# ============================================================

# Shaded region between position top-5 and DHB top-1 to show the gap
ax1.fill_between(offsets, pos_top5, dhb_top1, alpha=0.06, color='#888888')

# Position-based lines (degrading)
ax1.plot(offsets, pos_top1, 'v--', color=C_POS_1, linewidth=2, markersize=9,
         label='Position top-1', zorder=4, markeredgecolor='white', markeredgewidth=0.8)
ax1.plot(offsets, pos_top5, '^--', color=C_POS_5, linewidth=2, markersize=9,
         label='Position top-5', zorder=4, markeredgecolor='white', markeredgewidth=0.8)

# DHB invariant lines (flat)
ax1.plot(offsets, dhb_top1, 'o-', color=C_DHB_1, linewidth=2.5, markersize=10,
         label='DHB top-1', zorder=5, markeredgecolor='white', markeredgewidth=1)
ax1.plot(offsets, dhb_top5, 's-', color=C_DHB_5, linewidth=2.5, markersize=10,
         label='DHB top-5', zorder=5, markeredgecolor='white', markeredgewidth=1)

# Annotate the crossover region
# At 10cm, position top-1 (60.4%) crosses below position top-5 at 5cm
# Find where DHB top-1 matches or exceeds position top-1
# DHB top-1 is 19.8, position top-1 goes 99.6 -> 60.4, so no crossover in accuracy
# But we can highlight the degradation
ax1.annotate('Position degrades\n39% drop at 10cm',
             xy=(10, pos_top1[-1]), xytext=(8, 38),
             fontsize=9, color=C_POS_1, fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color=C_POS_1, lw=1.2),
             ha='center')

ax1.annotate('DHB: perfectly flat\n(offset-invariant)',
             xy=(6, dhb_top5[2]), xytext=(6.5, 63),
             fontsize=9, color=C_DHB_1, fontstyle='italic',
             arrowprops=dict(arrowstyle='->', color=C_DHB_1, lw=1.2),
             ha='center')

# Data value labels on key points
for i, o in enumerate(offsets):
    if o in (0, 10):
        ax1.annotate(f'{pos_top1[i]:.1f}%', (o, pos_top1[i]),
                     textcoords="offset points", xytext=(-12, -14),
                     fontsize=8, color=C_POS_1, fontweight='bold')
        ax1.annotate(f'{pos_top5[i]:.1f}%', (o, pos_top5[i]),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=8, color=C_POS_5, fontweight='bold')

# Label DHB values once
ax1.annotate(f'{dhb_top1[0]:.1f}%', (0.3, dhb_top1[0]),
             textcoords="offset points", xytext=(8, -14),
             fontsize=8, color=C_DHB_1, fontweight='bold')
ax1.annotate(f'{dhb_top5[0]:.1f}%', (0.3, dhb_top5[0]),
             textcoords="offset points", xytext=(8, 6),
             fontsize=8, color=C_DHB_5, fontweight='bold')

ax1.set_xlabel('Workspace Offset (cm)')
ax1.set_ylabel('Same-Task Retrieval Accuracy (%)')
ax1.set_title('Retrieval Accuracy vs. Workspace Offset')
ax1.set_xlim(-0.5, 11)
ax1.set_ylim(0, 105)
ax1.set_xticks(offsets)
ax1.set_xticklabels(offset_labels)
ax1.legend(fontsize=9, loc='center left', framealpha=0.9, edgecolor='#cccccc')
ax1.grid(True, alpha=0.2, linestyle='-')

# Subtle vertical guides at each offset
for o in offsets:
    ax1.axvline(o, color='#dddddd', linewidth=0.5, zorder=0)

# ============================================================
# Panel 2: Self-retrieval bar chart
# ============================================================

bars = ax2.bar(['DHB\nInvariant', 'Position\nBased'], [dhb_self, pos_self],
               color=[C_DHB_BAR, C_POS_BAR], width=0.55, edgecolor='white',
               linewidth=1.5, zorder=3)

# Value labels on bars
ax2.text(0, dhb_self + 2, f'{dhb_self:.0f}%', ha='center', va='bottom',
         fontsize=14, fontweight='bold', color=C_DHB_BAR)
ax2.text(1, pos_self + 2, f'{pos_self:.1f}%', ha='center', va='bottom',
         fontsize=14, fontweight='bold', color=C_POS_BAR)

# Gap annotation — red arrow showing the difference
ax2.annotate('', xy=(0.5, dhb_self - 2), xytext=(0.5, pos_self + 5),
             arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.5))
ax2.text(0.5, (dhb_self + pos_self) / 2, f'{dhb_self - pos_self:.1f}%\ngap',
         ha='center', va='center', fontsize=9, color='#555555',
         fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#cccccc'))

ax2.set_ylabel('Self-Retrieval Accuracy (%)')
ax2.set_title('Self-Retrieval Under\nRandom SE(3) Transform')
ax2.set_ylim(0, 115)
ax2.grid(True, alpha=0.2, axis='y', linestyle='-')

# Subtitle with experiment details
ax2.text(0.5, -0.12, f'n={data["self_retrieval"]["total_queries"]} queries '
         f'({data["self_retrieval"]["n_transforms"]} transforms '
         f'x {data["self_retrieval"]["n_samples"]} samples)',
         transform=ax2.transAxes, ha='center', fontsize=8, color='#888888')

plt.tight_layout(w_pad=3)

out_path = '/home/andypark/Projects/repos/dhb_xr/docs/paper/assets/motion_db_visualization.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.close()
