#!/usr/bin/env python3
"""Plot 3-way augmentation comparison: Original vs DHB vs Naive."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'DejaVu Sans'],
    'font.size': 12,
    'figure.dpi': 150,
})

offsets = [0, 3, 5, 7]

# Load results
base = json.load(open('_workspace/degradation_curve_results.json'))
orig = json.load(open('_workspace/eval_original_finetuned.json'))
dhb = json.load(open('_workspace/eval_dhb_finetuned.json'))
naive = json.load(open('_workspace/eval_naive_finetuned.json'))

def get_rates(data, offsets):
    rates = []
    for o in offsets:
        key = f"{o}cm" if f"{o}cm" in data['summary'] else f"{float(o)}cm"
        rates.append(data['summary'][key]['rate'] * 100)
    return rates

base_rates = [70, 55, 10, 0]  # from 20-seed degradation curve (0,3,5,7)
orig_rates = get_rates(orig, offsets)
dhb_rates = get_rates(dhb, offsets)
naive_rates = get_rates(naive, offsets)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(offsets, base_rates, 'o--', color='#6b7280', linewidth=1.5, markersize=7,
        label='Base SmolVLA (n=20)', alpha=0.7)
ax.plot(offsets, orig_rates, 's-', color='#2563eb', linewidth=2.5, markersize=9,
        label='Original FT (50 eps, 0cm)')
ax.plot(offsets, dhb_rates, 'D-', color='#059669', linewidth=2.5, markersize=9,
        label='DHB-aug FT (500 eps)')
ax.plot(offsets, naive_rates, '^-', color='#dc2626', linewidth=2.5, markersize=9,
        label='Naive-aug FT (500 eps)')

# Annotate key point
ax.annotate('+10%', xy=(3, 30), xytext=(3.5, 42),
            fontsize=11, fontweight='bold', color='#059669',
            arrowprops=dict(arrowstyle='->', color='#059669'))

ax.set_xlabel('Workspace Offset (Y-axis, cm)')
ax.set_ylabel('Task Success Rate (%)')
ax.set_title('SmolVLA + DHB Augmentation\nLIBERO-Spatial Task 0 (n=10, 5K LoRA steps)')
ax.set_ylim(-5, 95)
ax.set_xlim(-0.5, 7.5)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('docs/paper/assets/augmentation_comparison.png', dpi=200, bbox_inches='tight')
plt.savefig('_workspace/augmentation_comparison_results.png', dpi=150, bbox_inches='tight')
print("Saved: docs/paper/assets/augmentation_comparison.png")
plt.close()
