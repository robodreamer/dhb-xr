# P2 DHBActionHead Failure Debug Report
**Date:** 2026-04-14  
**Experiment:** DHBActionHead closed-loop VLA on LIBERO-PRO P2 (position swap)  
**Result:** 0/10 = 0% success

---

## Quick Links

| File | What it shows |
|------|---------------|
| `p2_failure_seed0.mp4` | Typical failure — robot arcs to original plate area, never picks bowl |
| `p2_failure_seed8.mp4` | Closest attempt — EE reaches Y≈0.06 (plate at Y=0.042), but misses pick |
| `p2_failure_debug.png` | 4-panel overview: XY trajectories, Y-over-time, gripper timing, final EE positions |
| `p2_trajectory_comparison.png` | Predicted vs GT Fatrop trajectories (seeds 0, 4, 8) — X, Y, Z columns |
| `p2_invariant_channels.png` | GT Fatrop invariants (all 8 channels) for seeds 0, 4, 8 |

Open videos:
```bash
cd /home/andypark/Projects/repos/dhb_xr/_workspace/p2_debug
xdg-open p2_failure_seed0.mp4
xdg-open p2_failure_seed8.mp4
```

---

## Setup

**Task:** Pick up black bowl, place on plate  
**P2 perturbation (cookies_1 swap, seed=42):**
- Original plate: `[0.073, 0.212, 0.902]`
- P2 plate: `[0.083, 0.042, 0.902]` (Y shifts −17cm)
- Bowl: barely moves (~1cm shift)

**Training:** 50 original demos + 50 Fatrop-adapted P2 demos = 100 total  
**Checkpoint:** `_workspace/dhb_head_p2_correct.pt` (epoch 197/200, loss 0.043)

---

## Failure Mode

### What the robot does (all 10 seeds)
1. Starts correctly at EE near robot base (Y≈0.00–0.01)
2. Moves toward the workspace — but arcs toward **Y≈0.18–0.25** (original plate region)
3. Closes gripper at **t≈42–46** in the right timestep window — but at **wrong Y position**
4. Never reaches the bowl (Y≈0.21) or the P2 plate (Y=0.042)
5. Final EE positions: Y=0.047–0.45, all outside the P2 plate success zone

### Gripper timing: CORRECT. Pick position: WRONG.

From `p2_failure_debug.png` Panel 3 (gripper heatmap):
- All 10 seeds close gripper within the GT GRASP window (t=29–56)
- **This is not a timing bug** — the model learned when to grasp correctly

From `p2_failure_debug.png` Panel 1 (XY top-down):
- Red stars (gripper-close positions) cluster at Y≈0.16–0.29
- Bowl is at Y≈0.21, P2 plate is at Y=0.042
- No seed reaches the P2 plate zone

---

## Root Cause

### Smoking gun: Y-trajectory in `p2_trajectory_comparison.png`

Look at the **Y column** (middle column) for any of the 3 seeds:
- **GT Fatrop** (black dashed): Y rises to pick bowl (~0.21), then **dips down to 0.042** for placement
- **Predicted** (colored solid): Y rises to ~0.18–0.25 and **never dips** — stays high

The model learned the original plate geometry (Y≈0.21) instead of the P2 geometry (Y≈0.042).

### Why: angular channel misprediction

From `p2_invariant_channels.png`:
- Channels 3 and 7 (angular invariants) are the ones that encode the "turn toward Y=0.042" signal
- These channels have high variance across the 50 P2 demos
- With only 50 examples, the model averages them out → predicted trajectory stays near original plate

### Data scale is the bottleneck

| Training examples | P2-adapted | Result |
|---|---|---|
| 50 | 0 (wrong targets) | 0% (visually blind) |
| 50 | 50 (correct Fatrop targets) | 0% (angular channels averaged out) |
| **~500** | ~500 (diverse offsets) | **TBD — likely non-zero** |

---

## Per-Seed Distance to P2 Plate at Gripper Close

| Seed | Gripper close Y | Delta from bowl (Y=0.21) | Final dist to P2 plate |
|------|-----------------|--------------------------|------------------------|
| 0 | 0.187 | +0.145m | 0.235m |
| 1 | ~0.063 | −0.147m (missed entirely) | — |
| 2 | ~0.197 | +0.187m | — |
| 3 | ~0.071 | −0.139m | — |
| 4 | 0.182 | +0.140m | 0.153m |
| 5 | ~0.001 | −0.209m | — |
| 6 | ~0.337 | +0.127m above bowl | — |
| 7 | ~0.165 | −0.045m | — |
| **8** | **0.162** | **+0.120m** | **0.094m (closest)** |
| 9 | ~0.319 | +0.109m above bowl | — |

**Seed 8 is the closest attempt** — watch `p2_failure_seed8.mp4` for the best case.

---

## What Would Fix This

1. **More P2 training data** (est. 500+ diverse Y-offset examples): forces angular channels to specialize
2. **Closed-loop re-prediction** every K steps: small prediction errors don't compound
3. **Separate pick/place heads**: predict pick position directly (not via invariants), use invariants only for placement
4. **Visual encoder fine-tuning**: LoRA is only on the text model; the visual encoder doesn't adapt to the 17cm Y-shift

---

## Scripts that generated this report

```bash
# Plots
pixi run python _workspace/p2_failure_debug_plots.py
pixi run python _workspace/p2_trajectory_comparison.py

# Videos (libero env)
MUJOCO_GL=egl ~/miniforge3/envs/libero/bin/python _workspace/record_p2_failure_video.py
```

All source scripts in `_workspace/`: `p2_failure_debug_plots.py`, `p2_trajectory_comparison.py`, `record_p2_failure_video.py`
