# Next Iteration Plan — Prioritized Improvement Roadmap

## Current Strengths (updated 2026-04-09)

| Area | Result | Status |
|------|--------|--------|
| Geometric adaptation | 0.0mm at 0-20cm, 10 tasks x 8 dirs | PERFECT |
| Token stability | 0.0% change (magnitudes) | PERFECT |
| Position sensitivity | 0.08x (QR) | EXCELLENT |
| Orientation sensitivity | 1.12x (intrinsic to decoder) | UNDERSTOOD |
| Learnability | inv MSE 0.023, 10-task multi-task | GOOD |
| FAST+DHB compression | 49.8x on magnitudes, 0% instability | EXCELLENT |
| OAT on magnitudes | k=1 MSE=0.00005, 0% instability | EXCELLENT |
| Blended controller | 100% at 0-5cm | GOOD |
| SE(3) augmentation | 2% improvement with EE proxy | VALIDATED |

## Completed Steps — Status Summary

### Step 1: SE(3) Training Augmentation -- DONE
**Result**: 2% eval MSE improvement with EE proxy (0.2738 -> 0.2696).
Augmented model has higher train loss (0.1607 vs 0.1072) but better generalization.
Theory validated: invariant targets unchanged under SE(3) transforms = free augmentation.

**File**: `results/exp_se3_augmentation.json`

### Step 2: Blended Controller Simulation -- DONE
**Result**: 100% success at 0, 3, and 5cm offsets (9 trials each).
Root cause of prior failure: wrong init_state source. Fix: use `robot0_eef_pos` at runtime.
Naive drops to 89% at 3cm; blended maintains 100%.

**File**: `results/exp_blended_success.json`

### Step 3: Multi-Task Training -- DONE
**Result**: 0.023 avg MSE across 10 LIBERO-Spatial tasks with single model.
301 training samples, 200 epochs, SE(3) augmentation enabled.
Per-task MSE ranges from 0.011 (task 3) to 0.044 (task 1).

**File**: `results/exp_multi_task.json`

### Step 4: OAT on Real LIBERO Data -- DONE
**Result (full 10-dim invariants)**: k=1 MSE=0.0054, k=8 MSE=0.0052. Spread=1.03.
**Result (2-dim magnitudes)**: k=1 MSE=0.00005, k=8 MSE=0.00005. Spread=0.91.
Magnitude-only signal is trivially compressible -- 1 token captures it entirely.

**Files**: `results/exp_oat_real_improved.json`, `results/exp_magnitude_oat.json`

### Step 5: FAST+DHB Invariant Subset Compression -- DONE
**Result**: 49.8x compression on 2-col magnitudes, 0% SE(3) instability.
5-col invariant subset: 4.0x compression, also 0% instability.
Raw actions for comparison: 20.4x compression, 93.9% instability.

**File**: `results/exp_invariant_subset_compression.json`

### Step 6: Geodesic Strategy Sweep / NaN Fix -- DONE
**Result**: All 5 geodesic strategies produce identical 1.121 orientation ratio.
NaN in rot_to_quat fixed with Shepperd's method.
Key insight: orientation sensitivity is intrinsic to the decoder's frame propagation,
not a training loss property. Magnitude-only tokenization sidesteps it entirely.

**Files**: `results/exp_geodesic_strategies.json`, `results/exp_full_decode_geodesic.json`

---

## Remaining Work Before Submission

### Priority 1: Paper-Blocking

| Task | Effort | Notes |
|------|--------|-------|
| StarVLA-DHB policy training | 1-2 weeks | Requires LIBERO->LeRobot adapter + GPU |
| Larger offset simulation (10-20cm) | 3 days | Requires full Fatrop + waypoint PD |
| Write final paper sections | 1 week | Integrate all new results into manuscript |

### Priority 2: Strengthen Claims

| Task | Effort | Notes |
|------|--------|-------|
| ResNet visual backbone | 1 week | Replace EE proxy with real vision |
| EquivAct/RISE comparison | 2 weeks | Same protocol, direct comparison |
| Differentiable full decoder | 1 week | End-to-end training through decode |

### Priority 3: Stretch Goals

| Task | Effort | Notes |
|------|--------|-------|
| RoboCasa generalization | 2 weeks | Requires AllenAI harness |
| Cross-embodiment (dual-arm) | 2 weeks | embodik dual-iiwa + DHB |
| Contact-aware invariants | 2 weeks | Phase-segmented encoding |

## Recommended Execution Order

```
Week 1:   StarVLA-DHB training setup + begin training
Week 1:   Write updated experiment sections with new results
Week 2:   Larger offset simulation (Fatrop + PD)
Week 2:   ResNet visual backbone
Week 3:   StarVLA-DHB evaluation + final paper polish
```

Total: ~3 weeks to submission-ready paper.
