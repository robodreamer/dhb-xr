# Next Iteration Plan — Prioritized Improvement Roadmap

## Current Strengths (no improvement needed)

| Area | Result | Status |
|------|--------|--------|
| Geometric adaptation | 0.0mm at 0–20cm, 10 tasks × 8 dirs | PERFECT |
| Token stability | 0.0% change (magnitudes) | PERFECT |
| Position sensitivity | 0.08× (QR) | EXCELLENT |
| Orientation sensitivity | 0.96× (QR + geodesic) | GOOD (< 1.0) |
| Learnability | inv MSE 0.022, act L1 0.14 | GOOD |

## Priority 1: Paper-Blocking (must do before submission)

### 1a. SE(3) Training Augmentation (1 day)
**What**: During training, apply random SE(3) transforms to input EE states.
DHB invariant targets stay IDENTICAL (they're invariant!) → free data augmentation.

**Why**: This is the single most elegant argument for the paper: "because targets are
invariant, SE(3) augmentation comes for free — no label recomputation needed."
No other action representation has this property.

**Implementation**:
```python
def augment_se3(ee_state, invariants):
    """Augmentation that's FREE for invariant representations."""
    R = random_rotation()
    t = random_translation()
    ee_augmented = apply_se3(ee_state, R, t)  # observations change
    inv_augmented = invariants  # targets DON'T change!
    return ee_augmented, inv_augmented
```

**Expected impact**: Better generalization, compelling paper argument.

### 1b. Waypoint PD + Object Perturbation (3 days)
**What**: Patch LIBERO `init_state` to move object positions, run waypoint PD controller.

**Why**: Gets real task success rates for Table 1 — the single biggest gap from reviewer.

**Implementation**: 
1. Find object qpos indices in LIBERO state vector
2. Offset them by perturbation amount
3. Run waypoint PD controller with DHB-adapted trajectory
4. Measure reward > 0

### 1c. Multi-Task Training (2 days)
**What**: Train one model on all 10 LIBERO-Spatial tasks (not per-task).

**Why**: Shows the invariant representation generalizes across task structure,
not just spatial configuration.

## Priority 2: Strengthen Existing Results

### 2a. Differentiable Full Decoder (1 week)
**What**: Fix NaN in `dhb_qr_torch.py` via numerically stable `rot_to_quat`.

**Why**: Enables end-to-end training through decode → expected ori ratio 0.7×.

**Technical fix**:
```python
def rot_to_quat_stable(R):
    """Shepperd's method with 4-branch selection for numerical stability."""
    diag = torch.diagonal(R, dim1=-2, dim2=-1)  # (..., 3)
    trace = diag.sum(-1)
    
    # Branch on which diagonal element is largest
    # This avoids division by small numbers
    s0 = torch.sqrt(torch.clamp(trace + 1, min=1e-8)) * 2
    s1 = torch.sqrt(torch.clamp(1 + R[...,0,0] - R[...,1,1] - R[...,2,2], min=1e-8)) * 2
    s2 = torch.sqrt(torch.clamp(1 - R[...,0,0] + R[...,1,1] - R[...,2,2], min=1e-8)) * 2
    s3 = torch.sqrt(torch.clamp(1 - R[...,0,0] - R[...,1,1] + R[...,2,2], min=1e-8)) * 2
    
    # Select branch based on largest diagonal
    # Use torch.where for differentiable selection
    ...
```

### 2b. Hybrid FAST+DHB Compression (3 days)
**What**: Apply FAST (DCT+BPE) to DHB invariant sequences instead of raw actions.

**Why**: Gets FAST's 27.8× compression on invariant signals that are ALSO SE(3)-stable.
Best of both worlds: high compression + geometric invariance.

**Expected**: FAST compression ratio on invariants may be even HIGHER than on raw actions,
because invariant magnitudes are smoother (low-frequency dominated).

### 2c. OAT on Real LIBERO Data (2 days)
**What**: Train OAT tokenizer on real LIBERO invariant sequences (not synthetic figure-8).

**Why**: Addresses reviewer critique C4 (synthetic data only).
Prefix decoding quality on real data is the missing evidence.

## Priority 3: Stretch Goals

### 3a. ResNet Visual Backbone (1 week)
Replace VisProxy MLP with ResNet-18 on LIBERO agentview_rgb images.
Full vision → invariant → adaptation pipeline.

### 3b. EquivAct/RISE Comparison (2 weeks)
Run EquivAct on same LIBERO-Spatial tasks. Compare spatial generalization
under the same perturbation protocol.

### 3c. Contact-Aware Invariants (2 weeks)
Segment trajectories by contact phase. Encode each segment independently.
Test on peg-insertion or assembly tasks.

## Recommended Execution Order

```
Day 1:    1a (SE3 augmentation — 1 day, compelling argument)
Day 2-4:  1b (Object perturbation — 3 days, fills Table 1)
Day 5-6:  1c (Multi-task — 2 days, stronger generalization claim)
Day 7-8:  2c (OAT on real data — 2 days, addresses C4)
Day 9-11: 2b (FAST+DHB hybrid — 3 days, best-of-both-worlds)
Week 3:   2a (Differentiable decoder — 1 week, ori → 0.7×)
```

Total: ~2.5 weeks to submission-ready paper with comprehensive experiments.
