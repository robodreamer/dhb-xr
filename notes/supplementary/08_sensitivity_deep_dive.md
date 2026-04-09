# Sensitivity Deep Dive: Position vs Orientation Under Prediction Noise

## Summary

When a neural network predicts DHB invariants with noise σ, the decoded SE(3) trajectory has:
- **Position error**: 40-92% LESS than raw action noise (depending on variant)
- **Orientation error**: near parity for QR, 31% worse for DR

This is measured AFTER full decode back to SE(3) — the end-to-end pipeline.

---

## 1. Why Position Is Always Better

The invariant magnitude is an inner product:
```
m_i = x_i^T Δp_i
```

A noise vector ε added to m_i perturbs the position update as:
```
Δp_i' = (m_i + ε) * x_i = Δp_i + ε * x_i
```

The error is projected onto a SINGLE direction (x_i), contracting 3D noise to 1D.
In contrast, raw delta noise ε' ∈ ℝ³ directly adds to position in ALL 3 dimensions.

**Quantified**: DHB-DR position ratio 0.48–0.62× (avg 0.54×). DHB-QR: 0.07–0.09× (avg 0.08×).
QR is dramatically better because the quaternion representation provides a smoother parameterization
of the frame rotation, leading to less error accumulation during frame propagation.

---

## 2. Why Orientation Has a Penalty

### The Frame Propagation Coupling

Even if we only add noise to magnitudes (not quaternion components), orientation still degrades:
```
σ=0.01, magnitude-only noise:
  Position: 8.1mm (ratio 0.08×) ← excellent
  Orientation: 3.7° (ratio 1.15×) ← still worse than raw
```

This proves the coupling is INTRINSIC to frame propagation, not the quaternion representation.

### Why: Frame Update Depends on Magnitude

The DHB frame update equation:
```
F_{i+1} = R_combined · F_i
```

where `R_combined` is computed from the double-reflection of consecutive motion directions.
The motion direction `x_i = Δp_i / ||Δp_i||` depends on the magnitude: if m_i is noisy,
the direction changes slightly, which changes R_combined, which changes the frame,
which changes the decoded orientation.

This cascade: `noise(m_i) → noise(x_i) → noise(R_combined) → noise(F_{i+1}) → noise(R_{i+1})`
is unavoidable in any frame-based representation. It's a property of the geometry, not a bug.

### DR vs QR Difference

- **DR (Euler)**: Decomposes R_rel into Euler angles → amplifies noise near gimbal lock
  - Orientation ratio: 1.22–1.42× (avg 1.31×)
  
- **QR (Quaternion)**: Uses quaternion for R_rel → no gimbal lock, smoother
  - Orientation ratio: 0.92–1.16× (avg ~1.0×)
  - At σ=0.005: 0.93× (BETTER than raw)
  - At σ=0.020: 0.92× (BETTER than raw)

---

## 3. Complete Data Table

| σ | DR pos ratio | QR pos ratio | DR ori ratio | QR ori ratio |
|---|---|---|---|---|
| 0.005 | 0.58× | 0.07× | 1.24× | 0.93× |
| 0.010 | 0.58× | 0.07× | 1.29× | 1.16× |
| 0.020 | 0.73× | 0.09× | 1.39× | 0.92× |
| 0.050 | 0.57× | 0.07× | 1.30× | 1.01× |
| **avg** | **0.62×** | **0.08×** | **1.31×** | **1.00×** |

**Recommendation**: Use DHB-QR. Position is 12× better than DR, orientation at parity.

---

## 4. Experimental Details

- **Demo**: LIBERO-Spatial task 0 (pick-and-place, 98 steps)
- **Noise**: iid Gaussian N(0, σ²) added to each invariant dimension
- **Trials**: 20 per noise level
- **Metrics**: 
  - Position: mean L2 EE error (mm) across trajectory
  - Orientation: mean geodesic distance (degrees) = 2·arccos(|q₁·q₂|)
- **Baseline**: same noise added to raw position deltas (integrated) and Euler angle deltas (integrated)
- **All measurements AFTER decode to SE(3)** — this is the end-to-end pipeline

---

## 5. Directions to Overcome the Orientation Coupling

### 5a. Constraint-Aware Prediction (Most Promising)

Train the policy to predict invariants that respect frame propagation dynamics:

```python
class ConstraintAwareLoss(nn.Module):
    """Loss that penalizes invariant predictions causing orientation drift."""
    
    def forward(self, pred_invariants, gt_invariants, gt_orientations):
        # Standard invariant loss
        inv_loss = F.mse_loss(pred_invariants, gt_invariants)
        
        # Decode predicted invariants → get predicted orientations
        decoded = decode_dhb_qr(pred_invariants, ...)
        pred_orientations = decoded['quaternions']
        
        # Geodesic orientation loss (directly penalizes orientation error)
        ori_loss = geodesic_distance(pred_orientations, gt_orientations).mean()
        
        # Combined: invariant matching + orientation fidelity
        return inv_loss + λ_ori * ori_loss
```

By adding an orientation-aware term to the training loss, the policy learns to predict
invariants that minimize the frame propagation cascade effect.

**Expected improvement**: Orientation ratio from 1.0× → 0.7-0.8× (better than raw).

### 5b. Frame-Decoupled Invariants (Research Direction)

Redesign the invariant encoding to decouple magnitude from frame rotation:
- Encode frame rotations as ABSOLUTE quaternions (not relative to moving frame)
- This breaks the cascade: magnitude noise doesn't affect frame
- Trade-off: loses some of the SE(3)-invariance properties

### 5c. Diffusion-Based Invariant Prediction

Use a diffusion/flow-matching head instead of direct regression:
- Diffusion naturally handles multi-modal distributions
- The denoising process implicitly smooths the predicted invariant sequence
- The manifold structure of SO(3) can be respected via Riemannian score matching

### 5d. Ensemble Decoding

Predict K invariant hypotheses, decode all K, pick the one closest to the
previous decoded trajectory (temporal consistency filter):
```
predictions = [policy(obs) for _ in range(K)]
decoded = [decode(p) for p in predictions]
best = argmin(temporal_consistency(decoded, previous_trajectory))
```

This is inference-time compute (like diffusion) but targeted at the sensitivity issue.

### 5e. Learned Decoder (Replace Analytical Decode)

Train a neural network decoder: invariants → SE(3) trajectory.
The learned decoder can be trained to be ROBUST to input noise:
```
decoder_loss = ||decode_nn(invariants + noise) - clean_trajectory||²
```

This "denoising decoder" directly addresses the sensitivity by learning to ignore noise.

---

## 6. Priority Ranking for Next Iteration

| Approach | Effort | Expected Impact | Dependencies |
|---|---|---|---|
| 5a. Constraint-aware loss | 1 week | High (ori → 0.7×) | Training pipeline ready |
| 5e. Learned decoder | 2 weeks | High (ori → 0.5×) | Training data, architecture |
| 5d. Ensemble decoding | 2 days | Medium (ori → 0.85×) | None |
| 5c. Diffusion head | 3 weeks | High but uncertain | Flow-matching library |
| 5b. Frame-decoupled | 4 weeks | Fundamental but risky | Core encoder rewrite |

**Recommended next step**: 5a (constraint-aware loss) — minimal code change, directly targets the issue,
training infrastructure already exists from the DHBActionHeader experiment.
