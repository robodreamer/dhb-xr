# Extended Methods — DHB-XR CoRL 2026

Full technical derivations and implementation details. Supplement to manuscript/methods.tex.
No page limit. Internal reference for authors, reviewers, and journal extension.

---

## Table of Contents

1. [DHB-DR: Double-Reflection Frame Transport](#1-dhb-dr-double-reflection-frame-transport)
2. [DHB-QR: Quaternion Rotation Variant](#2-dhb-qr-quaternion-rotation-variant)
3. [DHB-TI: Time-Invariant Arc-Length Reparameterization](#3-dhb-ti-time-invariant-arc-length-reparameterization)
4. [OAT Tokenizer Architecture](#4-oat-tokenizer-architecture)
5. [Fatrop NLP Formulation](#5-fatrop-nlp-formulation)
6. [SE(3)-Invariance Proof Sketch](#6-se3-invariance-proof-sketch)

---

## 1. DHB-DR: Double-Reflection Frame Transport

### 1.1 Overview

DHB-DR (Double-reflection variant) is the primary encoder. It transports a moving
frame along the trajectory using double Householder reflections — a method adapted
from the rotation-minimizing frame (RMF / Bishop frame) literature — and encodes
motion as a 4-tuple per step: one scalar magnitude and three Euler angles.

### 1.2 Notation

Let the input trajectory be:
- `p_0, p_1, ..., p_{N-1}` — end-effector positions in R^3
- `q_0, q_1, ..., q_{N-1}` — unit quaternions (wxyz convention) for orientation

Define position differences:
```
Δp_i = p_{i+1} - p_i,   i = 0, ..., N-2
```

And the moving-frame x-axis (tangent direction) at step i:
```
x_i = Δp_i / ||Δp_i||    (when ||Δp_i|| > ε)
```

The moving frame at each step is a right-handed orthonormal frame `F_i = [x_i | y_i | z_i]`
where `z_i = x_i × y_i`.

### 1.3 Double-Reflection Step

The key computation is transporting the y-axis from step i to step i+1 using two
Householder reflections (Bishop-frame-style, avoiding explicit integration of curvature):

**Step 1**: Compute the first reflection vector:
```
v1 = Δp_{i+1} - Δp_i
R1 = I - 2 * (v1 v1^T) / (v1^T v1)      [Householder about v1]
x_ref = R1 @ x_i                          [intermediate tangent]
```

**Step 2**: Compute the second reflection to align `x_ref` with `x_{i+1}`:
```
v2 = x_{i+1} - x_ref
R2 = I - 2 * (v2 v2^T) / (v2^T v2)      [Householder about v2]
```

**Combined transport rotation**:
```
R = R2 @ R1
y_{i+1} = R @ y_i                         [transported y-axis]
```

Numerical cleanup: after transport, re-orthogonalize `y_{i+1}` against `x_{i+1}`:
```
y_{i+1} ← y_{i+1} - (y_{i+1}^T x_{i+1}) x_{i+1}
y_{i+1} ← y_{i+1} / ||y_{i+1}||
```

If either `||v1||` or `||v2||` is below threshold ε, fall back to `R = I` (or RMF
projection in robust mode).

**Relative rotation**: between consecutive frames:
```
F_i = [x_i | y_i | z_i]
F_{i+1} = [x_{i+1} | y_{i+1} | z_{i+1}]
R_rel = F_i^T @ F_{i+1}
```

### 1.4 Invariant Encoding

Per step i, the 4-tuple invariant is:

**Magnitude (scalar)**:
```
m_i = x_i^T · Δp_i       [projection of motion onto local tangent]
```

**Relative frame rotation (3 Euler angles, XYZ extrinsic)**:
```
φ_i = EulerXYZ(F_i^T @ F_{i+1})
    = [φ_x, φ_y, φ_z]    [roll, pitch, yaw of frame change]
```

The orientation channel uses the same double-reflection procedure applied to the
rotation difference sequence (axis-angle representation of `R_{i+1} R_i^T`).

Full invariant per step:
```
I_i = [m_i, φ_x, φ_y, φ_z]  ∈ R^4   (linear component)
J_i = [ω_i, ψ_x, ψ_y, ψ_z]  ∈ R^4   (angular component)
```

where `ω_i` is the projected rotation magnitude and `ψ` is the Euler XYZ rotation
of the orientation frame transport.

### 1.5 Implementation: Padding Convention

The encoder pads the trajectory at both ends to handle boundary conditions:
```python
# Default mode (use_default_initial_frames=True):
positions = vstack([p[0], p[0], *positions])   # 2 leading copies
positions = vstack([*positions, p[-1], p[-1], p[-1]])  # 3 trailing copies
```

This ensures the first and last invariant steps have well-defined frame contexts.
The resulting invariant array has shape `(N-2, 4)` for an N-pose trajectory.

### 1.6 Decoding Procedure

Given invariants `{I_i}` and an initial pose `(p_0, q_0)`:

1. Initialize linear frame `L_0 = I_4` (4x4 homogeneous) with `L_0[:3, 3] = p_0`
2. Initialize angular rotation `A_0 = R(q_0)` (3x3 rotation matrix)
3. For each step i:

```
R_lin_i = Euler_to_Rot(φ_i)        [recover frame rotation]
T_i = [[R_lin_i, [m_i, 0, 0]^T],
        [0, 0, 0, 1]]              [4x4 homogeneous step]
L_{i+1} = L_i @ T_i               [propagate linear frame]
p_i = L_i[:3, 3]                  [extract position]

r_i = A_i @ [ω_i, 0, 0]^T         [rotation step in world frame]
A_{i+1} = A_i @ AxisAngle_to_Rot(r_i)^T  [propagate angular frame]
q_i = Rot_to_Quat(A_i)            [extract quaternion]
```

4. Strip 2 leading padded poses (`drop_padded=True`), returning N poses matching input.

### 1.7 Robust Mode

When `robust_mode=True`, additional protections are enabled:

- **180-degree reversal detection**: if `dot(Δp_i, Δp_{i+1}) < -0.9`, use RMF projection
  instead of double-reflection (prevents gimbal-like flip of y-axis)
- **Zero-motion handling**: if `||Δp_i|| < 1e-6`, preserve previous frame direction
- **Frame validation**: check orthogonality tolerance at each step, log failures
- **NaN/Inf guards** in decode: replace with previous valid pose

### 1.8 Key Results

| Property | Value |
|----------|-------|
| Invariant dimension | 4 per component (linear + angular) |
| Geometric endpoint error | 0.0 mm (all offsets 0-20 cm, n=480 trials) |
| Solver | No solver required for pure geometric adaptation |
| SE(3) invariance | Exact for translation; proven below |

---

## 2. DHB-QR: Quaternion Rotation Variant

### 2.1 Motivation

DHB-DR uses Euler XYZ angles to represent the relative frame rotation. While compact
(3 values), Euler angles suffer from:
- Gimbal lock when pitch approaches ±90°
- Discontinuities at Euler singularities
- Non-unique representations

DHB-QR replaces the 3 Euler angles with a unit quaternion (4 values), producing a
5-tuple per step. This avoids all singularities at the cost of one extra value.

### 2.2 5-Tuple per Step

For both linear and angular components:
```
I_i^QR = [m_i, q_w, q_x, q_y, q_z]  ∈ R^5
```

where `(q_w, q_x, q_y, q_z)` is the unit quaternion representation of `R_rel = F_i^T F_{i+1}`.

The magnitude `m_i` is identical to DHB-DR (inner product with local tangent).

### 2.3 Sign Convention for Continuity

Unit quaternions have a double-cover ambiguity: `q` and `-q` represent the same
rotation. To ensure temporal continuity of the quaternion sequence:

1. **Dot-product sign flip**: if `q_i · q_{i-1} < 0`, replace `q_i ← -q_i`
2. **Canonical form**: additionally enforce `q_w >= 0` (flip if negative)

The `enforce_qw_nonnegative=True` parameter applies both conventions, ensuring
`q_w ∈ [0, 1]` throughout the sequence. This is critical for:
- VLA tokenization (monotone range prevents token boundary issues)
- Interpolation (geodesic interpolation along the shorter arc)
- Consistency across demos of the same task

### 2.4 Reconstruction

Reconstruction from DHB-QR invariants follows the same frame-propagation logic as
DHB-DR, except the rotation matrix `R_lin` is recovered from `(q_w, q_x, q_y, q_z)`:
```
R_lin = quat_to_rot(I_i^QR[1:5])
```

### 2.5 When to Use DHB-QR vs DHB-DR

| Criterion | DHB-DR | DHB-QR |
|-----------|--------|--------|
| Invariant dimension | 4 | 5 |
| Gimbal lock risk | Possible at ±90° pitch | None |
| Sequence continuity | Generally smooth | Guaranteed with sign fix |
| VLA token embedding | Slightly more compact | Preferred for orientation-heavy tasks |
| Compression with FAST | 27.8× | ~25× (1 extra dim) |

For LIBERO-Spatial experiments, DHB-DR is sufficient (pitch rarely near ±90°).
DHB-QR is recommended for tasks with large end-effector rotations (e.g., pouring,
rotating objects, peg-in-hole).

---

## 3. DHB-TI: Time-Invariant Arc-Length Reparameterization

### 3.1 Motivation

Raw demonstrations have non-uniform time parameterization: the robot may slow near
contact, speed up in free space, or pause for grasping. This causes the DHB invariant
sequence to vary across demonstrations of the same task with different execution speeds.

DHB-TI (Time-Invariant) normalizes the trajectory to a fixed number of M steps using
arc-length (or hybrid) progress as the independent variable.

### 3.2 Progress Variable

Three progress variable types are supported:

**Translation (arc-length)**:
```
s_i = s_{i-1} + ||Δp_i||,   s_0 = 0
```

**Angular (rotation arc)**:
```
θ_i = θ_{i-1} + ||Δr_i||,   θ_0 = 0
```
where `Δr_i` is the axis-angle of the relative rotation between steps i and i+1.

**Hybrid** (default, `alpha=0.5`):
```
σ_i = σ_{i-1} + α ||Δp_i|| + (1-α) ||Δr_i||
```

The hybrid variable captures both translational and rotational effort, preventing
degenerate reparameterizations when the trajectory is mostly rotational or mostly
translational.

A minimum step `min_step=1e-10` is clamped at each segment to avoid degeneracy at
zero-motion segments.

### 3.3 Normalization to M Steps

Given total progress `Σ = σ_{N-1}`, define M uniform knots:
```
σ_k = k * Σ / (M - 1),   k = 0, 1, ..., M-1
```

Resample the trajectory at these knots:
- **Positions**: cubic spline interpolation in progress (one spline per coordinate)
- **Orientations**: SLERP between the two bounding quaternions, with local progress fraction

```python
t_local = (σ_k - σ_{lower}) / (σ_{upper} - σ_{lower})
q_k = SLERP(q_{lower}, q_{upper}, t_local)
```

The resampled trajectory `(pos_M, quat_M)` has exactly M poses regardless of the
original trajectory length N.

### 3.4 Full DHB-TI Encoding

```python
encode_dhb_dr_ti(positions, quaternions, M=100, progress_kind='hybrid', alpha=0.5)
```

Internally:
1. Compute progress `σ_i` via `compute_progress()`
2. Resample to M poses via `resample_by_progress()`
3. Apply standard `encode_dhb_dr()` to the M-pose trajectory

The output invariant array always has shape `(M-2, 4)` regardless of demo length.

### 3.5 Geometric Equivalence to DHB-DR

DHB-TI produces identical geometric endpoints as DHB-DR (0.0 mm error in all
perturbation experiments). This is expected: the arc-length reparameterization
only changes the timing, not the shape of the trajectory. The SE(3) invariance
proof in Section 6 applies equally to DHB-TI.

The practical benefit of DHB-TI is **speed-invariance**: two demos of the same
task at different execution speeds produce nearly identical invariant sequences,
enabling better generalization with few demonstrations.

---

## 4. OAT Tokenizer Architecture

### 4.1 Design Goals

The OAT (Ordered Action Tokenizer) compresses a T-step invariant sequence to K
discrete tokens (K << T) with the following properties:

1. **Compression**: T=50 timesteps → K=8 tokens (6.2× compression)
2. **No codebook collapse**: FSQ replaces learned VQ codebooks
3. **Ordered token space**: token k encodes finer detail than token k-1
4. **Anytime decoding**: prefix of k tokens produces valid reconstruction

### 4.2 Pipeline

```
Input (B, T, D) → RegisterEncoder → (B, K, latent_dim)
                → FSQ             → (B, K, latent_dim) + indices (B, K)
                → NestedDropout   → (B, K, latent_dim) [some tokens masked]
                → OATDecoder      → (B, T, D) [reconstructed]
```

Default hyperparameters (as used in rate-distortion experiments):
```python
OATTokenizer(
    input_dim=8,          # DHB invariant dimension (4 linear + 4 angular)
    input_horizon=50,     # T timesteps
    emb_dim=64,           # transformer hidden dim
    latent_dim=4,         # FSQ latent dim (= len(fsq_levels))
    num_registers=8,      # K tokens
    fsq_levels=[8, 6, 5, 4],  # codebook size = 8*6*5*4 = 960
    encoder_depth=4,
    decoder_depth=4,
    num_heads=4,
)
```

### 4.3 Register Encoder (Causal Attention)

The Register Encoder compresses T input timesteps into K register tokens.

**Architecture**:
- Linear input projection: `(B, T, input_dim)` → `(B, T, emb_dim)`
- Sinusoidal positional encoding added to input embeddings
- K learnable register parameters `r ∈ R^{K × emb_dim}` appended to the sequence
- Joint sequence `[x_emb | r]` of length T+K passes through L transformer layers
- Register tokens `out[:, T:]` extracted and projected to latent_dim

**Causal-Last Attention Mask**: The key architectural constraint is an asymmetric
attention mask that enforces the coarse-to-fine ordering:

```
Mask pattern (True = blocked):
  Input tokens:    attend to all input tokens (bidirectional)
  Input tokens:    cannot attend to register tokens
  Register k:      attends to all input tokens
  Register k:      attends causally to registers 0..k-1 only
```

In matrix form (rows = queries, cols = keys):
```
         [Input | Regs]
Input    [  0   |  1  ]    0 = attend, 1 = blocked
Regs     [  0   | UT  ]    UT = upper triangular (causal mask)
```

This means register k receives information from all input tokens plus the k-1
previously computed registers. The first register token captures the global
trajectory shape; subsequent tokens add progressively finer detail.

### 4.4 FSQ: No Codebook Collapse

Finite Scalar Quantization (Mentzer et al., 2024) replaces learned VQ codebooks
with deterministic rounding in a bounded space.

**Forward pass**:
```
z ∈ R^{B×K×D}           [continuous latents from encoder]
z_bounded = tanh(z + shift) * half_l - offset    [bound to [-L_i/2, L_i/2]]
z_q = round_ste(z_bounded)                        [round with STE gradients]
z_q = z_q / half_l                               [normalize to [-1, 1]]
```

where `half_l = (levels - 1) * (1 + ε) / 2` and `round_ste` uses straight-through
estimator: `dL/dz = dL/dz_q` (gradient passes through rounding).

**Codebook indexing**:
```
index = sum_d( (z_q_d + L_d//2) * basis_d )
basis = cumprod([1, L_0, L_1, ..., L_{D-2}])    [mixed-radix encoding]
```

For `levels=[8, 6, 5, 4]`: codebook size = 8×6×5×4 = 960.

**Why no collapse**: VQ-VAE codebooks suffer from codebook collapse (only a few
entries are used) because the discrete argmin operation creates vanishing gradients
for unused codes. FSQ has no learned codebook — all `prod(L_i)` codes are available
by construction, and the STE gradient flows through the rounding operation.

### 4.5 Nested Dropout (Matryoshka Masking)

Nested dropout forces the coarse-to-fine ordering during training by randomly masking
trailing tokens with a learnable mask token.

**Training**:
```
For each batch element b:
  Sample keep_k_b ~ Uniform(1, K)
  Replace tokens[b, keep_k_b:] with mask_token   [learned parameter]
```

**Inference (prefix decoding)**:
```
Given tokens[:, :k] (first k tokens):
  Pad to K tokens with zeros → tokens_full
  Replace tokens_full[:, k:] with mask_token
  Decode with OATDecoder
```

Sampling modes for `keep_k`:
- `uniform`: equal weight to all prefix lengths (default)
- `quadratic_biased`: favor longer prefixes (reduces underfitting of later tokens)
- `pow2`: only power-of-2 lengths (matches hierarchical structure)

The mask token `m ∈ R^{latent_dim}` is a learned parameter initialized from
`N(0, 0.02)`. During decoding, the decoder learns to predict "use less information"
when it sees the mask token.

### 4.6 Cross-Attention Decoder

The OATDecoder reconstructs the full T-step sequence from K register latents.

**Architecture**:
- Learned positional queries `Q ∈ R^{T × emb_dim}` (one per output timestep)
- K register latents projected to emb_dim as keys/values
- L transformer decoder layers (each layer: self-attention on queries + cross-attention to latents)
- Linear head: `(B, T, emb_dim)` → `(B, T, output_dim)`

The cross-attention allows each output timestep to attend to all K register tokens,
with the nested dropout ensuring that the decoder is robust to missing trailing tokens.

### 4.7 Training Loss

MSE reconstruction loss only (no VQ commitment loss or perplexity terms):
```
L = (1/BT) Σ_{b,t} ||x_hat_{b,t} - x_{b,t}||^2
```

This is the key simplification enabled by FSQ: standard VQ-VAE requires additional
codebook loss and commitment loss terms that can be unstable. FSQ eliminates these
entirely.

### 4.8 Rate-Distortion Table (Full)

Trained on 20 synthetic figure-8 DHB invariant sequences (50 timesteps, dim=8):

| Tokenizer | K tokens | Compression | MSE | Prefix k=1 | Prefix k=2 | Prefix k=4 |
|-----------|----------|-------------|-----|------------|------------|------------|
| OAT-style (ours) | **8** | 6.2× | 0.018 | 0.0013 | 0.0008 | 0.0005 |
| FSQ only | 8 | 6.2× | 0.019 | N/A | N/A | N/A |
| FAST (DCT+BPE) | 14 | 27.8× | 0.0002 | N/A | N/A | N/A |
| VQ-VAE | 50 | 8.0× | 0.020 | N/A | N/A | N/A |
| RVQ | 88 | 5.0× | 0.018 | N/A | N/A | N/A |
| Hierarchical | 200 | 2.2× | 0.015 | partial | partial | partial |

Notes:
- FAST dominates compression but lacks geometric invariance and prefix decoding
- OAT uniquely provides k=1 MSE=0.0013 (first token captures dominant motion shape)
- Trained on synthetic data; real-robot performance to be measured with StarVLA-DHB

---

## 5. Fatrop NLP Formulation

### 5.1 Problem Statement

Given a reference invariant sequence `Ĩ_demo = {I_i^demo}` from a demonstration, and
target boundary poses `(p_start, R_start)` and `(p_end, R_end)`, find adapted invariants
`Ĩ = {I_i}` that:
1. Stay close to the demo invariants (preserve motion shape)
2. Satisfy exact boundary constraints (start/end at the target poses)

### 5.2 State and Control Variables

**States** (at each of N+1 nodes):
```
p_k ∈ R^3          [end-effector position]
R_k ∈ SO(3)        [end-effector rotation, stored as 3 column vectors]
```

**Controls** (at each of N steps):
```
u_lin_k = [m_k, φ_x, φ_y, φ_z] ∈ R^4    [linear invariant]
u_ang_k = [ω_k, ψ_x, ψ_y, ψ_z] ∈ R^4    [angular invariant]
```

Total decision variables: `(N+1) * 15 + N * 8 = 15N + 15 + 8N = 23N + 15`.

### 5.3 Dynamics Constraints

The trajectory dynamics are the DHB decoder equations, implemented in CasADi:

**Linear dynamics**:
```
dR_lin_k = EulerXYZ_to_Rot(φ_k)          [frame rotation this step]
R_{k+1} = R_k @ dR_lin_k                 [propagate frame]
direction_k = R_{k+1}[:, 0]              [first column = tangent direction]
p_{k+1} = p_k + m_k * direction_k        [step along tangent]
```

### 5.4 Objective Function

Minimize deviation from demo invariants, weighted equally:
```
min  Σ_{k=0}^{N-1} [ w_inv * ||u_lin_k - u_lin_k^demo||^2
                   + w_inv * ||u_ang_k - u_ang_k^demo||^2 ]
```

Optional smoothness regularization (not used in paper experiments):
```
+ w_smooth * Σ_{k=1}^{N-1} ||u_lin_k - u_lin_{k-1}||^2
```

Default: `w_inv = 1.0`, `w_smooth = 0.01`.

### 5.5 Boundary Constraints

Hard equality constraints at start and end:
```
p_0 = p_start
p_N = p_end
R_0 = R_start     [lower-triangular entries, 6 constraints]
R_N = R_end       [lower-triangular entries, 6 constraints]
```

The rotation constraints use lower-triangular off-diagonal entries only
(3 per boundary) to avoid redundancy in the 9-element parameterization.

Initial orthogonality is also constrained:
```
R_0^T R_0 - I = 0     [6 independent constraints via tril]
```

### 5.6 Fatrop Solver Setup

The OCP is formulated using Rockit (rockit-meco) with multiple shooting:
```
ocp = rockit.Ocp(T=1.0)
ocp.method(rockit.MultipleShooting(N=N-1))
ocp.solver('fatrop', solver_opts)
```

Fatrop exploits the block-banded structure of the multiple-shooting KKT system for
6× speedup over IPOPT:

| Solver | First solve (JIT) | Subsequent solves | Speedup |
|--------|------------------|-------------------|---------|
| IPOPT  | ~45 ms | ~45 ms | 1× |
| Fatrop | ~500 ms | ~7 ms | 6× |

The first solve includes JIT compilation of the CasADi function graph. Subsequent
solves (same N, different boundary conditions) reuse the compiled function.

### 5.7 Warm Starting

Initial guesses for all variables:
- **Positions**: linear interpolation from `p_start` to `p_end`
- **Rotations**: constant at `R_start` (SLERP initialization is better but not implemented)
- **Invariants**: directly from demo invariants (resample if length differs)

### 5.8 Decode-Only Fallback

When Fatrop is unavailable (no rockit-meco), the system falls back to a decode-only
adaptation: encode the demo from `p_start` as the initial pose, decode directly to
get the adapted trajectory. This gives 0.0 mm endpoint error for pure translation
perturbations but does not satisfy goal-pose constraints for rotation offsets.

---

## 6. SE(3)-Invariance Proof Sketch

### 6.1 Claim

The DHB magnitude tokens `m_i = x_i^T · Δp_i` are invariant to global SE(3)
transformations of the trajectory (rigid translation + rotation).

### 6.2 Setup

Let `T_G = (R_G, t_G)` be a global SE(3) transform. The transformed trajectory is:
```
p_i' = R_G p_i + t_G
```

The initial frame axes (when `use_default_initial_frames=False`) are constructed from:
```
x_i' = Δp_i' / ||Δp_i'||
```

### 6.3 Translation Invariance

For a pure translation `t_G`:
```
Δp_i' = p_{i+1}' - p_i' = (p_{i+1} + t_G) - (p_i + t_G) = Δp_i
```

All differences are unchanged. Therefore `x_i' = x_i`, `y_i' = y_i`, `R_rel' = R_rel`,
and `m_i' = (x_i')^T Δp_i' = x_i^T Δp_i = m_i`. Exact invariance. QED.

### 6.4 Rotation Invariance

For a pure rotation `R_G`:
```
Δp_i' = R_G Δp_i
x_i' = R_G Δp_i / ||R_G Δp_i|| = R_G (Δp_i / ||Δp_i||) = R_G x_i
```

The x-axis transforms equivariantly (rotates with `R_G`).

The Householder double-reflection step:
```
v1' = Δp_{i+1}' - Δp_i' = R_G (Δp_{i+1} - Δp_i) = R_G v1
R1' = I - 2 (v1' v1'^T) / (v1'^T v1') = I - 2 R_G (v1 v1^T) R_G^T / ||v1||^2
    = R_G (I - 2 v1 v1^T / ||v1||^2) R_G^T = R_G R1 R_G^T
```

Similarly `R2' = R_G R2 R_G^T`. The combined transport:
```
R' = R2' R1' = R_G R2 R_G^T R_G R1 R_G^T = R_G (R2 R1) R_G^T = R_G R R_G^T
y_{i+1}' = R' y_i' = R_G R R_G^T R_G y_i = R_G R y_i = R_G y_{i+1}
```

The y-axis also transforms equivariantly.

**Relative rotation invariance**:
```
F_i' = [x_i' | y_i' | z_i'] = R_G [x_i | y_i | z_i] = R_G F_i
R_rel' = (F_i')^T F_{i+1}' = (R_G F_i)^T (R_G F_{i+1}) = F_i^T R_G^T R_G F_{i+1}
       = F_i^T F_{i+1} = R_rel
```

The relative frame rotation is exactly invariant.

**Magnitude invariance**:
```
m_i' = (x_i')^T Δp_i' = (R_G x_i)^T (R_G Δp_i) = x_i^T R_G^T R_G Δp_i = x_i^T Δp_i = m_i
```

Exact magnitude invariance. QED.

### 6.5 Important Caveat: Initial Frame Dependence

The invariance proof requires `use_default_initial_frames=False` (trajectory-derived
initial frames). Under the default mode (`use_default_initial_frames=True`), initial
frames are fixed to the world-frame x and y axes. A global rotation `R_G` changes the
relative orientation between the fixed world frames and the trajectory tangent, causing
the initial few invariants to change.

Measured effect (n=40 SE(3) transforms):

| Mode | Token change rate |
|------|------------------|
| `use_default_initial_frames=False` | **0.0%** |
| `use_default_initial_frames=True` (default) | **95.7%** |

The decode pipeline uses `use_default_initial_frames=True` (default) because it needs
a fixed reference for absolute pose reconstruction. The 0.0% token stability is measured
separately with non-default frames, demonstrating the theoretical property.

### 6.6 Summary of Invariance Properties

| Property | DHB-DR | Raw actions |
|----------|--------|-------------|
| Translation invariance | Exact | None |
| Rotation invariance (magnitude) | Exact | None |
| Rotation invariance (frame angles) | Exact (relative R_rel) | None |
| SE(3) token stability | 0.0% change | 99.1% change |
| Practical decode error | 0.0 mm | 50-200 mm at 5-20 cm |
