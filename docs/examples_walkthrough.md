# Examples Walkthrough

This document walks through each example script and notebook in `dhb_xr`, what it does, and how to run it.

## Test results (summary)

All seven example scripts were run from the repo root with `PYTHONPATH=src python3 examples/<script>.py`. Results:

| Example | Output |
|--------|--------|
| `01_basic_encoding.py` | Encoded invariants `(52, 4)`, decoded positions `(50, 3)`, quaternions `(50, 4)` |
| `02_trajectory_adaptation.py` | Adapted positions `(28, 3)`, first/last pose printed |
| `03_dhb_qr_comparison.py` | DHB-DR linear `(42, 4)`, DHB-QR linear `(42, 5)` |
| `04_gpu_batch_optimization.py` | Adapted positions `(2, 30, 3)`, last poses per trajectory |
| `05_vla_tokenization.py` | Invariants `(1, 42, 8)`, token indices `(1, 42)`, reconstruction MSE printed |
| `06_motion_database.py` | Added 4 trajectories; top-3 retrieval with L2 distances and metadata |
| `07_imitation_learning.py` | Invariant matching loss, SE(3) geodesic loss, hybrid loss printed |
| `08_dhb_ti_time_invariant.py` | Progress totals (trans/angular/hybrid), resampled shapes, DHB-TI (DR/QR) invariant shapes |

## Prerequisites

From the repo root:

```bash
cd /path/to/dhb_xr
pip install -e .   # or pip install -e ".[gpu,database]" for optional features
```

To run examples without installing (development):

```bash
export PYTHONPATH=src
python3 examples/01_basic_encoding.py
```

---

## 01_basic_encoding.py — Encode and decode a trajectory

**What it does:** Builds a short SE(3) trajectory (50 poses: positions + identity quaternions), encodes it to DHB-DR invariants, then decodes back to poses.

**Concepts:**
- **Encode:** `encode_dhb_dr(positions, quaternions, ...)` returns `linear_motion_invariants` (N-2, 4) and `angular_motion_invariants` (N-2, 4), plus `initial_pose` and frame info.
- **Decode:** `decode_dhb_dr(linear_inv, angular_inv, initial_pose, ...)` returns `positions` and `quaternions` (wxyz).
- With default padding, encoder output length is slightly longer than N-2; decoder `drop_padded=True` returns a pose sequence aligned with the original length convention.

**Run:**
```bash
python3 examples/01_basic_encoding.py
```
**Expected output:** Encoded invariants shape `(52, 4)` (for N=50 with padding), decoded positions/quaternions shapes `(50, 3)` and `(50, 4)`.

---

## 02_trajectory_adaptation.py — Resample and retarget

**What it does:** Takes a demo trajectory, resamples it to a target length, and produces an “adapted” trajectory that starts at `pose_target_init` and is decoded from the demo’s invariants (no full CasADi optimization; retargeting is by resample + decode with new initial pose).

**Concepts:**
- **generate_trajectory(...)** resamples the demo to `traj_length`, encodes, then decodes using `pose_target_init`, so the new trajectory starts at the desired pose.
- Returns `adapted_pos_data`, `adapted_quat_data`, and resampled demo data.

**Run:**
```bash
python3 examples/02_trajectory_adaptation.py
```
**Expected output:** Adapted positions shape e.g. `(28, 3)` and printed first/last pose.

---

## 03_dhb_qr_comparison.py — DHB-DR vs DHB-QR

**What it does:** Encodes the same trajectory with DHB-DR (Euler) and DHB-QR (quaternion) and compares output shapes.

**Concepts:**
- **DHB-DR:** 4 values per component (magnitude + 3 Euler angles).
- **DHB-QR:** 5 values per component (magnitude + unit quaternion wxyz), avoiding gimbal lock and giving smooth quaternion-geodesic losses.

**Run:**
```bash
python3 examples/03_dhb_qr_comparison.py
```
**Expected output:** Linear invariants shapes `(42, 4)` (DR) and `(42, 5)` (QR).

---

## 04_gpu_batch_optimization.py — Batched trajectory optimization

**What it does:** Uses `BatchedTrajectoryOptimizer` to optimize invariants for two different start/goal pose pairs so that decoding matches the goal pose (scipy L-BFGS over invariant sequence).

**Concepts:**
- One demo trajectory; two initial poses and two goal poses.
- Optimizer minimizes a loss that decodes invariants from each init and penalizes distance of the decoded end pose to the goal (position + rotation).
- Returns batched adapted positions and quaternions.

**Run:**
```bash
python3 examples/04_gpu_batch_optimization.py
```
**Expected output:** Adapted positions shape `(2, 30, 3)` and last poses for each trajectory.

---

## 05_vla_tokenization.py — VQ-VAE tokenization (DHB-Token)

**What it does:** Encodes a trajectory to invariants, then runs it through the causal VQ-VAE tokenizer (`DHBTokenizer`) to get discrete token indices and a reconstruction.

**Concepts:**
- Invariants shape `(T, 8)` (linear 4 + angular 4).
- Tokenizer: causal encoder → VQ → decoder; output token indices `(1, T)` and reconstructed invariants `(1, T, 8)`.
- Useful for VLA action representation: invariant sequence → tokens → autoregressive or streaming decode.

**Run:** Requires `torch`.
```bash
python3 examples/05_vla_tokenization.py
```
**Expected output:** Invariants shape, token indices shape, reconstructed shape, and reconstruction MSE.

---

## 06_motion_database.py — Motion DB add and retrieve

**What it does:** Builds a small motion database by adding four trajectories (as positions + quaternions), then retrieves the top-3 closest trajectories to a query trajectory by L2 distance in invariant space.

**Concepts:**
- **MotionDatabase:** stores encoded invariants and optional metadata; `add(positions, quaternions, metadata)`; `retrieve(query_positions, query_quaternions, k=3)`.
- Similarity is L2 on flattened invariants (optional DTW via `use_dtw=True`).

**Run:**
```bash
python3 examples/06_motion_database.py
```
**Expected output:** “Added 4 trajectories” and top-3 retrieval with distances and metadata.

---

## 07_imitation_learning.py — Invariant and geodesic losses

**What it does:** Compares a “predicted” trajectory to a “demo” trajectory using:
1. **Invariant matching loss:** L2 (or quaternion-aware) between predicted and demo invariant sequences.
2. **SE(3) geodesic loss:** On the final pose (position + rotation).
3. **Hybrid loss:** Weighted combination of invariant and pose losses.

**Concepts:**
- `invariant_matching_loss(U_pred, U_demo, method="dhb_dr")`
- `se3_geodesic_loss_np(pos_pred, quat_pred, pos_demo, quat_demo, beta=1.0)`
- `hybrid_invariant_pose_loss(..., alpha=0.5)` for imitation / behavior cloning.

**Run:**
```bash
python3 examples/07_imitation_learning.py
```
**Expected output:** Invariant matching loss, SE(3) geodesic loss (final pose), and hybrid loss values.

---

## 08_dhb_ti_time_invariant.py — DHB-TI (time-invariant reparameterization)

**What it does:** Reparameterizes a trajectory by a geometric progress variable (translational arc-length, angular, or hybrid) and resamples at uniform progress knots so that invariants are approximately independent of execution speed and sampling rate. Then encodes with DHB-DR or DHB-QR.

**Concepts:**
- **Progress variables:** `translation` (s_{i+1} = s_i + ||Δp_i||), `angular` (θ_{i+1} = θ_i + ||Δr_i||), `hybrid` (σ_{i+1} = σ_i + α||Δp_i|| + (1-α)||Δr_i||).
- **Uniform knots:** σ_k = k Σ/(M-1); interpolate poses at σ_k (position spline, quaternion SLERP).
- **encode_dhb_dr_ti**, **encode_dhb_qr_ti**: resample by progress to M samples, then encode.

**Run:**
```bash
python3 examples/08_dhb_ti_time_invariant.py
```
**Expected output:** Progress totals for translation, angular, hybrid; resampled positions (M, 3); DHB-TI (DR) and (QR) linear invariant shapes.

---

## benchmark_backends.py — Encode/decode timing

**What it does:** Times 50 encode and 50 decode calls for a trajectory of length 100 and reports average milliseconds per call (numpy backend).

**Run:**
```bash
python3 examples/benchmark_backends.py
```
**Expected output:** Encode ~20 ms and decode ~6 ms per call (machine-dependent).

---

## Running notebooks with pixi

From the repo root, use the pixi environment (dev feature includes Jupyter):

```bash
cd /path/to/dhb_xr
pixi install          # installs deps including jupyter, ipykernel
pixi run notebook     # editable install + launch Jupyter in notebooks/
```

Then open **demo_dhb_features_and_use_cases.ipynb** (or any notebook) in the browser. The kernel uses the pixi environment, so `dhb_xr` is importable after the editable install.

Alternatively, start a shell and run Jupyter manually:

```bash
pixi shell
pip install -e .      # so `import dhb_xr` works
jupyter notebook notebooks/
```

---

## Notebooks

- **demo_dhb_features_and_use_cases.ipynb:** **Comprehensive demo** (inspired by MATLAB Live Scripts): 6-DOF trajectory construction, encode/decode, SE(3) invariance, DHB-DR vs DHB-QR, DHB-TI (time-invariant), trajectory adaptation, and optional VLA tokenization — with **elaborate markdown descriptions and plots** (3D trajectories, invariant time series). Best starting point for understanding features and use-cases.
- **tutorial_dhb_basics.ipynb:** Encode → decode and print shapes (same idea as `01_basic_encoding.py`).
- **tutorial_invariance_demo.ipynb:** Apply a random SE(3) to a trajectory, encode both; shows that invariant magnitudes (first column) match.
- **tutorial_vla_integration.ipynb:** Encode trajectory to invariants, run through `DHBTokenizer`, show token indices and reconstructed shape.

Open in Jupyter:

```bash
jupyter notebook notebooks/
```

---

---

## Integration Examples (LIBERO / RoboCASA)

Located in `examples/integration/`, these scripts demonstrate DHB-XR with VLA benchmarks.

### test_libero_adapter.py — Load LIBERO datasets

**What it does:** Tests the `LiberoAdapter` for loading LIBERO HDF5 datasets.

**Prerequisites:** Download LIBERO-Spatial dataset:
```bash
mkdir -p ~/Projects/data/libero && cd ~/Projects/data/libero
wget -O libero_spatial.zip "https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip"
unzip libero_spatial.zip
```

**Run:**
```bash
pixi run python examples/integration/test_libero_adapter.py
```
**Expected output:** 50 episodes loaded with positions (N, 3) and quaternions (N, 4).

---

### test_libero_encoding.py — DHB encoding on LIBERO data

**What it does:** Loads LIBERO demos and encodes them with DHB-DR.

**Run:**
```bash
pixi run python examples/integration/test_libero_encoding.py
```
**Expected output:** Linear/angular invariant shapes for 5 episodes, no NaN/Inf values.

---

### test_libero_retrieval.py — Motion retrieval across tasks

**What it does:** Builds a motion database from 5 LIBERO tasks and queries similar motions using DTW.

**Run:**
```bash
pixi run python examples/integration/test_libero_retrieval.py
```
**Expected output:** Top-5 similar motions for each query, showing that DHB captures motion structure rather than spatial location (SAME/DIFF task labels).

---

### libero_simulation_demo.py — Basic LIBERO demo

**What it does:** Demonstrates DHB encoding/decoding with LIBERO. If LIBERO is installed, runs the demo in simulation.

**Run (without LIBERO installed):**
```bash
pixi run python examples/integration/libero_simulation_demo.py
```
**Expected output:** DHB encoding/decoding roundtrip with near-zero reconstruction error.

---

### libero_full_demo.py — Complete LIBERO integration demo

**What it does:** Comprehensive demo showing:
1. DHB encoding/decoding on real LIBERO data
2. Trajectory adaptation (retargeting to new poses)
3. Full simulation with LIBERO environment
4. Motion retrieval across tasks

**Run modes:**

```bash
# DHB-only mode (no simulation dependencies required)
pixi run python examples/integration/libero_full_demo.py --dhb-only

# Motion retrieval demo
pixi run python examples/integration/libero_full_demo.py --retrieval

# Full simulation (requires libero conda environment)
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py

# With OpenCV display (requires X11 display, press 'q' to quit)
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --render

# Save video for later viewing (works headless)
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --save-video demo.mp4

# Both display and save video
~/miniforge3/bin/mamba run -n libero python examples/integration/libero_full_demo.py --render --save-video demo.mp4
```

**Expected output:**
- DHB-only: Encoding shape `(100, 4)`, reconstruction error ~0.000 mm, trajectory adaptation demo, plot saved to `/tmp/dhb_demo_plot.png`
- Simulation: Task completion at step 78 (varies by task)
- Retrieval: Top-5 matches with distances showing similar task structures grouped together
- Video: 3-4 second MP4 showing the robot executing the pick-and-place task

**Visualizations:**
- DHB-only mode saves a matplotlib plot showing 3D trajectories (original vs adapted) and invariant magnitudes
- Full simulation mode can display live OpenCV window or save video

---

## Running all examples

```bash
cd /path/to/dhb_xr
export PYTHONPATH=src
for f in examples/0*.py examples/07_imitation_learning.py examples/08_dhb_ti_time_invariant.py; do echo "=== $f ==="; python3 "$f"; done
```
