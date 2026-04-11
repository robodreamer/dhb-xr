# DHB-XR Honest VLA Integration Pipeline Plan

**Date:** 2026-04-10
**Status:** In progress — Step 1.1 running (20 seeds x 7 offsets)
**Goal:** Ground DHB-XR's value proposition for VLAs with honest, peer-review-surviving experiments

---

## Background & Motivation

Two independent critic agents reviewed the current SmolVLA + DHB-XR pipeline and identified a fundamental honesty problem:

**Current pipeline (REJECTED):**
1. Run SmolVLA at 0cm (easy case) -> collect EE trajectory
2. DHB-encode trajectory -> DHB-decode from offset start
3. Execute open-loop via embodik IK at offset

**Why it fails peer review:**
- Apples-to-oranges: open-loop geometric replay vs closed-loop VLA inference
- SmolVLA never sees the offset scene -- used as trajectory database, not policy
- Oracle knowledge of offset is given to DHB but not SmolVLA
- A reviewer asks: "Why not just fine-tune on offset data?"

**Agreed framing (strongest -> weakest):**
1. DHB as invariant action representation layer (pre-processing before any tokenizer)
2. DHB as free SE(3) data augmentation (generate unlimited spatial variations)
3. DHB as post-processing correction (weakest, avoid)

---

## Execution Plan

### Step 1: Degradation Curve Experiment (FIRST PRIORITY)
- [x] **1.1** Run SmolVLA on LIBERO-Spatial task 0 at offsets: 0, 1, 2, 3, 5, 7, 10 cm (Y-axis) — RUNNING
- [x] **1.2** 20 seeds per offset condition — script: `_workspace/eval_degradation_curve.py`
- [ ] **1.3** Record success rate + mean reward at each offset
- [ ] **1.4** For each trajectory, measure FAST token edit distance (0cm vs offset) and DHB invariant distance
- [ ] **1.5** Plot: success rate vs offset (the paper's motivating figure)
- [ ] **1.6** Include rotational offsets: 0, 5, 10, 15, 20, 30 degrees

**Script base:** `examples/integration/eval_smolvla_libero_offset.py`
**Environment:** `~/miniforge3/envs/libero/bin/python` with `MUJOCO_GL=egl`
**Expected output:** `_workspace/degradation_curve_results.json` + matplotlib figure

**What this proves:** Quantifies exactly where SmolVLA breaks. If it doesn't degrade, the paper's premise needs rethinking.

**Key details from previous work:**
- SmolVLA state format: `eef_pos(3) + axis_angle(3) + gripper_qpos(2)` = 8D
- Images must be flipped 180 degrees (`torch.flip(img, dims=[2,3])`)
- Task description needs newline appended (`task_name + '\n'`)
- Use `make_pre_post_processors()` from `lerobot.policies.factory`
- SmolVLA achieves ~67-83% at 0cm (3-6 trials) -- need 20+ for credibility

---

### Step 2: DHB Augmentation Training (~10 GPU-hours)
- [x] **2.1** DHB-encode all 50 LIBERO-Spatial task 0 demos — DONE (18.8s, 0 failures)
- [x] **2.2** For each demo, DHB-decode at 9 random offsets (uniform [-5cm, +5cm] per axis) -> 450 augmented demos — DONE
- [x] **2.3** Include arc-length invariant gripper encoding for augmented demos — DONE
  - Output: `_workspace/augmented_demos_task0.npz` (5.4 MB)
- [ ] **2.4** Fine-tune SmolVLA (3 runs):
  - (a) Original 50 demos only (baseline)
  - (b) 50 original + 450 DHB-augmented demos
  - (c) 50 original + 450 naive-offset demos (translate EE positions without shape preservation)
- [ ] **2.5** Evaluate all 3 at 0cm + offsets (0, 3, 5, 7 cm), 20 seeds each, closed-loop inference
- [ ] **2.6** Plot: success rate vs offset for all 3 conditions

**What this proves:** DHB augmentation genuinely improves VLA spatial generalization. Same model, same inference, only training data differs. The comparison against naive augmentation isolates the value of geometric invariance.

**Technical notes:**
- SmolVLA is 500M params -- fine-tuning on LIBERO ~1-2 GPU-hours per run
- DHB encode/decode: `src/dhb_xr/core.py` (`encode_dhb_dr`, `decode_dhb_dr`)
- LIBERO demo loading: `examples/integration/libero_full_demo.py`
- Augmented demos must be in lerobot HDF5 format for SmolVLA training

---

### Step 3: DHB Invariant Action Head (strongest, ~1-2 weeks if time permits)
- [ ] **3.1** Implement `DHBActionHead` for SmolVLA that predicts DHB invariant tokens (8D per step)
- [ ] **3.2** Use OAT-style tokenizer (already implemented in `src/dhb_xr/tokenization/`) to compress invariants to 8 register tokens
- [ ] **3.3** At inference: VLA sees offset scene -> predicts DHB tokens -> decode from actual EE pose -> embodik IK
- [ ] **3.4** Train on zero-offset demos only
- [ ] **3.5** Evaluate at 0cm + offsets -- VLA genuinely infers at every offset
- [ ] **3.6** Ablation: same architecture predicting raw 7-DoF vs DHB invariants (isolates representation effect)

**What this proves:** VLA trained to predict SE(3)-invariant tokens generalizes across spatial configs without offset-augmented data. The invariant representation makes the policy inherently robust.

**This is the experiment described in methods.tex Section 4.4 (`DHBActionHeader`).**

---

### Step 4 (Alternative to Step 3): Invariant Retrieval + Adaptation
- [ ] **4.1** Build DHB motion database from 50 LIBERO demos (`MotionDatabase` already exists)
- [ ] **4.2** At test time: SmolVLA produces trajectory sketch at offset -> encode to DHB -> retrieve nearest demo -> Fatrop solver adapts to actual pose -> execute
- [ ] **4.3** Compare: (a) SmolVLA raw, (b) SmolVLA + naive offset, (c) SmolVLA sketch + DHB retrieval, (d) Demo + DHB adaptation (oracle upper bound)

**Framing:** "VLA handles semantic intent, DHB handles geometric adaptation"
**Everything already implemented:** `MotionDatabase`, `generate_trajectory_fatrop`, `LiberoAdapter`

---

## Critical Baselines (reviewers will demand these)

- [ ] **B1** "Just fine-tune on offset data": Collect 10 more demos at each offset, fine-tune SmolVLA, evaluate
- [ ] **B2** "Just add translational noise to training": Standard domain randomization of workspace positions
- [ ] **B3** "Naive geometric shift": Translate EE trajectory by offset vector without DHB shape preservation

---

## Paper Narrative Update

**Current (rejected):** "DHB-XR adapts trajectories to new spatial configs in 7ms"
-> This is trajectory retargeting, not a VLA contribution

**Proposed narrative:** "VLAs are data-hungry for spatial diversity. DHB-XR provides free SE(3) data augmentation -- geometrically correct trajectory variations from a single demo in 7ms. SmolVLA + DHB augmentation matches models trained on 10x more real data."

**Key numbers to produce:**
1. Degradation curve: SmolVLA drops from X% to Y% at 5cm offset
2. DHB augmentation recovery: SmolVLA+DHB-aug achieves Z% at 5cm (vs Y% baseline)
3. Naive augmentation comparison: naive-aug achieves W% (worse than DHB-aug because shape is distorted)
4. Token stability: DHB tokens change 0% under SE(3) vs FAST 100% (already have this)
5. Compression: 49.8x under FAST on DHB invariants (already have this)

---

## Project Page Updates After Experiments

- [ ] Replace "SmolVLA vs DHB-XR" comparison table with degradation curve figure
- [ ] Add augmentation training results table (original vs DHB-aug vs naive-aug)
- [ ] Remove or honestly relabel open-loop DHB adaptation results
- [ ] Update "Execution Gap" callout to reflect actual closed-loop results
- [ ] Add "Data Efficiency" section showing augmentation multiplier

---

## Files Reference

| File | Purpose |
|------|---------|
| `examples/integration/eval_smolvla_libero_offset.py` | SmolVLA eval script (needs multi-offset loop) |
| `src/dhb_xr/core.py` | DHB encode/decode (`encode_dhb_dr`, `decode_dhb_dr`) |
| `src/dhb_xr/tokenization/` | OAT-style tokenizer for DHB invariants |
| `_workspace/offset_search.py` | Existing offset search (IK stage reference) |
| `scripts/record_dhb_success_video.py` | Video recording with JOINT_POSITION controller |
| `docs/paper/index.html` | Project page (needs narrative update) |
| `manuscript/methods.tex` | Paper methods (Section 4.4 describes DHB action head) |

## Environment Setup

- **SmolVLA inference:** `~/miniforge3/envs/libero/bin/python` (py310, lerobot + transformers)
- **LIBERO sim:** Same env, `MUJOCO_GL=egl`
- **embodik IK:** pixi env (py312), `pixi run python`
- **DHB-XR core:** `pixi run python` or any env with `dhb_xr` installed

## Key Constraints

- LIBERO renders at 256x256 max (hardware cap)
- SmolVLA expects 224x224 input images
- JOINT_POSITION controller: gains must be set AFTER `env.reset()`
- embodik targets `panda_hand_tcp` (not `panda_link8`) with base_offset Z=0.919
- Bowl Y position = `init_state[11]` (= qpos[10])
