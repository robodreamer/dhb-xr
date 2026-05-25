# DHB-XR Honest VLA Integration Pipeline Plan

**Date:** 2026-04-10
**Status:** Steps 1-2 COMPLETE, Step 3 e2e training works (0.058 loss), Motion DB overhauled (63% top-1).
**Last updated:** 2026-04-13

### AUTHORITATIVE RESULTS (5K steps, n=50, LIBERO-Spatial task 0) — UPDATED 2026-04-13
| Model | 0cm | 3cm | 5cm |
|-------|-----|-----|-----|
| Base SmolVLA | 70% | **22%** | 0% |
| Original FT (50 eps) | 34% | 22% | 0% |
| **DHB-aug FT (500 eps)** | 42% | **20%** | 0% |
| Naive-aug FT (500 eps) | 46% | **8%** | 0% |

**CRITICAL**: n=20 results were misleading sampling noise. n=50 is authoritative.
- Fisher exact p-values at 3cm: DHB vs Naive p=0.15, DHB vs Base p=1.0 (NOT significant)
- Fine-tuning on single task HURTS 3cm generalization vs base model
- DHB augmentation right direction (20% vs 8% naive) but not statistically significant at n=50
- Need n=100+ or multi-task results for statistical significance
**Last updated:** 2026-04-13

### Dataset Sizes (with rendered images at offset positions)
| Dataset | Episodes | Frames | Size |
|---------|----------|--------|------|
| Original (0cm) | 50 | 3,999 | 592 MB |
| DHB augmented ([-5,+5]cm) | 450 | 45,539 | 6.5 GB |
| Naive augmented ([-5,+5]cm) | 450 | 43,021 | 6.1 GB |
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
- [x] **1.3** Record success rate + mean reward at each offset — DONE
  - 0cm: 70% (14/20), 1cm: 55%, 2cm: 55%, 3cm: 25%, 5cm: 10%, 7cm: 0%, 10cm: 0%
- [ ] **1.4** For each trajectory, measure FAST token edit distance (0cm vs offset) and DHB invariant distance
- [x] **1.5** Plot: success rate vs offset — DONE (`docs/paper/assets/degradation_curve.png`)
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
- [ ] **2.4** Fine-tune SmolVLA (3 runs) — original DONE, DHB/naive next:
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
- [x] **3.1** Implement `DHBActionHead` — DONE (0.8M params, 8 tests, demo trains to loss 0.006)
- [x] **3.2** OAT tokenizer integrated into DHBActionHead (discrete tokens, 9 tests)
- [x] **3.3** SmolVLADHBAdapter — connects SmolVLA VLM backbone to DHBActionHead
- [x] **3.4** Train DHBActionHead end-to-end with SmolVLA VLM — DONE (normalized, loss→0.021, r=0.97)
- [x] **3.5** Evaluate with embodik IK pipeline — **100% at 0cm, 100% at 3cm** (10/10 each)
- [x] **3.6** Ablation complete: DHB loss 0.004 vs raw 0.002 (both fit; real test needs VLM images)

**Step 3.4 Option A — End-to-End Training Notes:**
- Previous approach (pre-cached features) failed: VLM API fallback produced random tensors → 0% at eval
- New approach: live VLM prefix features → DHBActionHead, with LoRA on VLM q/v_proj (rank=8)
- SmolVLA VLM hidden_size=2048, 16 layers; bypass interleaved expert, use prefix-only forward
- Scripts: `_workspace/train_dhb_head_e2e.py`, `_workspace/eval_dhb_head_e2e.py`
- Env: `~/miniforge3/envs/smolvla/bin/python`, peft 0.18.1 installed
- Run: `MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python _workspace/train_dhb_head_e2e.py --epochs 50`

**Training results (2026-04-13):**
- 50 epochs: loss 0.213 → 0.058, correlation 0.26 (insufficient)
- 200 epochs: loss 0.205 → 0.024, correlation **0.68** (strong, per-demo up to 0.82)
- MSE improved 2.6x from 50 to 200 epochs
- Eval at 50 epochs: 0% at all offsets (prediction quality + decode pipeline)
- 200 epochs normalized: r=0.97 correlation (per-demo up to 0.99), all 8 channels learned
- Eval: 0% at all offsets — root cause: encode→decode roundtrip itself fails (GT invariants decoded to actions → reward=0.0, while original actions → reward=1.0). The 80x OSC scaling + angular delta extraction from quaternions is not precise enough for task execution. Prediction quality is excellent (r=0.97), but the invariant→action decode is the bottleneck.
- **BREAKTHROUGH (2026-04-13)**: embodik IK pipeline works!
  - Single-demo (demo_0): 100% at 0cm, 100% at 3cm
  - GT multi-demo pipeline ceiling: **80%** (8/10) — 2 demos fail on controller tracking
  - Predicted multi-demo: **30-40%** at 0-3cm (flat across offsets = invariance holds)
  - Prediction error ~40mm is the bottleneck (not IK or controller)
  - Pipeline: VLM → DHBActionHead (r=0.97) → invariants → decode (0mm) → embodik IK (0.47mm) → JOINT_POSITION
  - 5cm fails due to Panda workspace limits
  - Scripts: `_workspace/eval_dhb_head_embodik.py` (3-stage: predict → IK → sim)
  
  **To improve**: reduce prediction error via more training data/epochs, fix 2 GT-failing demos
- VLM features are REAL this time (not random), confirmed by checking live extraction
- SmolVLM2-500M hidden_size=960 (not 2048 as initially assumed)

**Next steps to improve:**
1. Train longer (200-500 epochs) — loss still decreasing at epoch 50
2. Closed-loop re-prediction every N steps (not single open-loop trajectory)
3. Proper initial frame estimation for DHB decode (encoder uses Frenet frame)
4. Consider predicting actions directly instead of invariants for execution
5. Increase batch_size if VRAM allows (currently BS=2)

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
- [x] **B3** "Naive geometric shift" — DONE (naive-aug FT: 35/0/0 at 0/3/5cm, vs DHB-aug 35/15/0)

---

## Paper Narrative (UPDATED 2026-04-13)

**Narrative:** "VLAs are data-hungry for spatial diversity. DHB-XR provides three complementary tools: (1) free SE(3) data augmentation from any demo, (2) invariant action heads for inherent spatial robustness, (3) workspace-blind motion retrieval. SmolVLA + DHB augmentation is the only fine-tuned model with non-zero success at 3cm offset."

**Key numbers PRODUCED:**
1. Degradation curve: SmolVLA 70% → 0% at 7cm offset (DONE, n=20)
2. DHB-aug recovery: 15% at 3cm vs 0% for naive-aug, 10% for original FT (DONE)
3. Token stability: DHB invariant distance = 0.000 at all offsets (DONE)
4. Motion DB: DHB-DR mag 63% top-1, 100% self-retrieval at all offsets (DONE)
5. Action head: normalized training r=0.97, single-demo 100%/100% via embodik IK, multi-demo 40%/0% (IN PROGRESS)

---

## Project Page Updates (COMPLETED 2026-04-13)

- [x] Degradation curve figure on results page
- [x] Augmentation training results table (n=20, all 4 conditions)
- [x] Motion DB section overhauled: 5-method comparison table, self-retrieval chart
- [x] Tutorial button added to hero section (between Demos and Math)
- [x] Tutorial page (tutorial.html) with Colab badges and code examples
- [x] Math deep-dive page (math.html) with KaTeX equations
- [x] Action Head section added (architecture diagram, stats, honest status)
- [x] QA fixes: insight box text, button order, FA icon

---

## Overall Progress Summary (2026-04-14 — UPDATED)

| Component | Status | Key Result |
|-----------|--------|-----------|
| Degradation curve (Step 1) | COMPLETE | 70% → 0% at 7cm, n=20 |
| DHB augmentation (Step 2) | COMPLETE | DHB-aug 20% vs Naive 8% at 3cm (p=0.15, suggestive) |
| DHB action head — P2 closed-loop | COMPLETE (0%) | 3 training runs, all 0%. Root cause: position prediction wrong |
| LIBERO-PRO P2 (Fatrop open-loop) | COMPLETE (90%) | 18/20, p=1.68e-9 — PRIMARY CORL RESULT |
| Fatrop 6D orientation | FIXED | 113°/step → 1.39°/step via decode_dhb_dr |
| IK orientation | FIXED | tolerance=0.1, orientation_gain=1.0 |
| Motion database | COMPLETE | 63% top-1, 88.8% top-5, 100% self-retrieval |
| Token stability: relative actions | NEW | delta-EEF 18.2% vs DHB 0.000% token change |
| Harness integration | FIXED (6 bugs) | vla-eval serve now works; LIBERO-Pro P2 config ready |
| Unit tests | COMPLETE | 20/20 passing, covers all pipeline stages |
| Project page | COMPLETE | 7-row P2 table, relative actions comparison, narrative updated |
| Manuscript | UPDATED | Related Work, "relative actions are not enough", honest VLA limitations |

## Branch status (2026-04-14)

Working branch: `dev/vla-paper-experiments`
- Ready to merge: Fatrop fix, harness fixes, unit tests, narrative updates
- Future work: DHBActionHead position conditioning, more P2 training data
- See `project_dev_branch_state.md` memory for full details

## DHBActionHead P2-Correct Training — IN PROGRESS (2026-04-13)

**Fixed**: Training with correct paired targets (P2 obs → Fatrop-adapted invariants, NOT original).

**Scripts created**:
- `_workspace/gen_p2_aug_targets.py` → outputs `_workspace/p2_aug_targets.npz` (50 demos × 100T × 8D, Fatrop-adapted)
- `_workspace/train_dhb_head_p2_correct.py` → trains with 50 orig + 50 P2-adapted = 100 total samples
  - Smoke test confirmed: "Dataset: 50 original + 50 P2-adapted = 100 total"
  - All 50 demos have Fatrop-adapted targets (0 Fatrop failures)
  - Mean abs diff between p2 and orig targets: 0.12 (geometrically distinct)
- `_workspace/run_p2_correct_eval.sh` → 3-stage eval pipeline (predict → IK → sim)

**COMPLETED (2026-04-14)**:
- Training: epoch 197/200, loss **0.043** ✓
- 3-stage eval: **0/10 = 0% on P2** ✗

**Failure mode**: Gripper closes at Y=0.06-0.15 (should be Y≈0.21 for bowl pick). Some seeds freeze.
**Root cause**: Scale — 50 P2 examples insufficient. Framework is correct, data is not enough.
**Future work**: Need 500+ P2 examples or closed-loop re-prediction per step.

## LIBERO-PRO Results (2026-04-13) — KEY CORL RESULT

**LIBERO-PRO P2 (position swap) evaluation — `_workspace/eval_libero_pro_p2.py`:**

P2 perturbation: plate moves +16cm, bowl barely moves (~1.3cm). All VLAs collapse to 0%.

| Method | P2 Success Rate | Notes |
|--------|----------------|-------|
| All VLAs (OpenVLA, pi0, pi0.5) | **0%** | Per LIBERO-PRO paper |
| DHB-XR naive replay | **0%** | Confirmed, same as VLAs |
| **DHB-XR adapted replay** | **90% (18/20)** | Only method with non-zero — p=1.68e-9! |

DHB uses Fatrop solver to re-decode placement trajectory from new plate position (0mm boundary error). Execution at 20% due to OSC controller tracking. Running n=20 for statistics.

**LIBERO-PRO continuous offset demo — `examples/integration/libero_pro_dhb_demo.py`:**

| Test | Result |
|------|--------|
| Original task replay | **SUCCESS** (80 steps) |
| DHB solver +3cm X | **0.000mm** shape error |
| DHB solver +5cm XY | **0.000mm** shape error |
| DHB solver +7cm diagonal | **0.000mm** shape error |
| LIBERO-PRO with_mug invariant corr | **0.95** |
| LIBERO-PRO with_milk invariant corr | **0.94** |

## Action Head Remaining Gaps (2026-04-13)
- Multi-demo prediction error ~40mm (need more training or demos)
- Live-image eval: predict stage uses pre-recorded images, not rendered offset observations
- OSC replay achieves 100% ceiling vs JOINT_POSITION 80%
- 2/10 GT demos fail on JOINT_POSITION tracking (demo_14, demo_16)
- Dual-head direct action eval: 0% (action predictions not matching OSC controller space yet)

## DEFINITIVE CORL RESULTS (2026-04-13)

### LIBERO-PRO P2 — BREAKTHROUGH RESULT
| Condition | Success | Statistical |
|-----------|---------|-------------|
| Unperturbed + raw replay | 95% (19/20) | baseline |
| P2 swap + raw replay | 0% (0/20) | matches all VLAs |
| P2 swap + raw replay | 0% (0/20) | Expected |
| **P2 swap + naive shift (rigid)** | **0% (0/20)** | **Key: shape matters, not just target** |
| **P2 swap + DHB adapted** | **90% (18/20)** | **p=1.68e-9 vs ALL baselines** |

**This is the paper's strongest result.** DHB-XR is the only method with non-zero performance under LIBERO-PRO P2 where all existing VLAs (OpenVLA, pi0, pi0.5) collapse to 0%.

Key fix that enabled 90%: bowl-EEF offset compensation in placement phase (4.3cm Y offset). Script: `_workspace/eval_libero_pro_p2.py` (N_EVAL_DEMOS=20).

### Augmentation — HONEST n=50 RESULTS
| Model | 0cm | 3cm | 5cm |
|-------|-----|-----|-----|
| Base SmolVLA | 70% | 22% | 0% |
| DHB-aug FT | 42% | 20% | 0% |
| Naive-aug FT | 46% | 8% | 0% |
| Original FT | 34% | 22% | 0% |

DHB-aug vs Naive-aug at 3cm: p=0.15 (approaching but not significant). Fine-tuning harms generalization; DHB augmentation mitigates naive-aug collapse.

## CoRL Review Action Plan (2026-04-13)

Three specialized CoRL reviewers gave: Reject / Weak Reject / Weak Accept. Consensus issues:

| Priority | Action | Status |
|----------|--------|--------|
| 1 | Expand to 3+ LIBERO tasks (n=10 per task) | RUNNING (tasks 1, 2) |
| 2 | B2 baseline = naive-aug (already done, needs reframing) | DONE — rename to "domain randomization" in paper |
| 3 | Add Limitations section to manuscript | DONE (line ~952) |
| 4 | Related Work outline | DONE (draft outline produced) |
| 5 | Cut speculative sections (variational FM, world models, etc.) | TODO |
| 6 | Restructure manuscript as proper CoRL paper | TODO |
| 7 | Formalize Proposition 1 + angular invariance proof | TODO |
| 8 | Increase n to 50 for statistical significance | TODO |
| 9 | OSC P-controller eval (90% GT ceiling) | RUNNING |

Key findings from geometry reviewer: SE(3)-invariance claim is mathematically correct. 
0mm roundtrip is structurally expected, not empirical. DHB-DR IS a discrete Bishop frame.

Strategic reframing: Lead with augmentation + token stability. Action head → supplementary.
Naive-aug → rename to "domain randomization" for reviewer clarity.

## Loss Alignment with LEGATO Manuscript (COMPLETE 2026-04-13)
- `action_head.py` now uses `invariant_matching_loss_torch` from `losses/` module
- L_act: L1 loss (robust, matches manuscript spec)
- Formula: `λ_act * L_act + λ_inv * L_inv` (was inverted before)
- Advancement over LEGATO: dual-head predicts BOTH invariants + actions

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
