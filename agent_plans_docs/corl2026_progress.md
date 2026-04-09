# CoRL 2026 Progress Tracker

**Branch**: `paper/corl-2026`  
**Worktree**: `/home/andypark/Projects/repos/dhb_xr_paper`  
**Target deadline**: CoRL 2026 (~June–July 2026)  
**Started**: 2026-04-09

---

## Phase 0: Setup

- [x] Git worktree `paper/corl-2026` created at `dhb_xr_paper/`
- [ ] CoRL LaTeX skeleton (`manuscript/corl2026_main.tex` + shared includes + 3 variant files)
- [ ] Vendor clone: AllenAI vla-evaluation-harness → `vendor/vla-evaluation-harness/`
- [ ] Vendor clone: StarVLA → `vendor/starVLA/`
- [ ] Baseline tests pass

## Phase 1: Literature Survey

- [x] Survey papers from awesome-vla-wam + priority list (7/7 papers)
- [x] Gap table written (all 14-18% degradation quantified)
- [x] `agent_plans_docs/literature_survey_corl2026.md` complete
- [x] `manuscript/related_work.tex` drafted (~305 words, ≤0.5 page CoRL format)

## Phase 2: Benchmark Integration

- [x] `src/dhb_xr/integration/vla/harness_server.py` (AllenAI DHBModelServer)
- [x] `configs/model_servers/dhbxr.yaml`
- [x] `vendor/starVLA/.../DHBActionHeader.py` (OAT-style, OATTokenizer integration)
- [ ] `starVLA/configs/libero_dhbxr.yaml`
- [x] `libero_pro_dhb_demo.py --perturbation-sweep` extension (+ JSON output)
- [x] `examples/integration/panda_dhb_sim_poc.py` (0.00mm vs 100mm naive — validated)

## Phase 3: Experiments

- [x] Exp 1: Spatial generalization sweep (LIBERO-PRO, geometric)
  - 5 tasks × 5 levels × 3 dirs: DHB=0mm, naive=proportional to offset ✅
  - results/libero_pro_sweep.json complete
  - MuJoCo success rate: pending (requires StarVLA training)
- [~] Exp 2: StarVLA ablation — infrastructure ready, training pending
  - `WM4A_DHB.py` framework created + `@FRAMEWORK_REGISTRY.register("WM4A_DHB")` ✅
  - `precompute_dhb_invariants.py` validated (dry-run OK) ✅
  - `starvla_dhb_training.py` validation: forward pass loss=0.977 ✅
  - LIBERO data loads at N=5/10/20/50 (6k-62k transitions) ✅
  - **Blocker**: LIBERO→LeRobot format adapter needed for StarVLA dataloader
  - Training command ready (see starvla_dhb_training.py)
- [x] Exp 3: Tokenization analysis — token stability (0% vs 99.1%) + rate-distortion
  - `results/exp3_token_stability.json`, `results/exp3_tokenization_analysis.json`
- [x] Exp 4: Sample efficiency (geometric proxy: 0mm with N=1 demo, flat)
  - `results/exp4_sample_efficiency.json`
- [ ] Exp 5: RoboCASA (harness — requires harness install)

## Phase 4: Paper Drafts

- [ ] `manuscript/methods.tex` (condense from main.tex)
- [ ] `manuscript/experiments.tex`
- [ ] `manuscript/related_work.tex`
- [ ] `manuscript/discussion.tex`
- [x] `manuscript/corl2026_varA.tex` → `output/dhb_xr_corl2026_varA_v0.1_20260409.pdf` (193KB, 8pp)
- [x] `manuscript/corl2026_varB.tex` → `output/dhb_xr_corl2026_varB_v0.1_20260409.pdf` (193KB, 8pp)
- [x] `manuscript/corl2026_varC.tex` → `output/dhb_xr_corl2026_varC_v0.1_20260409.pdf` (201KB, 8pp)
- [x] Compiled with official CoRL 2026 .sty + bibtex (3-pass, non-fatal warnings only)
- [x] Figures generated: output/figures/ (pipeline, stability, sweep, sample efficiency)

## Phase 5: Autoresearch Loop

- [ ] Experiment loop converged (≥3 saturated iterations)
- [ ] Writing loop converged (no critical issues in 3 rounds)
- [ ] Strongest draft selected
- [ ] `ralph-loop` regression check passing

## Verification

- [ ] `pytest tests/test_invariance.py tests/test_oat_tokenizer.py tests/test_fatrop_solver.py`
- [ ] `vla-eval serve` starts cleanly
- [ ] `vla-eval run --config configs/libero_spatial.yaml` produces JSON
- [ ] `libero_pro_dhb_demo.py --perturbation-sweep` > 50% at 10cm
- [ ] `panda_dhb_sim_poc.py --perturbation-magnitude 0.10 --save-video` succeeds
- [ ] `pdflatex manuscript/corl2026_varA.tex` compiles cleanly

---

## Notes / Blockers

_(running log — add entries as work progresses)_

- 2026-04-09: Worktree created, Phase 0 started
