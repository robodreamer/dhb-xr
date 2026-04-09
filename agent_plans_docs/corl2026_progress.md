# CoRL 2026 Progress Tracker

**Branch**: `paper/corl-2026`  
**Worktree**: `/home/andypark/Projects/repos/dhb_xr_paper`  
**Target deadline**: CoRL 2026 (~June–July 2026)  
**Started**: 2026-04-09  
**Last updated**: 2026-04-09

---

## Phase 0: Setup

- [x] Git worktree `paper/corl-2026` created at `dhb_xr_paper/`
- [x] CoRL LaTeX skeleton (varA/B/C + shared includes)
- [x] Vendor clone: AllenAI vla-evaluation-harness → `vendor/vla-evaluation-harness/`
- [x] Vendor clone: StarVLA → `vendor/starVLA/`
- [x] Baseline tests: 218 passed, 0 failures

## Phase 1: Literature Survey

- [x] Survey 7 papers from awesome-vla-wam
- [x] Gap table (14-18% degradation across all SOTA)
- [x] `agent_plans_docs/literature_survey_corl2026.md`
- [x] `manuscript/related_work.tex` (~300 words, EquivAct/RISE added, OAT attribution)

## Phase 2: Benchmark Integration

- [x] `src/dhb_xr/integration/vla/harness_server.py` (AllenAI DHBModelServer)
- [x] `configs/model_servers/dhbxr.yaml`
- [x] `vendor/starVLA/.../DHBActionHeader.py` + `WM4A_DHB.py` framework
- [x] `vendor/starVLA/configs/libero_dhbxr.yaml`
- [x] `libero_pro_dhb_demo.py --perturbation-sweep` + JSON output
- [x] `examples/integration/panda_dhb_sim_poc.py` (0.00mm geometric validation)
- [x] `examples/integration/libero_to_lerobot.py` (710 lines, unblocks training)
- [x] `examples/integration/precompute_dhb_invariants.py`

## Phase 3: Experiments

- [x] Exp 1: Geometric sweep — 10 tasks × 8 dirs × 6 mags (0–20cm): 0.0mm ± 0 for DHB-DR + DHB-TI
- [~] Exp 2: StarVLA ablation — infrastructure complete; training blocked on GPU + data pipeline
  - `WM4A_DHB.py`, `DHBActionHeader.py`, `starvla_dhb_training.py` all validated ✅
  - `libero_to_lerobot.py` adapter complete ✅ (was the blocker)
  - Training command ready — need GPU runtime
- [x] Exp 3: Token stability (0.0% DHB vs 99.1% raw, n=40), rate-distortion (7 tokenizers)
- [x] Exp 4: Sample efficiency (0mm with N=1 demo, geometric proxy)
- [ ] Exp 5: RoboCASA — pending harness install + trained StarVLA-DHB checkpoint

## Phase 4: Paper Drafts

- [x] `manuscript/methods.tex` (condensed from main.tex)
- [x] `manuscript/experiments.tex` (10-task sweep, token stability, sample efficiency)
- [x] `manuscript/related_work.tex` (EquivAct, RISE, EquivAct, OAT attribution)
- [x] `manuscript/discussion.tex` (limitations, future work)
- [x] Three draft variants: varA v0.5 (7pp), varB v0.5 (7pp), varC v0.5 (7pp)
- [x] Compiled with official CoRL 2026 .sty + bibtex
- [x] Figures: fig1-8 (stability, sweep, efficiency, pipeline, 6D adaptation, timing)
- [x] Videos: arc_obstacle_adaptation.{gif,mp4}, trajectory_adaptation.{gif,mp4}, LIBERO sim videos

## Phase 5: Autoresearch / Writing Loop

- [x] `autoresearch:reason` adversarial critique run → C1 reproducibility fix applied
- [x] Three draft variants as basis for framing selection
- [ ] `autoresearch:reason` second pass on updated drafts (post-experiment)
- [ ] Pick strongest draft (recommendation: varA for CoRL, varC for journal)
- [ ] `ralph-loop` regression check

## Supplementary Notes

- [x] `notes/supplementary/00_index.md` — index + key results table
- [x] `notes/supplementary/01_extended_methods.md` (689 lines)
- [x] `notes/supplementary/02_extended_results.md`
- [x] `notes/supplementary/03_simulation_analysis.md` (OSC scaling diagnosis)
- [x] `notes/supplementary/04_media_guide.md` (402 lines)
- [x] `notes/supplementary/05_future_work_technical.md` (637 lines)
- [x] `notes/supplementary/06_framing_and_positioning.md` (optimization vs NN, community)
- [x] `notes/supplementary/07_generalization_beyond_shape.md` (complementarity argument)

## Verification

- [x] `pytest tests/test_invariance.py tests/test_oat_tokenizer.py` — passing
- [x] `pdflatex manuscript/corl2026_varA.tex` — 7 pages, compiles cleanly
- [x] `libero_pro_dhb_demo.py --perturbation-sweep` — 0mm DHB (geometric)
- [x] `panda_dhb_sim_poc.py` — 0.00mm pipeline validated
- [ ] `vla-eval serve/run` — pending harness docker setup
- [ ] `libero_pro_dhb_demo.py` task success > 50% at 10cm — pending OSC fix + policy training
- [ ] `panda_dhb_sim_poc.py --save-video` MuJoCo execute — pending OSC PD controller

---

## Blockers & Next Actions

| Priority | Blocker | Resolution |
|----------|---------|-----------|
| HIGH | StarVLA-DHB policy training | GPU needed; adapter ready; training command ready |
| HIGH | MuJoCo task success rate | Implement waypoint PD controller (see notes/supplementary/03) |
| MED | RoboCaSA eval | Needs trained checkpoint + harness docker |
| LOW | `vla-eval` serve/run | Docker + harness install |

## Notes / Key Decisions

- **Reproducibility**: 0.0% token stability requires `use_default_initial_frames=False`; decode pipeline uses default. Both are correct but for different purposes.
- **Simulation success rates**: `--` in paper due to OSC variable scaling (20×–577×). Geometric proxy (0mm) is the valid primary evidence.
- **Draft recommendation**: varA (spatial gen framing) best for CoRL practitioners; varC (invariant learning) best for journal.
- **LIBERO adapter**: Uses LeRobot v2.0 parquet format with 5 metadata files. Optional DHB invariant columns.
