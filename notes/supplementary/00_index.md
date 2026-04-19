# DHB-XR CoRL 2026 — Supplementary Notes Index

**Branch**: `paper/corl-2026` | **Worktree**: `dhb_xr_paper/` | **Updated**: 2026-04-19

These notes capture everything that doesn't fit in the 8-page CoRL format.
No page limit. Meant for internal use, future extensions, and journal version.

---

## Files

| File | Contents |
|------|----------|
| [01_extended_methods.md](01_extended_methods.md) | Full math, DHB-DR/QR/TI derivations, OAT architecture, Fatrop formulation |
| [02_extended_results.md](02_extended_results.md) | All experiment data — full tables, methodology, statistical details |
| [03_simulation_analysis.md](03_simulation_analysis.md) | LIBERO simulation investigation, OSC action scaling issue, path to fix |
| [04_media_guide.md](04_media_guide.md) | All videos, figures, frames — what they show and how to regenerate |
| [05_future_work_technical.md](05_future_work_technical.md) | Detailed technical roadmap: training, simulation, cross-embodiment |
| [06_framing_and_positioning.md](06_framing_and_positioning.md) | Community positioning, objection handling, comparison tables |
| [07_generalization_beyond_shape.md](07_generalization_beyond_shape.md) | Why DHB-XR is robust to unexpected situations; complementarity with learned methods; extensions |
| [08_sensitivity_deep_dive.md](08_sensitivity_deep_dive.md) | Position vs orientation sensitivity, geodesic loss, strategy sweep, decoder-intrinsic insight |
| [09_next_iteration_plan.md](09_next_iteration_plan.md) | Prioritized roadmap with all 6 steps completed + remaining work |
| [10_triple_invariant_tokenizer.md](10_triple_invariant_tokenizer.md) | FAST+DHB and OAT on magnitude invariants: 49.8x compression, 0% instability |
| [11_session_summary.md](11_session_summary.md) | Complete session summary: all experiments, results, insights, remaining work |
| [12_smolvla_dhb_action_head_summary.md](12_smolvla_dhb_action_head_summary.md) | SmolVLA DHB action-head validation summary |
| [13_smolvla_results_discussion_draft.md](13_smolvla_results_discussion_draft.md) | Paper-style results/discussion subsection draft |
| [14_dhb_action_head_internal_note.md](14_dhb_action_head_internal_note.md) | Internal explainer note for the DHB action head itself |
| [15_qwen_dhb_integration_internal_note.md](15_qwen_dhb_integration_internal_note.md) | Internal Q&A note: how Qwen and DHB-XR are combined, what is validated now, and what is not yet proven |
| [16_starvla_dhb_gap_closure_plan.md](16_starvla_dhb_gap_closure_plan.md) | Phased plan for closing the remaining StarVLA+DHB execution, comparison, and benchmark gaps |
| [17_dhbxr_scaling_story_note.md](17_dhbxr_scaling_story_note.md) | Internal framing note for the larger DHB-XR low-data scaling, augmentation, and cross-embodiment story |

---

## Key Results Summary (for quick reference)

| Metric | Value | Experiment |
|--------|-------|-----------|
| DHB geometric error at 0-20cm | **0.0 mm** (mean+/-0) | 10 tasks, 8 dirs, 6 mags |
| Naive replay error at 5/10/15/20cm | 50/100/150/200 mm | Same |
| DHB-DR vs DHB-TI difference | 0.0 mm both | Confirmed equivalent on LIBERO-Spatial |
| Token stability (DHB magnitude) | **0.0%** change | 40 SE(3) transforms |
| Token stability (raw/FAST) | **93.9-99.1%** change | Same |
| FAST+DHB magnitude compression | **49.8x** (8 tokens, 0% instability) | exp_invariant_subset_compression |
| OAT on magnitudes k=1 MSE | **0.00005** | exp_magnitude_oat |
| OAT on full invariants k=1 MSE | **0.0054** | exp_oat_real_improved |
| FAST compression (raw) | **20.4x** (14 tokens) | Invariant sequences |
| Multi-task avg MSE | **0.023** (10 tasks, single model) | exp_multi_task |
| Blended controller success | **100%** at 0-5cm | exp_blended_success |
| SE(3) augmentation improvement | **1.5%** eval MSE | exp_se3_augmentation |
| Geodesic strategy sweep | **1.121** (all 5 identical) | exp_geodesic_strategies |
| Sample efficiency at 10cm | **0mm with N=1 demo** | Geometric proxy |
| Pipeline timing | 66ms encode + <=7ms adapt + 6ms IK | Measured |
| StarVLA-FAST in-dist | 97.8% | From paper |
| StarVLA-MLP in-dist | 97.1% | From paper |
| StarVLA-DHB in-dist | **pending** | Training blocked on LeRobot adapter |

---

## Important Reproducibility Notes

1. **Token stability 0.0%** requires `use_default_initial_frames=False` in `encode_dhb_dr`.
   Under the default (`True`), the 95.7% token change is observed.
   The decode pipeline uses the default (no flag) and achieves 0mm geometric error.
   These are two different properties: decode uses default frames; stability measurement uses non-default.

2. **Simulation success rates** are `--` because:
   - LIBERO OSC action scaling is variable (20×–577×, not constant)
   - Need Fatrop + waypoint PD controller for correct DHB execution
   - See [03_simulation_analysis.md](03_simulation_analysis.md)

3. **Video quality**: Real videos are 16-41KB (long-named files in output/videos/).
   Small (<3KB) files are black/empty and have been removed.
   The `trajectory_adaptation.{gif,mp4}` are matplotlib-rendered animations showing the 3D geometric adaptation.
