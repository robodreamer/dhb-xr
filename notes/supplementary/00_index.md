# DHB-XR CoRL 2026 — Supplementary Notes Index

**Branch**: `paper/corl-2026` | **Worktree**: `dhb_xr_paper/` | **Updated**: 2026-04-09

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

---

## Key Results Summary (for quick reference)

| Metric | Value | Experiment |
|--------|-------|-----------|
| DHB geometric error at 0–20cm | **0.0 mm** (mean±0) | 10 tasks, 8 dirs, 6 mags |
| Naive replay error at 5/10/15/20cm | 50/100/150/200 mm | Same |
| DHB-DR vs DHB-TI difference | 0.0 mm both | Confirmed equivalent on LIBERO-Spatial |
| Token stability (DHB magnitude) | **0.0%** change | 40 SE(3) transforms |
| Token stability (raw/FAST) | **99.1%** change | Same |
| OAT prefix k=1 MSE | **0.0013** | Synthetic benchmark |
| FAST compression | **27.8×** (14 tokens) | Invariant sequences |
| OAT compression | **6.2×** (8 tokens) | Invariant sequences |
| Sample efficiency at 10cm | **0mm with N=1 demo** | Geometric proxy |
| Pipeline timing | 66ms encode + ≤7ms adapt + 6ms IK | Measured |
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
