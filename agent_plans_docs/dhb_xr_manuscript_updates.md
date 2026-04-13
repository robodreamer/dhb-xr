# DHB-XR CoRL 2026 Manuscript Update Plan

**Date:** 2026-04-12
**Status:** Ready to execute
**Goal:** Update CoRL paper with actual experimental results

## Priority Updates

### 1. New Experiment: VLA Degradation Curve (Section 4.1)
- SmolVLA 500M on LIBERO-Spatial task 0
- Translation: 70%/55%/55%/25%/10%/0%/0% at 0/1/2/3/5/7/10cm (n=20)
- Rotation: 70%/70%/70%/20%/0%/0% at 0/5/10/15/20/30° (n=10)
- Figure: `combined_degradation.png` (already generated)

### 2. Updated Table 1 (Spatial Generalization)
Replace cited approximations with measured data:
| Model | 0cm | 3cm | 5cm |
|-------|-----|-----|-----|
| SmolVLA (baseline) | 70% | 25% | 10% |
| SmolVLA + Original FT | 50% | 10% | 0% |
| SmolVLA + DHB-aug FT | 35% | **15%** | 0% |
| SmolVLA + Naive-aug FT | 35% | 0% | 0% |

### 3. New Experiment: DHB Augmentation (Section 4.2)
- 450 augmented demos from 50 originals in 18.8s
- DHB-aug is only model with non-zero 3cm success (15%)
- Naive-aug proves visual diversity without consistency fails

### 4. New Experiment: Motion Database (Section 4.3)
- 500 demos encoded in 19.5s
- Retrieval perfectly offset-invariant (12% at ALL offsets)
- Position: 98% → 12% at 20cm
- Self-retrieval: DHB 100% vs position 11%
- Compression: 100x at k=5

### 5. Language Tightening
- Teaser: "14-18% degradation" → "70% to 0% at 7cm"
- Conclusion: Clarify geometric vs policy success
- Remove "pending" placeholders

## Files to Modify
- `manuscript/experiments.tex` — add degradation curve, augmentation, motion DB
- `manuscript/main.tex` / `corl2026_varB.tex` — update Table 1
- `manuscript/discussion.tex` — add motion DB discussion
- `manuscript/figures/` — add degradation_curve, augmentation_comparison figures
