# Compression Story — Preserved for Future Use

**Date preserved:** 2026-04-14  
**Source:** `manuscript/corl2026_varB.tex` (merged from `paper/corl-2026` branch)

---

## The Pitch

> *"Tokenizing in SE(3)-invariant space simultaneously improves compression and adds geometric stability."*

DHB magnitude invariants fed into FAST (DCT+BPE) yield **49.8× compression** — 2.4× better than FAST on raw coordinates — with **0.0% token change** under arbitrary SE(3) transforms.

An OAT-style register encoder on the same invariants achieves k=1 prefix MSE = 0.00005, meaning **a single token reconstructs the full motion shape**.

---

## Key Numbers (from `results/exp_fast_invariants.json`, `results/exp3_token_stability.json`)

| Method | Compression | Token stability (SE3) |
|--------|------------|----------------------|
| FAST on raw coords | 20.7× | 93.9% change |
| FAST on DHB magnitudes | **49.8×** | **0.0% change** |
| OAT on DHB invariants | T/K per step | 0.0% change |

---

## Why It's Compelling

1. **Free data augmentation**: invariant targets don't change under SE(3) transforms → unlimited spatial augmentation with zero label recomputation
2. **Better compression than FAST on raw**: invariants are smoother signals (fewer DCT coefficients needed)
3. **Drop-in layer**: any existing tokenizer (FAST, OAT, VQ-VAE) inherits geometric stability by operating on invariant input

---

## What It Needs to Become the Main Story

- The current primary result (P2 90%, p=1.68e-9) is stronger for CoRL because it's a concrete robot task result
- The compression story needs a **downstream task** that benefits from it: e.g. "VLA trained on DHB tokens achieves X% at offset Y where FAST-token VLA gets 0%"
- Multi-task invariant learning (0.023 MSE across 10 LIBERO tasks) is the supporting experiment
- Consider as a companion paper or as Section 4 of the main paper once the robot result is the lead

---

## Relevant Files

- `manuscript/corl2026_varB.tex` — full varB draft with this framing
- `manuscript/intro_varB.tex` — intro for this framing
- `manuscript/figures/fig1_token_stability_ratedistortion.pdf` — rate-distortion figure
- `results/exp3_token_stability.json` — 0.0% stability result
- `results/exp_fast_invariants.json` — 49.8× compression result
- `notes/supplementary/10_triple_invariant_tokenizer.md` — triple-invariant extension
