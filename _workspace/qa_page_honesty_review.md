# QA Critic: Project Page Honesty Review (2026-04-10)

## CRITICAL Issues (must fix before any submission)

### CRITICAL-1: Apples-to-oranges comparison table (lines 1209-1282)
SmolVLA at 67%/0% (closed-loop VLA) vs DHB-XR at 100% (open-loop geometric replay).
**Action:** Remove entirely. Replace with degradation curve once 20-seed data exists.

### CRITICAL-2: "15 of 25 tested offset configurations achieve 100% task success" (lines 1269, 1379, 1389)
Based on only 3-6 trials per condition. Inconsistency between "25" and "22".
**Action:** Remove all claims until 20-seed experiments run.

### CRITICAL-3: SmolVLA "67%" at 0cm (line 1226)
From only 3-6 trials. Confidence interval absurdly wide.
**Action:** Remove until 20+ seeds run.

### CRITICAL-4: SmolVLA "0%" at offsets with n<10 (lines 1232-1254)
A single success would change 0% to 17-33%.
**Action:** Remove until 20-seed degradation curve complete.

## IMPORTANT Issues

### IMPORTANT-1: Hero stat "0.0 mm Goal error" (line 548)
Reader interprets as VLA task success. It's geometric reconstruction.
**Action:** Relabel to "0.0 mm geometric reconstruction error"

### IMPORTANT-2: "Spatial Generalization" column (line 888)
Implies task-level generalization. It's geometric accuracy.
**Action:** Rename to "EE Reconstruction Error"

### IMPORTANT-3: Learnability tab placeholder (line 1053)
Reuses perturbation sweep image.
**Action:** Create proper figure or remove tab.

### IMPORTANT-4: "StarVLA-DHB" reference (line 1280)
System doesn't exist yet.
**Action:** Remove. Replace with honest planned experiments.

### IMPORTANT-5: "due to token instability" causal claim (line 1265)
Unsupported causal attribution.
**Action:** Remove causal claim or provide token edit distance evidence.

## What's Valid and Compelling (KEEP)
1. SE(3) invariance proof (mathematical guarantee)
2. Compression results (49.8x, tokenizer comparison table)
3. Token stability visualization (FAST vs DHB)
4. Geometric adaptation demos (DHB vs naive replay, NOT vs SmolVLA)
5. Interactive Three.js demo
6. Noise robustness comparison
7. Runtime/timing facts
8. Abstract and contributions list
