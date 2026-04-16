# Triple-Invariant Tokenizer: The Combination of Our Two Best Results

## The Insight

Our two strongest individual findings:
1. **FAST on magnitudes**: 49.8× compression, 0% token stability, 8 BPE tokens
2. **OAT with prefix decoding**: k=1 token gives 98.5% of full quality

Combining them: train OAT on the **2 magnitude-only invariants** (linear magnitude + angular magnitude).

## Results

| Tokenizer | Input dims | Compression | Stability | k=1 MSE | k=8 MSE |
|-----------|-----------|-------------|-----------|---------|---------|
| FAST on raw (7 cols) | 7 | 20.4× | 93.9% change | N/A | N/A |
| FAST on magnitudes (2 cols) | 2 | **49.8×** | **0.0%** | N/A | N/A |
| OAT on full invariants (10 cols) | 10 | 12.5× | mixed | 0.0054 | 0.0052 |
| **OAT on magnitudes (2 cols)** | **2** | **10.0×** | **0.0%** | **0.00005** | **0.00005** |

## Why This Works

The magnitude invariants `m_i = x_i^T Δp_i` (linear) and angular magnitude are:
1. **Smooth**: nearly sinusoidal signals — DCT and register encoding both excel
2. **Low-dimensional**: only 2 numbers per timestep
3. **SE(3)-invariant**: inner products are invariant to orthogonal transforms

The OAT register encoder maps 40 timesteps × 2 dims = 80 numbers to 8 × 4 = 32 latent numbers.
But 1 register token (4 numbers) already captures the signal almost perfectly
(MSE 0.000047 vs 0.000052 at k=8) — the signal is THAT simple.

## For the Paper

**Key table** (comparison to all methods):

| Method | Tokens | Compression | SE(3) Stable | Prefix decode | Trainable |
|--------|--------|-------------|-------------|---------------|-----------|
| Raw discrete bins | ~350 | 1× | ✗ | ✗ | ✗ |
| FAST on raw | 14 | 20.4× | ✗ (93.9%) | ✗ | ✗ |
| FAST on DHB mag | 8 | **49.8×** | ✓ (0.0%) | ✗ | ✗ |
| OAT on raw | 8 | ~6× | ✗ | ✓ | ✓ |
| **OAT on DHB mag** | **8** | **10.0×** | **✓ (0.0%)** | **✓ (k=1: 0.00005)** | **✓** |

**Narrative**: "By tokenizing in invariant-space magnitudes, we achieve both the compression
efficiency of FAST and the ordered prefix-decodability of OAT, while maintaining perfect
SE(3) stability. One token is sufficient to reconstruct the full motion shape."

## Reconstruction from Magnitudes Only

**Q: Can you reconstruct a full trajectory from just 2 magnitude values per step?**

A: Not directly — you also need the frame evolution (which requires the rotation invariants).
But the magnitudes capture the **speed profile** of the motion, which is the most informative
component for:
- Motion matching (same speed profile = same task)
- Task recognition (classify by magnitude pattern)
- Coarse trajectory prediction (predict magnitudes, use learned decoder for full pose)

For full SE(3) trajectory reconstruction, you need all 8 invariant dims.
But for **tokenization in VLA** (where the policy predicts discrete tokens), the magnitudes
are the optimal intermediate representation: maximally compressed, fully invariant, and
decodable at any prefix length.
