# SmolVLA Fine-tuning Results & Next Steps (Updated 2026-04-11)

## Results (all 0% — training pipeline issue)

| Model | 0cm | 3cm | 5cm | 7cm |
|-------|-----|-----|-----|-----|
| Base SmolVLA (no LoRA) | **70%** | 55% | 55% | 25% |
| Original fine-tuned | 0% | 0% | 0% | 0% |
| DHB-augmented fine-tuned | 0% | 0% | 0% | 0% |
| Naive-augmented fine-tuned | 0% | 0% | 10% | 0% |

## Root Cause: Normalization Mismatch

The LoRA fine-tuning broke the base model because:

1. **Preprocessor uses HF normalization stats** — our `load_smolvla()` loads preprocessor
   from `HuggingFaceVLA/smolvla_libero` which has mean/std computed from the original
   libero_10 training data

2. **Our custom dataset has different stats** — actions/states from our 50 task-0 demos
   have different distributions than the full libero_10 data

3. **LoRA adapted to mismatched normalized inputs** — the model learned to compensate
   for the normalization mismatch instead of learning the actual task

## Fix Options (for next session)

### Option A: Compute custom normalization stats
- Compute mean/std from our custom dataset
- Override the preprocessor's normalization with these stats
- Retrain with correct normalization

### Option B: Use the official lerobot training pipeline end-to-end
- Use `lerobot-train` directly with our dataset repo
- Let lerobot compute normalization stats from the new data
- This requires proper dataset registration

### Option C: Use the HF libero_spatial dataset directly
- Don't create custom datasets
- Fine-tune on the existing `HuggingFaceVLA/libero_spatial` with task 0 only
- Add DHB-augmented episodes to this existing dataset
- This ensures normalization compatibility

### Option D: Skip LoRA, train full model
- LoRA may not have enough capacity to learn offset-aware actions
- Full fine-tuning with small LR on the action head only

## RESOLVED: Normalization Mismatch (2026-04-11)

Fixed by loading preprocessors from checkpoint path instead of base model.
Original fine-tuned now gets 80% at 0cm (up from 0%).

## RESOLVED: DHB Action Scale Bug (2026-04-11)

DHB-augmented actions were in raw Cartesian meters (max ~0.01) instead of
OSC-normalized space (max ~0.94). For pure translation, original actions
are already correct by SE(3) invariance. Fixed: DHB and naive now use
identical original actions, differing only in rendered images at offsets.

## Current Status

- Original 20K training RUNNING (GPU)
- DHB dataset rebuilding with fixed actions (CPU)
- After DHB dataset done: train DHB 20K, then naive 20K, then evaluate all 3

## Results After Normalization Fix (5K steps)

| Model | 0cm | 3cm | 5cm | 7cm |
|-------|-----|-----|-----|-----|
| Base SmolVLA | 70% | 25% | 10% | 0% |
| Original FT (5K, 4 epochs) | 80% | 20% | 0% | 0% |
| DHB-aug FT (5K, 0.4 epochs, BAD ACTIONS) | 0% | 0% | 0% | 0% |
| Naive-aug FT (5K, 0.4 epochs) | 40% | 10% | 0% | 0% |

Next: retrain with fixed actions + 20K steps for fair comparison.
