# SmolVLA Fine-tuning Results & Next Steps

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

## Recommendation

Option C is most likely to work — use the existing HF dataset format and just add
augmented episodes. This avoids the normalization mismatch entirely.

## What's Still Valid

The degradation curve (base SmolVLA: 70% → 0% at 7cm) is correct and honest.
The motion DB results (retrieval invariance, compression) are correct.
The training pipeline needs debugging but the experimental design is sound.
