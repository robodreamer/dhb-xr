# SmolVLA Fine-tuning Comparison Notes

## Training Results (5000 steps each)

| Model | Dataset Size | Final Loss | Epochs | Steps/Epoch |
|-------|-------------|------------|--------|-------------|
| Original | 50 eps (4K frames) | 0.39 | 4.0 | 1,250 |
| DHB-aug | 500 eps (45K frames) | 0.47 | 0.4 | 12,500 |
| Naive-aug | 500 eps (43K frames) | TBD | ~0.4 | ~12,000 |

## Fair Comparison Issue

The augmented models (DHB, naive) only completed 0.4 epochs through their larger datasets.
The original model completed 4.0 epochs. This means:
- Original saw each demo ~4 times
- DHB/naive saw each demo only ~0.4 times

For a fair comparison, we should either:
1. **Scale steps by dataset size**: 5000 × (500/50) = 50,000 steps for augmented models
2. **Match epochs**: Train all to 4 epochs → 50,000 steps for augmented
3. **Keep same steps, compare loss/epoch curves** (what we have now)

Option 3 is actually interesting: if DHB-aug at 0.4 epochs already shows better spatial
generalization than original at 4.0 epochs, that's a strong result. It means the
augmented data is more efficient — you need fewer passes through more diverse data.

## Next: Evaluation

Run the degradation curve evaluation for each checkpoint:
```bash
MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python _workspace/eval_degradation_curve.py \
    --seeds 20 --axis Y
```

But we need to modify eval to load from checkpoint, not from HF.
