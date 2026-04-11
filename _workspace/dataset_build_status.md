# Dataset Build Status

## Background Processes Running
- DHB augmented: ~215/450 episodes (pid in `ps aux | grep build_augmented | grep dhb`)
- Naive augmented: ~218/450 episodes (pid in `ps aux | grep build_augmented | grep naive`)
- ETA: ~7 min each

## Completed
- Original: 50 episodes, 3999 frames, 592 MB -> `_workspace/lerobot_dataset_original_task0/`

## After Datasets Complete
Run training:
```bash
# Convert npz -> LeRobot v3.0 + train with LoRA
~/miniforge3/envs/smolvla/bin/python _workspace/train_smolvla_augmented.py --mode original --steps 5000
~/miniforge3/envs/smolvla/bin/python _workspace/train_smolvla_augmented.py --mode dhb --steps 5000
~/miniforge3/envs/smolvla/bin/python _workspace/train_smolvla_augmented.py --mode naive --steps 5000
```

Then evaluate:
```bash
# Eval each fine-tuned model at offsets 0,3,5,7 with 20 seeds
MUJOCO_GL=egl ~/miniforge3/envs/smolvla/bin/python _workspace/eval_degradation_curve.py \
    --model _workspace/smolvla_finetuned_original/checkpoint --seeds 20
```
