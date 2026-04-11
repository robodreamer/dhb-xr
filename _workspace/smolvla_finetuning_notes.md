# SmolVLA Fine-tuning Notes

## Training Command
```bash
lerobot-train \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --dataset.repo_id=<local_or_hf_repo> \
  --env.type=libero --env.task=libero_spatial
```

## Key Config (from configuration_smolvla.py)
- LR: 1e-4, AdamW, betas=(0.9, 0.95)
- Warmup: 1,000 steps, cosine decay to 2.5e-6 over 30,000 steps
- Batch size: 8, chunk size: 50
- Image resize: 512x512 (padded)
- `freeze_vision_encoder=True, train_expert_only=True, train_state_proj=True`
- LoRA: `--peft.method_type=LORA --peft.r=16`

## Data Format
LeRobot v3.0: Parquet metadata + images/videos per episode.
Must convert LIBERO HDF5 -> LeRobot format using `LeRobotDataset` write mode.

## Three Fine-tuning Runs Needed
1. Original 50 demos only (baseline)
2. 50 + 450 DHB-augmented demos
3. 50 + 450 naive-augmented demos

## Data Conversion Steps
1. Read LIBERO HDF5 demos
2. For each timestep: extract actions(7D), images(agentview, eye_in_hand), state(eef_pos, eef_quat, gripper)
3. Call `dataset.add_frame()` per timestep
4. Call `dataset.save_episode()` per demo
5. For augmented demos: images from original demo, actions from augmented

## Important: Augmented images
DHB augmentation only changes the action trajectory. The images remain from the
original demo at 0cm offset. This is a limitation - the VLA will see 0cm images
but be supervised with offset-adapted actions. This is fine for training the
action representation, but may confuse the visual encoder.

Alternative: Use LIBERO's programmatic offset to render images at each offset
position. This would require running the sim for each augmented demo.
