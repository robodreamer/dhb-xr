#!/usr/bin/env python
"""Sample Efficiency Sweep for CoRL 2026 Experiment 2.

This script:
1. Validates that demo subsets (N=5/10/20/50) can be loaded from the
   LIBERO-Spatial HDF5 files (no GPU, no model weights).
2. Prints the full StarVLA training commands for each demo count.
3. Serves as the thin wrapper that could be used to kick off a sweep
   (grid launch) once the WM4A_DHB framework class is implemented.

Usage:
    cd /home/andypark/Projects/repos/dhb_xr_paper
    pixi run python examples/integration/sample_efficiency_sweep.py

No GPU or model weights needed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ── Path setup ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DHB_SRC_PATH = _REPO_ROOT / "src"
if str(_DHB_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_DHB_SRC_PATH))

# ── Config ───────────────────────────────────────────────────────────────────
LIBERO_SPATIAL_DIR = Path("/home/andypark/Projects/data/libero/libero_spatial")
STARVLA_DIR = _REPO_ROOT / "vendor" / "starVLA"
DEMO_COUNTS = [5, 10, 20, 50]

# All 10 LIBERO-Spatial tasks
LIBERO_SPATIAL_TASKS = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate_demo.hdf5",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5",
]


# ── 1. LIBERO HDF5 subset loading validation ─────────────────────────────────

def _load_demo_subset(hdf5_path: Path, n_demos: int) -> List[Dict]:
    """Load the first n_demos demonstrations from a LIBERO HDF5 file.

    Returns a list of dicts with keys: actions, agentview_rgb, ee_pos, ee_ori.
    Does NOT load to GPU or create tensors — pure numpy.
    """
    import h5py
    import numpy as np

    demos = []
    with h5py.File(hdf5_path, "r") as f:
        data_group = f["data"]
        available = sorted(data_group.keys())  # demo_0, demo_1, …
        selected = available[:n_demos]
        for key in selected:
            demo = data_group[key]
            demos.append({
                "actions": demo["actions"][:],           # (T, 7)
                "agentview_rgb": demo["obs"]["agentview_rgb"][:],  # (T, 128, 128, 3)
                "ee_pos": demo["obs"]["ee_pos"][:],      # (T, 3)
                "ee_ori": demo["obs"]["ee_ori"][:],      # (T, 4) quaternion
            })
    return demos


def validate_demo_subsets() -> Dict[int, Dict]:
    """Validate that all N demo subsets load correctly across all 10 tasks."""
    import numpy as np

    print("\n=== 1. LIBERO-Spatial HDF5 subset loading validation ===")
    print(f"  Data directory: {LIBERO_SPATIAL_DIR}")

    if not LIBERO_SPATIAL_DIR.exists():
        print(f"  ERROR: Data directory not found: {LIBERO_SPATIAL_DIR}")
        return {}

    task_files = sorted(LIBERO_SPATIAL_DIR.glob("*.hdf5"))
    print(f"  Found {len(task_files)} task files")

    stats: Dict[int, Dict] = {}

    for n in DEMO_COUNTS:
        total_transitions = 0
        task_stats = []

        for hdf5_path in task_files:
            try:
                demos = _load_demo_subset(hdf5_path, n_demos=n)
                n_loaded = len(demos)
                transitions = sum(d["actions"].shape[0] for d in demos)
                action_shape = demos[0]["actions"].shape  # (T, 7)
                rgb_shape = demos[0]["agentview_rgb"].shape  # (T, 128, 128, 3)
                total_transitions += transitions
                task_stats.append({
                    "file": hdf5_path.name[:50] + "…",
                    "n_demos": n_loaded,
                    "transitions": transitions,
                    "action_shape": action_shape,
                    "rgb_shape": rgb_shape,
                })
            except Exception as e:
                print(f"  ERROR loading {hdf5_path.name}: {e}")

        stats[n] = {
            "total_tasks": len(task_stats),
            "total_transitions": total_transitions,
            "demos_per_task": n,
            "task_stats": task_stats,
        }

        first = task_stats[0] if task_stats else {}
        print(
            f"\n  N={n:2d} demos/task:"
            f" {len(task_stats):2d} tasks loaded,"
            f" {total_transitions:,} total transitions,"
            f" actions{first.get('action_shape', '?')},"
            f" rgb{first.get('rgb_shape', '?')}"
        )

    return stats


# ── 2. Training command generation ───────────────────────────────────────────

def generate_training_commands(demo_counts: List[int]) -> List[str]:
    """Generate accelerate launch commands for the sample efficiency sweep."""
    commands = []

    base_cmd = (
        f"cd {STARVLA_DIR}\n\n"
        "accelerate launch \\\n"
        "  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \\\n"
        "  --num_processes 4 \\\n"
        "  starVLA/training/train_starvla.py \\\n"
        "  --config_yaml configs/libero_dhbxr.yaml \\\n"
        "  --framework.name WM4A_DHB \\\n"
        "  --framework.action_model.use_oat_quantizer true \\\n"
        "  --datasets.vla_data.data_root_dir /home/andypark/Projects/data/libero \\\n"
        "  --datasets.vla_data.data_mix libero_spatial \\\n"
        "  --datasets.vla_data.per_device_batch_size 16 \\\n"
        "  --trainer.max_train_steps 50000 \\\n"
        "  --trainer.save_interval 5000 \\\n"
        "  --trainer.logging_frequency 100 \\\n"
        "  --trainer.eval_interval 1000 \\\n"
        "  --run_root_dir ./results/Checkpoints \\\n"
        "  --wandb_project dhb_xr_corl2026"
    )

    for n in demo_counts:
        cmd = (
            base_cmd
            + f" \\\n  --data.num_demos_per_task {n}"
            + f" \\\n  --run_id exp2_dhbxr_libero_spatial_{n}demo"
        )
        commands.append(cmd)

    return commands


def print_training_commands(demo_counts: List[int]) -> None:
    print("\n=== 2. Training commands for sample efficiency sweep ===")
    print(f"  Sweep: N = {demo_counts} demos per task")
    print(f"  Framework: WM4A_DHB (after patch — see starvla_dhb_training.py)")
    print(f"  Config: {_REPO_ROOT / 'vendor/starVLA/configs/libero_dhbxr.yaml'}")
    print()

    cmds = generate_training_commands(demo_counts)
    for i, (n, cmd) in enumerate(zip(demo_counts, cmds)):
        print(f"  --- Run {i+1}/{len(demo_counts)}: N={n} demos ---")
        print(cmd)
        print()


# ── 3. Spatial perturbation evaluation summary ───────────────────────────────

def print_eval_protocol() -> None:
    print("\n=== 3. Spatial perturbation evaluation protocol ===")
    print("""
  After each training run completes:

  Evaluation script (StarVLA harness):
    cd vendor/vla-evaluation-harness
    python eval_libero.py \\
      --checkpoint results/Checkpoints/exp2_dhbxr_libero_spatial_50demo/final_model \\
      --suite libero_spatial \\
      --perturbations none,translate_5cm,translate_10cm,rotate_15deg,rotate_30deg \\
      --num_eval_episodes 20 \\
      --output_dir results/eval/exp2_dhbxr_50demo

  Metrics collected:
    - Success rate per perturbation level (primary metric for Exp 2)
    - Success rate at zero perturbation (sanity check)
    - SE(3) stability score (measured by DHB-XR invariant drift)

  Expected comparison table (CoRL 2026 Table 2):
    Method           | N=5  | N=10 | N=20 | N=50
    MLP (OFT)        |      |      |      |
    FAST             |      |      |      |
    DHB-XR (ours)    |      |      |      |
    DHB-XR no-OAT    |      |      |      |
""")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Sample Efficiency Sweep — CoRL 2026 Experiment 2")
    print("StarVLA + DHB-XR on LIBERO-Spatial")
    print("=" * 70)

    stats = validate_demo_subsets()

    if stats:
        print("\n  Subset loading summary:")
        for n, s in stats.items():
            print(
                f"    N={n:2d}: {s['total_tasks']} tasks, "
                f"{s['total_transitions']:,} transitions total"
            )

    print_training_commands(DEMO_COUNTS)
    print_eval_protocol()

    print("\n=== Done ===")
    if stats:
        print("  LIBERO HDF5 subset loading: OK")
        print(f"  Tasks found: {list(stats.values())[0]['total_tasks']}")
        print(f"  Demo counts validated: {DEMO_COUNTS}")
    print("  Training commands: printed above (requires WM4A_DHB patch first)")


if __name__ == "__main__":
    main()
