from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from vla_arena.vla_arena import benchmark, get_vla_arena_path
    from vla_arena.vla_arena.envs import OffScreenRenderEnv

    bench_cls = benchmark.get_benchmark_dict()[args.suite]
    task_suite = bench_cls()
    task = task_suite.get_task_by_level_id(args.level, args.task_id)
    initial_states = task_suite.get_task_init_states(args.level, args.task_id)

    task_bddl_file = os.path.join(
        get_vla_arena_path("bddl_files"),
        task.problem_folder,
        f"level_{task.level}",
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=256,
        camera_widths=256,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=20)
    rng = np.random.default_rng(7)

    try:
        env.reset()
        obs = env.set_init_state(initial_states[0])
        for _ in range(10):
            obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

        for _ in range(args.steps):
            frame = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            writer.append_data(frame)
            action = rng.uniform(low=-1.0, high=1.0, size=7)
            action[-1] = np.clip(action[-1], -1.0, 1.0)
            obs, _, done, _ = env.step(action.tolist())
            if done:
                break
    finally:
        writer.close()
        env.close()

    print(f"saved_video={out_path}")


if __name__ == "__main__":
    main()
