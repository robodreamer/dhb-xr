from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


LIBERO_DATA_ROOT = Path("/home/andypark/Projects/data/libero")


def _matches(name: str | None, tokens: tuple[str, ...]) -> bool:
    if not name:
        return False
    lowered = name.lower()
    return all(tok in lowered for tok in tokens)


def _find_ref(env, tokens: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    model = env.env.sim.model
    data = env.env.sim.data
    for body_id in range(model.nbody):
        name = model.body_id2name(body_id)
        if _matches(name, tokens):
            return np.array(data.body_xpos[body_id]), f"body:{name}"
    for site_id in range(model.nsite):
        name = model.site_id2name(site_id)
        if _matches(name, tokens):
            return np.array(data.site_xpos[site_id]), f"site:{name}"
    for geom_id in range(model.ngeom):
        name = model.geom_id2name(geom_id)
        if _matches(name, tokens):
            return np.array(data.geom_xpos[geom_id]), f"geom:{name}"
    return None, None


def _event_indices(actions: np.ndarray, gripper_qpos: np.ndarray, threshold: float) -> dict[str, int]:
    cmd = actions[:, -1]
    close_cmd = np.flatnonzero(cmd > 0.0)
    open_cmd = np.flatnonzero(cmd < 0.0)
    closed = gripper_qpos <= threshold
    transitions = np.flatnonzero(np.diff(closed.astype(np.int8)) != 0) + 1

    events: dict[str, int] = {}
    if close_cmd.size:
        events["close_cmd"] = int(close_cmd[0])
        later_open = open_cmd[open_cmd > close_cmd[0]]
        if later_open.size:
            events["open_cmd"] = int(later_open[0])
    for idx in transitions:
        idx = int(idx)
        if closed[idx] and "close_phys" not in events:
            events["close_phys"] = idx
        if (not closed[idx]) and "open_phys" not in events:
            events["open_phys"] = idx
    return events


def _summarize(vectors: list[list[float]], target_vectors: list[list[float]]) -> dict:
    arr = np.asarray(vectors, dtype=np.float64)
    tgt = np.asarray(target_vectors, dtype=np.float64)
    xy = np.linalg.norm(arr[:, :2], axis=1)
    target_xy = np.linalg.norm(tgt[:, :2], axis=1)
    return {
        "count": int(arr.shape[0]),
        "ee_minus_object_mean_mm": (arr.mean(axis=0) * 1000.0).round(3).tolist(),
        "ee_minus_object_std_mm": (arr.std(axis=0) * 1000.0).round(3).tolist(),
        "ee_object_xy_mean_mm": float(xy.mean() * 1000.0),
        "ee_object_xy_std_mm": float(xy.std() * 1000.0),
        "ee_object_xy_min_mm": float(xy.min() * 1000.0),
        "ee_object_xy_max_mm": float(xy.max() * 1000.0),
        "object_minus_target_xy_mean_mm": float(target_xy.mean() * 1000.0),
        "object_minus_target_xy_std_mm": float(target_xy.std() * 1000.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--object-token", default="bowl")
    parser.add_argument("--target-token", default="plate")
    parser.add_argument("--gripper-qpos-threshold", type=float, default=0.015)
    parser.add_argument("--max-demos", type=int, default=0)
    parser.add_argument("--save-json", default="")
    args = parser.parse_args()

    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    hdf5_path = LIBERO_DATA_ROOT / args.task_suite_name / f"{task.name}_demo.hdf5"
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    object_tokens = tuple(tok for tok in args.object_token.lower().split(",") if tok)
    target_tokens = tuple(tok for tok in args.target_token.lower().split(",") if tok)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        controller="JOINT_POSITION",
        camera_heights=128,
        camera_widths=128,
    )
    env.seed(0)
    env.reset()

    rows: list[dict] = []
    with h5py.File(hdf5_path, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        if args.max_demos > 0:
            demo_names = demo_names[: args.max_demos]
        for demo_name in demo_names:
            demo = f["data"][demo_name]
            states = np.asarray(demo["states"], dtype=np.float64)
            ee_pos = np.asarray(demo["obs"]["ee_pos"], dtype=np.float64)
            gripper_qpos = np.asarray(demo["obs"]["gripper_states"], dtype=np.float64)[:, 0]
            actions = np.asarray(demo["actions"], dtype=np.float64)
            events = _event_indices(actions, gripper_qpos, args.gripper_qpos_threshold)
            for event_name, step in events.items():
                env.set_init_state(states[step])
                object_pos, object_ref = _find_ref(env, object_tokens)
                target_pos, target_ref = _find_ref(env, target_tokens)
                if object_pos is None or target_pos is None:
                    raise RuntimeError(
                        f"Could not resolve object={object_tokens} or target={target_tokens}"
                    )
                ee_minus_object = ee_pos[step] - object_pos
                object_minus_target = object_pos - target_pos
                rows.append(
                    {
                        "demo": demo_name,
                        "event": event_name,
                        "step": int(step),
                        "episode_len": int(len(states)),
                        "gripper_qpos": float(gripper_qpos[step]),
                        "command": float(actions[step, -1]),
                        "object_ref": object_ref,
                        "target_ref": target_ref,
                        "ee_minus_object_m": ee_minus_object.tolist(),
                        "object_minus_target_m": object_minus_target.tolist(),
                        "ee_object_xy_m": float(math.hypot(ee_minus_object[0], ee_minus_object[1])),
                        "object_target_xy_m": float(
                            math.hypot(object_minus_target[0], object_minus_target[1])
                        ),
                    }
                )

    env.close()

    by_event: dict[str, dict] = {}
    for event_name in sorted({row["event"] for row in rows}):
        event_rows = [row for row in rows if row["event"] == event_name]
        by_event[event_name] = _summarize(
            [row["ee_minus_object_m"] for row in event_rows],
            [row["object_minus_target_m"] for row in event_rows],
        )

    payload = {
        "task_suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "task_name": task.name,
        "hdf5_path": str(hdf5_path),
        "object_token": args.object_token,
        "target_token": args.target_token,
        "gripper_qpos_threshold": args.gripper_qpos_threshold,
        "events": rows,
        "summary": by_event,
    }

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps({"summary": by_event}, indent=2))


if __name__ == "__main__":
    main()
