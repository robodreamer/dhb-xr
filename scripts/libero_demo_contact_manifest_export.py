#!/usr/bin/env python3
"""Export LIBERO demo contact events as policy-trace contact manifests.

The policy-trace contact loader consumes NPZ chunks containing the image, state
history, and per-timestep close labels. This script creates the same format
from real LIBERO demonstration observations around gripper close events. It is
training data generation only; deployment still receives only observations and
robot state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from libero.libero import benchmark

from dhb_xr.integration.vla.libero import build_libero_state_vector, build_state_history_window


LIBERO_DATA_ROOT = Path("/home/andypark/Projects/data/libero")
LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD = 0.015


def _preprocess_image(image: np.ndarray, orientation: str) -> np.ndarray:
    if orientation == "raw":
        return np.ascontiguousarray(image)
    if orientation == "flipudlr":
        return np.ascontiguousarray(image[::-1, ::-1])
    raise ValueError(f"Unsupported image orientation: {orientation}")


def _event_indices(actions: np.ndarray, gripper_qpos: np.ndarray, threshold: float) -> dict[str, int]:
    cmd = np.asarray(actions[:, -1], dtype=np.float32)
    close_cmd = np.flatnonzero(cmd > 0.0)
    open_cmd = np.flatnonzero(cmd < 0.0)
    closed = np.asarray(gripper_qpos <= threshold, dtype=np.int8)
    transitions = np.flatnonzero(np.diff(closed) != 0) + 1

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


def _state_vectors(robot_states: np.ndarray) -> np.ndarray:
    states = []
    for row in np.asarray(robot_states, dtype=np.float32):
        states.append(
            build_libero_state_vector(
                row[2:5],
                row[5:9],
                float(row[0]),
            )
        )
    return np.asarray(states, dtype=np.float32)


def _chunk_labels(
    *,
    base_index: int,
    horizon: int,
    close_step: int,
    episode_len: int,
    positive_before: int,
    positive_after: int,
) -> tuple[list[float], list[float], list[str]]:
    positive_start = max(0, close_step - positive_before)
    positive_end = min(episode_len - 1, close_step + positive_after)
    target_close: list[float] = []
    train_mask: list[float] = []
    labels: list[str] = []
    for step in range(base_index, base_index + horizon):
        valid = 0 <= step < episode_len
        is_positive = valid and positive_start <= step <= positive_end
        target_close.append(1.0 if is_positive else 0.0)
        train_mask.append(1.0 if valid else 0.0)
        labels.append("positive" if is_positive else ("background" if valid else "padding"))
    return target_close, train_mask, labels


def _write_chunk_npz(
    *,
    path: Path,
    agentview: np.ndarray,
    wrist: np.ndarray,
    states: np.ndarray,
    base_index: int,
    history_window: int,
    orientation: str,
) -> None:
    state_window = build_state_history_window(
        states[: base_index + 1],
        history_window=history_window,
        state_dim=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        agentview_image=_preprocess_image(agentview[base_index], orientation),
        wrist_image=_preprocess_image(wrist[base_index], orientation),
        state_window=state_window.astype(np.float32, copy=False),
        ee_pos_live=states[base_index, :3].astype(np.float32, copy=False),
        close_probs=np.zeros((1,), dtype=np.float32),
        open_probs=np.zeros((1,), dtype=np.float32),
    )


def export_demo_contact_manifest(
    *,
    task_suite_name: str,
    task_id: int,
    output_dir: Path,
    manifest_jsonl: Path,
    summary_json: Path,
    max_demos: int,
    action_horizon: int,
    state_history_window: int,
    positive_before: int,
    positive_after: int,
    chunk_start_min_offset: int,
    chunk_start_max_offset: int,
    negative_chunks_per_demo: int,
    orientation: str,
    gripper_qpos_threshold: float,
) -> dict:
    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    hdf5_path = LIBERO_DATA_ROOT / task_suite_name / f"{task.name}_demo.hdf5"
    if not hdf5_path.exists():
        raise FileNotFoundError(f"LIBERO demo HDF5 not found: {hdf5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    stats = {
        "num_demos": 0,
        "num_chunks": 0,
        "positive_steps": 0,
        "negative_steps": 0,
        "skipped_demos_without_close": 0,
    }

    summary = {
        "task_suite_name": task_suite_name,
        "task_id": int(task_id),
        "task_name": task.name,
        "task_language": task.language,
        "hdf5_path": str(hdf5_path),
        "output_dir": str(output_dir),
        "action_horizon": int(action_horizon),
        "state_history_window": int(state_history_window),
        "positive_before": int(positive_before),
        "positive_after": int(positive_after),
        "chunk_start_min_offset": int(chunk_start_min_offset),
        "chunk_start_max_offset": int(chunk_start_max_offset),
        "negative_chunks_per_demo": int(negative_chunks_per_demo),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with h5py.File(hdf5_path, "r") as f:
        demo_names = sorted(f["data"].keys(), key=lambda name: int(name.split("_")[-1]))
        if max_demos > 0:
            demo_names = demo_names[:max_demos]

        for demo_name in demo_names:
            demo = f["data"][demo_name]
            actions = np.asarray(demo["actions"], dtype=np.float32)
            robot_states = np.asarray(demo["robot_states"], dtype=np.float32)
            gripper_qpos = robot_states[:, 0]
            events = _event_indices(actions, gripper_qpos, gripper_qpos_threshold)
            close_step = events.get("close_phys", events.get("close_cmd"))
            if close_step is None:
                stats["skipped_demos_without_close"] += 1
                continue

            obs = demo["obs"]
            agentview = np.asarray(obs["agentview_rgb"], dtype=np.uint8)
            wrist = np.asarray(obs["eye_in_hand_rgb"], dtype=np.uint8)
            states = _state_vectors(robot_states)
            episode_len = int(len(states))
            stats["num_demos"] += 1

            chunk_bases = set()
            for offset in range(chunk_start_min_offset, chunk_start_max_offset + 1):
                base_index = int(np.clip(close_step - offset, 0, max(0, episode_len - 1)))
                chunk_bases.add(base_index)

            if negative_chunks_per_demo > 0:
                negative_hi = max(0, close_step - positive_before - action_horizon)
                if negative_hi > 0:
                    for i in range(negative_chunks_per_demo):
                        frac = (i + 1) / (negative_chunks_per_demo + 1)
                        chunk_bases.add(int(round(frac * negative_hi)))

            for base_index in sorted(chunk_bases):
                target_close, train_mask, labels = _chunk_labels(
                    base_index=base_index,
                    horizon=action_horizon,
                    close_step=int(close_step),
                    episode_len=episode_len,
                    positive_before=positive_before,
                    positive_after=positive_after,
                )
                chunk_index = len(rows)
                npz_path = output_dir / f"{demo_name}_chunk_{chunk_index:05d}_t{base_index:04d}.npz"
                _write_chunk_npz(
                    path=npz_path,
                    agentview=agentview,
                    wrist=wrist,
                    states=states,
                    base_index=base_index,
                    history_window=state_history_window,
                    orientation=orientation,
                )
                pos_count = int(sum(1 for value in target_close if value > 0.5))
                neg_count = int(sum(1 for value, mask in zip(target_close, train_mask) if mask > 0.5 and value <= 0.5))
                stats["positive_steps"] += pos_count
                stats["negative_steps"] += neg_count
                rows.append(
                    {
                        "rollout_json": str(summary_json),
                        "source": "libero_demo_contact",
                        "demo": demo_name,
                        "task_id": int(task_id),
                        "chunk_index": int(chunk_index),
                        "t_start": int(base_index),
                        "n_use": int(min(action_horizon, max(0, episode_len - base_index))),
                        "npz_path": str(npz_path),
                        "target_close": target_close,
                        "train_mask": train_mask,
                        "hard_negative_mask": [0.0 for _ in target_close],
                        "labels": labels,
                        "close_step": int(close_step),
                    }
                )

    stats["num_chunks"] = len(rows)
    summary["stats"] = stats
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with manifest_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite-name", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--max-demos", type=int, default=0)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--state-history-window", type=int, default=4)
    parser.add_argument("--positive-before", type=int, default=1)
    parser.add_argument("--positive-after", type=int, default=3)
    parser.add_argument("--chunk-start-min-offset", type=int, default=0)
    parser.add_argument("--chunk-start-max-offset", type=int, default=7)
    parser.add_argument("--negative-chunks-per-demo", type=int, default=2)
    parser.add_argument("--image-orientation", choices=("raw", "flipudlr"), default="raw")
    parser.add_argument("--gripper-qpos-threshold", type=float, default=LIBERO_GRIPPER_OPEN_QPOS_THRESHOLD)
    args = parser.parse_args()

    summary = export_demo_contact_manifest(
        task_suite_name=args.task_suite_name,
        task_id=args.task_id,
        output_dir=Path(args.output_dir),
        manifest_jsonl=Path(args.manifest_jsonl),
        summary_json=Path(args.summary_json),
        max_demos=int(args.max_demos),
        action_horizon=int(args.action_horizon),
        state_history_window=int(args.state_history_window),
        positive_before=int(args.positive_before),
        positive_after=int(args.positive_after),
        chunk_start_min_offset=int(args.chunk_start_min_offset),
        chunk_start_max_offset=int(args.chunk_start_max_offset),
        negative_chunks_per_demo=int(args.negative_chunks_per_demo),
        orientation=args.image_orientation,
        gripper_qpos_threshold=float(args.gripper_qpos_threshold),
    )
    print(json.dumps(summary["stats"], indent=2))


if __name__ == "__main__":
    main()
