from __future__ import annotations

"""Shared receding-horizon chunk utilities for DHB trajectory rollout.

This keeps the loop mechanics closer to the vendor StarVLA chunk pattern
(`predict -> post-process -> execute prefix -> replan`) while preserving the
custom DHB execution path:

    predict_trajectory() -> waypoint chunk -> embodik / sim

The main goal is to centralize chunk stabilization, overlap blending, and
learned gripper decoding so Stage 1 and live rollout do not reimplement the
same logic independently.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dhb_xr.integration.vla.libero import GripperDecodeState, decode_model_gripper_cmds


def clip_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if max_norm <= 0.0 or norm <= max_norm or norm <= 1e-9:
        return vec
    return vec * (max_norm / norm)


def stabilize_chunk_positions(
    chunk_pos: np.ndarray,
    *,
    anchor_pos: np.ndarray,
    max_first_offset_m: float,
    max_step_m: float,
    smooth_window: int,
) -> tuple[np.ndarray, dict[str, float]]:
    stabilized = np.asarray(chunk_pos, dtype=np.float32).copy()
    debug = {
        "raw_first_offset_mm": 0.0,
        "stabilized_first_offset_mm": 0.0,
        "raw_mean_step_mm": 0.0,
        "stabilized_mean_step_mm": 0.0,
    }
    if len(stabilized) == 0:
        return stabilized, debug

    anchor = np.asarray(anchor_pos, dtype=np.float32).reshape(3)
    debug["raw_first_offset_mm"] = float(np.linalg.norm(stabilized[0] - anchor) * 1000.0)
    if len(stabilized) > 1:
        debug["raw_mean_step_mm"] = float(
            np.linalg.norm(np.diff(stabilized, axis=0), axis=1).mean() * 1000.0
        )

    if max_first_offset_m > 0.0:
        stabilized[0] = anchor + clip_vector_norm(stabilized[0] - anchor, max_first_offset_m)

    if max_step_m > 0.0:
        for i in range(1, len(stabilized)):
            step = stabilized[i] - stabilized[i - 1]
            stabilized[i] = stabilized[i - 1] + clip_vector_norm(step, max_step_m)

    if smooth_window > 1 and len(stabilized) > 2:
        smoothed = stabilized.copy()
        radius = smooth_window // 2
        for i in range(len(stabilized)):
            lo = max(0, i - radius)
            hi = min(len(stabilized), i + radius + 1)
            smoothed[i] = stabilized[lo:hi].mean(axis=0)
        stabilized = smoothed
        stabilized[0] = anchor + clip_vector_norm(stabilized[0] - anchor, max_first_offset_m)
        if max_step_m > 0.0:
            for i in range(1, len(stabilized)):
                step = stabilized[i] - stabilized[i - 1]
                stabilized[i] = stabilized[i - 1] + clip_vector_norm(step, max_step_m)

    debug["stabilized_first_offset_mm"] = float(np.linalg.norm(stabilized[0] - anchor) * 1000.0)
    if len(stabilized) > 1:
        debug["stabilized_mean_step_mm"] = float(
            np.linalg.norm(np.diff(stabilized, axis=0), axis=1).mean() * 1000.0
        )
    return stabilized.astype(np.float32), debug


def blend_overlap_positions(
    current_chunk: np.ndarray,
    *,
    current_start: int,
    prev_chunk: np.ndarray | None,
    prev_start: int | None,
    blend_alpha: float,
    blend_mode: str = "constant",
    commit_prefix_steps: int = 0,
    adaptive_commit_threshold_m: float = 0.0,
    adaptive_commit_max_steps: int = 0,
) -> tuple[np.ndarray, int, int, dict[str, float | int | None]]:
    if prev_chunk is None or prev_start is None:
        return current_chunk, 0, 0, {
            "adaptive_commit_steps": 0,
            "adaptive_commit_triggered": 0,
            "overlap_disagreement_mean_mm": None,
            "overlap_disagreement_max_mm": None,
        }

    prev_end = prev_start + len(prev_chunk)
    overlap = min(max(prev_end - current_start, 0), len(current_chunk))
    if overlap <= 0:
        return current_chunk, 0, 0, {
            "adaptive_commit_steps": 0,
            "adaptive_commit_triggered": 0,
            "overlap_disagreement_mean_mm": None,
            "overlap_disagreement_max_mm": None,
        }

    alpha = float(np.clip(blend_alpha, 0.0, 1.0))
    prev_tail = np.asarray(prev_chunk[-overlap:], dtype=np.float32)
    prefix = np.asarray(current_chunk[:overlap], dtype=np.float32)
    disagreement = np.linalg.norm(prefix - prev_tail, axis=1)
    commit = int(max(0, min(commit_prefix_steps, overlap)))
    adaptive_commit = 0
    if adaptive_commit_threshold_m > 0.0:
        adaptive_limit = overlap if adaptive_commit_max_steps <= 0 else min(overlap, adaptive_commit_max_steps)
        for idx in range(commit, adaptive_limit):
            if float(disagreement[idx]) > adaptive_commit_threshold_m:
                adaptive_commit = idx + 1
            else:
                break
    commit = max(commit, adaptive_commit)
    debug = {
        "adaptive_commit_steps": int(adaptive_commit),
        "adaptive_commit_triggered": int(adaptive_commit > 0),
        "overlap_disagreement_mean_mm": float(disagreement.mean() * 1000.0),
        "overlap_disagreement_max_mm": float(disagreement.max() * 1000.0),
    }
    if alpha <= 0.0 and commit <= 0:
        return current_chunk, 0, 0, debug

    blended = np.asarray(current_chunk, dtype=np.float32).copy()
    if commit > 0:
        blended[:commit] = prev_tail[:commit]

    if overlap > commit and alpha > 0.0:
        if blend_mode == "ramp":
            alpha_lo = min(alpha, 0.2)
            alpha_sched = np.linspace(alpha_lo, alpha, overlap - commit, dtype=np.float32)
            for idx, alpha_i in enumerate(alpha_sched, start=commit):
                blended[idx] = (1.0 - alpha_i) * prev_tail[idx] + alpha_i * blended[idx]
        else:
            blended[commit:overlap] = (
                (1.0 - alpha) * prev_tail[commit:overlap]
                + alpha * blended[commit:overlap]
            )
    return blended, int(overlap), int(commit), debug


def compute_motion_metrics(positions: np.ndarray) -> dict[str, float | int | None]:
    pos = np.asarray(positions, dtype=np.float32)
    if len(pos) == 0:
        return {
            "num_points": 0,
            "path_length_m": 0.0,
            "net_displacement_m": 0.0,
            "path_efficiency": None,
            "mean_step_mm": 0.0,
            "std_step_mm": 0.0,
            "max_step_mm": 0.0,
            "hold_fraction": 0.0,
            "mean_accel_mm": 0.0,
            "max_accel_mm": 0.0,
            "mean_jerk_mm": 0.0,
            "max_jerk_mm": 0.0,
            "direction_flip_fraction": 0.0,
        }

    if len(pos) == 1:
        return {
            "num_points": 1,
            "path_length_m": 0.0,
            "net_displacement_m": 0.0,
            "path_efficiency": None,
            "mean_step_mm": 0.0,
            "std_step_mm": 0.0,
            "max_step_mm": 0.0,
            "hold_fraction": 1.0,
            "mean_accel_mm": 0.0,
            "max_accel_mm": 0.0,
            "mean_jerk_mm": 0.0,
            "max_jerk_mm": 0.0,
            "direction_flip_fraction": 0.0,
        }

    steps = np.diff(pos, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    path_length_m = float(step_norms.sum())
    net_displacement_m = float(np.linalg.norm(pos[-1] - pos[0]))
    hold_fraction = float(np.mean(step_norms < 1e-4))

    accel = np.diff(steps, axis=0)
    accel_norms = np.linalg.norm(accel, axis=1) if len(accel) else np.zeros((0,), dtype=np.float32)
    jerk = np.diff(accel, axis=0)
    jerk_norms = np.linalg.norm(jerk, axis=1) if len(jerk) else np.zeros((0,), dtype=np.float32)

    direction_flip_fraction = 0.0
    if len(steps) > 1:
        prev = steps[:-1]
        nxt = steps[1:]
        denom = np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1)
        valid = denom > 1e-8
        if np.any(valid):
            cos = np.sum(prev[valid] * nxt[valid], axis=1) / denom[valid]
            direction_flip_fraction = float(np.mean(cos < 0.0))

    return {
        "num_points": int(len(pos)),
        "path_length_m": path_length_m,
        "net_displacement_m": net_displacement_m,
        "path_efficiency": None if path_length_m <= 1e-8 else float(net_displacement_m / path_length_m),
        "mean_step_mm": float(step_norms.mean() * 1000.0),
        "std_step_mm": float(step_norms.std() * 1000.0),
        "max_step_mm": float(step_norms.max() * 1000.0),
        "hold_fraction": hold_fraction,
        "mean_accel_mm": 0.0 if len(accel_norms) == 0 else float(accel_norms.mean() * 1000.0),
        "max_accel_mm": 0.0 if len(accel_norms) == 0 else float(accel_norms.max() * 1000.0),
        "mean_jerk_mm": 0.0 if len(jerk_norms) == 0 else float(jerk_norms.mean() * 1000.0),
        "max_jerk_mm": 0.0 if len(jerk_norms) == 0 else float(jerk_norms.max() * 1000.0),
        "direction_flip_fraction": direction_flip_fraction,
    }


def smooth_chunk_step_prefix(
    chunk_pos: np.ndarray,
    *,
    prev_tail_step: np.ndarray | None,
    max_step_m: float,
    delta_continuation_alpha: float,
    delta_continuation_steps: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    smoothed = np.asarray(chunk_pos, dtype=np.float32).copy()
    debug = {
        "delta_continuation_applied": 0,
        "delta_continuation_steps_used": 0,
    }
    if (
        prev_tail_step is None
        or float(delta_continuation_alpha) <= 0.0
        or int(delta_continuation_steps) <= 0
        or len(smoothed) < 2
    ):
        return smoothed, debug

    deltas = np.diff(smoothed, axis=0)
    n = min(len(deltas), max(0, int(delta_continuation_steps)))
    if n <= 0:
        return smoothed, debug

    alpha = float(np.clip(delta_continuation_alpha, 0.0, 1.0))
    prev_step = np.asarray(prev_tail_step, dtype=np.float32).reshape(3)
    for idx in range(n):
        if n == 1:
            weight = alpha
        else:
            weight = alpha * (1.0 - float(idx) / float(n - 1))
        deltas[idx] = (1.0 - weight) * deltas[idx] + weight * prev_step
        if max_step_m > 0.0:
            deltas[idx] = clip_vector_norm(deltas[idx], max_step_m)

    smoothed[0] = chunk_pos[0]
    for idx in range(1, len(smoothed)):
        smoothed[idx] = smoothed[idx - 1] + deltas[idx - 1]

    debug["delta_continuation_applied"] = 1
    debug["delta_continuation_steps_used"] = int(n)
    return smoothed.astype(np.float32), debug


@dataclass
class ChunkRolloutConfig:
    max_first_offset_m: float = 0.0
    max_step_m: float = 0.0
    smooth_window: int = 0
    blend_overlap_alpha: float = 0.0
    blend_mode: str = "constant"
    commit_prefix_steps: int = 0
    adaptive_commit_threshold_m: float = 0.0
    adaptive_commit_max_steps: int = 0
    delta_continuation_alpha: float = 0.0
    delta_continuation_steps: int = 0
    gripper_source: str = "demo"
    model_gripper_decode_source: str = "auto"
    gripper_threshold: float = 0.5
    gripper_close_threshold: float | None = 0.6
    gripper_open_threshold: float | None = 0.4
    gripper_transition_debounce_steps: int = 1


class TrajectoryChunkRunner:
    """Stateful helper for vendor-style chunk-boundary trajectory scheduling.

    The runner does not own environment stepping or IK. It only caches the
    previous predicted chunk so consecutive replans can be post-processed
    consistently across offline and live evaluation.
    """

    def __init__(self, config: ChunkRolloutConfig) -> None:
        self.config = config
        self._prev_chunk_pos: np.ndarray | None = None
        self._prev_chunk_start: int | None = None
        self._prev_tail_step: np.ndarray | None = None
        self._gripper_decode_state = GripperDecodeState()

    def reset(self) -> None:
        self._prev_chunk_pos = None
        self._prev_chunk_start = None
        self._prev_tail_step = None
        self._gripper_decode_state = GripperDecodeState()

    def prepare_chunk(
        self,
        result: dict[str, Any],
        *,
        anchor_pos: np.ndarray,
        current_start: int,
        n_use: int,
        initial_gripper_cmd: float,
        demo_gripper_chunk: np.ndarray | None = None,
        config_override: ChunkRolloutConfig | None = None,
    ) -> dict[str, Any]:
        cfg = config_override or self.config
        chunk_pos = np.asarray(result["positions"][0], dtype=np.float32)
        chunk_quat = np.asarray(result["quaternions"][0], dtype=np.float32)  # wxyz

        chunk_pos_stable, stability = stabilize_chunk_positions(
            chunk_pos,
            anchor_pos=anchor_pos,
            max_first_offset_m=max(0.0, cfg.max_first_offset_m),
            max_step_m=max(0.0, cfg.max_step_m),
            smooth_window=max(0, cfg.smooth_window),
        )
        chunk_pos_stable, overlap_steps, commit_steps, blend_debug = blend_overlap_positions(
            chunk_pos_stable,
            current_start=current_start,
            prev_chunk=self._prev_chunk_pos,
            prev_start=self._prev_chunk_start,
            blend_alpha=cfg.blend_overlap_alpha,
            blend_mode=cfg.blend_mode,
            commit_prefix_steps=cfg.commit_prefix_steps,
            adaptive_commit_threshold_m=max(0.0, cfg.adaptive_commit_threshold_m),
            adaptive_commit_max_steps=max(0, cfg.adaptive_commit_max_steps),
        )
        chunk_pos_stable, delta_debug = smooth_chunk_step_prefix(
            chunk_pos_stable,
            prev_tail_step=self._prev_tail_step,
            max_step_m=max(0.0, cfg.max_step_m),
            delta_continuation_alpha=float(cfg.delta_continuation_alpha),
            delta_continuation_steps=max(0, cfg.delta_continuation_steps),
        )

        if cfg.gripper_source == "demo":
            if demo_gripper_chunk is None:
                raise ValueError("demo_gripper_chunk is required when gripper_source='demo'")
            gripper_cmds = np.asarray(demo_gripper_chunk[:n_use], dtype=np.float32)
            gripper_debug = {"source": "demo"}
        else:
            gripper_cmds, gripper_debug = decode_model_gripper_cmds(
                result,
                n_use=n_use,
                initial_gripper_cmd=float(initial_gripper_cmd),
                threshold=float(cfg.gripper_threshold),
                close_threshold=cfg.gripper_close_threshold,
                open_threshold=cfg.gripper_open_threshold,
                transition_debounce_steps=max(1, int(cfg.gripper_transition_debounce_steps)),
                decode_source=cfg.model_gripper_decode_source,
                decode_state=self._gripper_decode_state,
            )

        self._prev_chunk_pos = chunk_pos_stable.copy()
        self._prev_chunk_start = int(current_start)
        if len(chunk_pos_stable) >= 2:
            self._prev_tail_step = (chunk_pos_stable[-1] - chunk_pos_stable[-2]).astype(np.float32)
        else:
            self._prev_tail_step = None

        return {
            "positions": chunk_pos_stable[:n_use],
            "positions_full": chunk_pos_stable,
            "quaternions": chunk_quat[:n_use],
            "gripper_cmds": gripper_cmds,
            "debug": {
                "gripper_source_used": gripper_debug.get("source"),
                "close_probs": gripper_debug.get("close_probs"),
                "open_probs": gripper_debug.get("open_probs"),
                "gripper_threshold": gripper_debug.get("threshold"),
                "gripper_close_threshold": gripper_debug.get("close_threshold"),
                "gripper_open_threshold": gripper_debug.get("open_threshold"),
                "gripper_transition_debounce_steps": gripper_debug.get("transition_debounce_steps"),
                "gripper_debounce_pending_target": gripper_debug.get("debounce_pending_target"),
                "gripper_debounce_pending_count": gripper_debug.get("debounce_pending_count"),
                "overlap_steps": int(overlap_steps),
                "commit_prefix_steps": int(commit_steps),
                "blend_mode": cfg.blend_mode,
                "adaptive_commit_threshold_mm": float(cfg.adaptive_commit_threshold_m * 1000.0),
                "adaptive_commit_max_steps": int(cfg.adaptive_commit_max_steps),
                "delta_continuation_alpha": float(cfg.delta_continuation_alpha),
                "delta_continuation_steps": int(cfg.delta_continuation_steps),
                **blend_debug,
                **delta_debug,
                **stability,
            },
            "first_pred_pos": chunk_pos[0].tolist() if len(chunk_pos) else None,
            "first_stabilized_pos": chunk_pos_stable[0].tolist() if len(chunk_pos_stable) else None,
        }


@dataclass
class PhaseAwareChunkRunner:
    """Generic phase-aware wrapper around one shared trajectory chunk runner.

    The caller decides *which* phase is active. This wrapper centralizes
    *how* each phase maps to a chunk-rollout policy while keeping the
    continuation state shared across phases.
    """

    default_config: ChunkRolloutConfig
    phase_configs: dict[str, ChunkRolloutConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._runner = TrajectoryChunkRunner(self.default_config)

    def reset(self) -> None:
        self._runner.reset()

    def get_phase_config(self, phase_name: str | None) -> ChunkRolloutConfig | None:
        if phase_name is None or phase_name == "default":
            return None
        return self.phase_configs.get(phase_name)

    def prepare_chunk(
        self,
        result: dict[str, Any],
        *,
        anchor_pos: np.ndarray,
        current_start: int,
        n_use: int,
        initial_gripper_cmd: float,
        demo_gripper_chunk: np.ndarray | None = None,
        phase_name: str | None = None,
    ) -> dict[str, Any]:
        active_phase = "default" if phase_name is None else str(phase_name)
        prepared = self._runner.prepare_chunk(
            result,
            anchor_pos=anchor_pos,
            current_start=current_start,
            n_use=n_use,
            initial_gripper_cmd=initial_gripper_cmd,
            demo_gripper_chunk=demo_gripper_chunk,
            config_override=self.get_phase_config(active_phase),
        )
        prepared["debug"]["phase_mode"] = active_phase
        return prepared
