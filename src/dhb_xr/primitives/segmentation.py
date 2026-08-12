"""Boundary proposals and shared-waypoint utilities for imported demonstrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from dhb_xr.primitives.contracts import MotionPath, PrimitiveContractError


@dataclass(frozen=True)
class BoundaryProposal:
    """One proposed primitive boundary and the observable reasons for it."""

    index: int
    reasons: tuple[str, ...]


def exact_gripper_run_control_indices(commands: np.ndarray) -> tuple[int, ...]:
    """Reproduce the sample-exact scalar gripper change-point baseline."""

    values = _commands(commands)
    indices = [0]
    indices.extend(int(index + 1) for index in np.flatnonzero(values[1:] != values[:-1]))
    if indices[-1] != len(values) - 1:
        indices.append(len(values) - 1)
    return tuple(indices)


def hysteretic_gripper_control_indices(
    commands: np.ndarray,
    *,
    closed_threshold: float = 0.3,
    open_threshold: float = 0.7,
    min_dwell_samples: int = 2,
    invert: bool = False,
) -> tuple[int, ...]:
    """Detect stable open/closed transitions without exploding on command ramps."""

    values = _commands(commands)
    if invert:
        values = 1.0 - values
    closed = float(closed_threshold)
    opened = float(open_threshold)
    dwell = int(min_dwell_samples)
    if not 0.0 <= closed < opened <= 1.0:
        raise PrimitiveContractError(
            "gripper thresholds must satisfy 0 <= closed < open <= 1"
        )
    if dwell < 1:
        raise PrimitiveContractError("min_dwell_samples must be at least one")
    state_open = bool(values[0] >= 0.5)
    candidate_start: Optional[int] = None
    candidate_count = 0
    transitions = [0]
    for index in range(1, len(values)):
        opposite = values[index] <= closed if state_open else values[index] >= opened
        if not opposite:
            candidate_start = None
            candidate_count = 0
            continue
        if candidate_start is None:
            candidate_start = index
            candidate_count = 1
        else:
            candidate_count += 1
        if candidate_count >= dwell:
            transitions.append(candidate_start)
            state_open = not state_open
            candidate_start = None
            candidate_count = 0
    if transitions[-1] != len(values) - 1:
        transitions.append(len(values) - 1)
    return tuple(transitions)


def propose_primitive_boundaries(
    gripper_commands: np.ndarray,
    *,
    contacts: Optional[np.ndarray] = None,
    object_positions: Optional[np.ndarray] = None,
    object_motion_threshold_m: float = 0.002,
    closed_threshold: float = 0.3,
    open_threshold: float = 0.7,
    min_dwell_samples: int = 2,
) -> tuple[BoundaryProposal, ...]:
    """Fuse gripper, contact, and object-motion signals into boundary proposals."""

    values = _commands(gripper_commands)
    reasons: dict[int, set[str]] = {0: {"trajectory_start"}, len(values) - 1: {"trajectory_end"}}
    gripper_indices = hysteretic_gripper_control_indices(
        values,
        closed_threshold=closed_threshold,
        open_threshold=open_threshold,
        min_dwell_samples=min_dwell_samples,
    )
    for index in gripper_indices[1:-1]:
        reasons.setdefault(index, set()).add("gripper_state_transition")
    if contacts is not None:
        contact_values = np.asarray(contacts, dtype=bool)
        if contact_values.shape != values.shape:
            raise PrimitiveContractError("contacts must have the same length as gripper_commands")
        for index in np.flatnonzero(contact_values[1:] != contact_values[:-1]) + 1:
            reasons.setdefault(int(index), set()).add("contact_transition")
    if object_positions is not None:
        positions = np.asarray(object_positions, dtype=np.float64)
        if positions.shape != (len(values), 3) or not np.all(np.isfinite(positions)):
            raise PrimitiveContractError("object_positions must have shape (N, 3) and be finite")
        threshold = float(object_motion_threshold_m)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise PrimitiveContractError("object_motion_threshold_m must be positive")
        moving = np.linalg.norm(np.diff(positions, axis=0), axis=1) >= threshold
        onset = np.flatnonzero(moving & np.concatenate([[True], ~moving[:-1]])) + 1
        for index in onset:
            reasons.setdefault(int(index), set()).add("object_motion_onset")
    return tuple(
        BoundaryProposal(index, tuple(sorted(boundary_reasons)))
        for index, boundary_reasons in sorted(reasons.items())
    )


def split_motion_at_indices(
    motion: MotionPath, control_indices: tuple[int, ...]
) -> tuple[MotionPath, ...]:
    """Split a path into pieces with shared endpoints at every boundary."""

    indices = tuple(int(index) for index in control_indices)
    if len(indices) < 2 or indices[0] != 0 or indices[-1] != motion.samples - 1:
        raise PrimitiveContractError("control indices must include path start and final sample")
    if any(goal <= start for start, goal in zip(indices[:-1], indices[1:])):
        raise PrimitiveContractError("control indices must be strictly increasing")
    pieces = []
    for start, goal in zip(indices[:-1], indices[1:]):
        if goal >= motion.samples:
            raise PrimitiveContractError("control index exceeds motion samples")
        timestamps = motion.timestamps_s[start : goal + 1] - motion.timestamps_s[start]
        pieces.append(
            MotionPath(
                motion.positions[start : goal + 1],
                motion.quaternions_wxyz[start : goal + 1],
                timestamps,
                frame_id=motion.frame_id,
            )
        )
    return tuple(pieces)


def stitch_motion_paths(
    pieces: tuple[MotionPath, ...], *, position_tolerance_m: float = 1e-8
) -> MotionPath:
    """Stitch path pieces only when adjacent endpoint poses agree."""

    if not pieces:
        raise PrimitiveContractError("at least one motion piece is required")
    frame_id = pieces[0].frame_id
    positions = []
    quaternions = []
    timestamps = []
    offset = 0.0
    for index, piece in enumerate(pieces):
        if piece.frame_id != frame_id:
            raise PrimitiveContractError("all motion pieces must use the same frame_id")
        if index:
            previous = pieces[index - 1]
            if np.linalg.norm(previous.positions[-1] - piece.positions[0]) > position_tolerance_m:
                raise PrimitiveContractError("adjacent motion positions do not share an endpoint")
            alignment = abs(float(np.dot(previous.quaternions_wxyz[-1], piece.quaternions_wxyz[0])))
            if 1.0 - alignment > 1e-8:
                raise PrimitiveContractError(
                    "adjacent motion orientations do not share an endpoint"
                )
        start = 0 if index == 0 else 1
        positions.append(piece.positions[start:])
        quaternions.append(piece.quaternions_wxyz[start:])
        timestamps.append(piece.timestamps_s[start:] + offset)
        offset += piece.duration_s
    return MotionPath(
        np.concatenate(positions),
        np.concatenate(quaternions),
        np.concatenate(timestamps),
        frame_id=frame_id,
    )


def _commands(commands: np.ndarray) -> np.ndarray:
    values = np.asarray(commands, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise PrimitiveContractError("gripper_commands must be a finite vector with N >= 2")
    result: np.ndarray = values
    return result
