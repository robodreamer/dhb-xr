"""Analytic SE(3) path generators for the cheapest-sufficient primitive tier."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from dhb_xr.core.geometry import axis_angle_to_quat, quat_multiply, quat_slerp
from dhb_xr.primitives.contracts import MotionPath, PrimitiveContractError


def minimum_jerk_blend(phase: np.ndarray | float) -> np.ndarray:
    """Return ``10t^3 - 15t^4 + 6t^5`` for phase values in ``[0, 1]``."""

    values = np.asarray(phase, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise PrimitiveContractError("minimum-jerk phase must be finite and lie in [0, 1]")
    result: np.ndarray = 10.0 * values**3 - 15.0 * values**4 + 6.0 * values**5
    return result


def _unit_axis(axis: np.ndarray, field_name: str = "axis") -> np.ndarray:
    value = np.asarray(axis, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise PrimitiveContractError(f"{field_name} must be a finite 3-vector")
    norm = float(np.linalg.norm(value))
    if norm <= 1e-10:
        raise PrimitiveContractError(f"{field_name} must be non-zero")
    result: np.ndarray = value / norm
    return result


def _position(value: np.ndarray, field_name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise PrimitiveContractError(f"{field_name} must be a finite 3-vector")
    validated: np.ndarray = result
    return validated


def _quaternion(value: np.ndarray, field_name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4,) or not np.all(np.isfinite(result)):
        raise PrimitiveContractError(f"{field_name} must be a finite wxyz quaternion")
    norm = float(np.linalg.norm(result))
    if norm <= 1e-10:
        raise PrimitiveContractError(f"{field_name} must be non-zero")
    normalized: np.ndarray = result / norm
    return normalized


def _sampling(samples: int, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    count = int(samples)
    duration = float(duration_s)
    if count < 2:
        raise PrimitiveContractError("samples must be at least two")
    if not np.isfinite(duration) or duration <= 0.0:
        raise PrimitiveContractError("duration_s must be finite and positive")
    phase = np.linspace(0.0, 1.0, count, dtype=np.float64)
    return phase, phase * duration


def _slerp_samples(q_start: np.ndarray, q_goal: np.ndarray, phase: np.ndarray) -> np.ndarray:
    start = _quaternion(q_start, "start_quaternion_wxyz")
    goal = _quaternion(q_goal, "goal_quaternion_wxyz")
    samples: np.ndarray = np.stack(
        [quat_slerp(start, goal, float(value)) for value in phase]
    )
    return samples


def minimum_jerk_path(
    start_position: np.ndarray,
    start_quaternion_wxyz: np.ndarray,
    goal_position: np.ndarray,
    goal_quaternion_wxyz: np.ndarray,
    *,
    samples: int = 50,
    duration_s: float = 2.0,
    frame_id: str = "world",
) -> MotionPath:
    """Generate synchronized translational and rotational minimum-jerk progress."""

    start = _position(start_position, "start_position")
    goal = _position(goal_position, "goal_position")
    phase, timestamps = _sampling(samples, duration_s)
    blend = minimum_jerk_blend(phase)
    positions = start[None, :] + blend[:, None] * (goal - start)[None, :]
    quaternions = _slerp_samples(start_quaternion_wxyz, goal_quaternion_wxyz, blend)
    return MotionPath(positions, quaternions, timestamps, frame_id=frame_id)


def arc_path(
    start_position: np.ndarray,
    start_quaternion_wxyz: np.ndarray,
    *,
    center: np.ndarray,
    axis: np.ndarray,
    angle_rad: float,
    goal_quaternion_wxyz: Optional[np.ndarray] = None,
    rotate_tool_with_arc: bool = True,
    samples: int = 50,
    duration_s: float = 2.0,
    frame_id: str = "world",
) -> MotionPath:
    """Generate a minimum-jerk-timed circular arc about an axis through ``center``."""

    start = _position(start_position, "start_position")
    origin = _position(center, "center")
    direction = _unit_axis(axis)
    angle = float(angle_rad)
    if not np.isfinite(angle):
        raise PrimitiveContractError("angle_rad must be finite")
    phase, timestamps = _sampling(samples, duration_s)
    blend = minimum_jerk_blend(phase)
    radial = start - origin
    positions = []
    quaternions = []
    q_start = _quaternion(start_quaternion_wxyz, "start_quaternion_wxyz")
    q_goal = None
    if goal_quaternion_wxyz is not None:
        q_goal = _quaternion(goal_quaternion_wxyz, "goal_quaternion_wxyz")
    for value in blend:
        theta = angle * float(value)
        rotation_q = axis_angle_to_quat(direction * theta)
        rotation_matrix = _quaternion_to_matrix(rotation_q)
        positions.append(origin + rotation_matrix @ radial)
        if q_goal is not None:
            quaternions.append(quat_slerp(q_start, q_goal, float(value)))
        elif rotate_tool_with_arc:
            quaternions.append(quat_multiply(rotation_q, q_start))
        else:
            quaternions.append(q_start)
    return MotionPath(
        np.asarray(positions),
        np.asarray(quaternions),
        timestamps,
        frame_id=frame_id,
    )


def screw_path(
    start_position: np.ndarray,
    start_quaternion_wxyz: np.ndarray,
    *,
    axis_point: np.ndarray,
    axis: np.ndarray,
    angle_rad: float,
    translation_m: float,
    rotate_tool: bool = True,
    samples: int = 50,
    duration_s: float = 2.0,
    frame_id: str = "world",
) -> MotionPath:
    """Generate a screw path around an axis with coupled axial translation."""

    start = _position(start_position, "start_position")
    origin = _position(axis_point, "axis_point")
    direction = _unit_axis(axis)
    angle = float(angle_rad)
    translation = float(translation_m)
    if not np.isfinite(angle) or not np.isfinite(translation):
        raise PrimitiveContractError("angle_rad and translation_m must be finite")
    phase, timestamps = _sampling(samples, duration_s)
    blend = minimum_jerk_blend(phase)
    radial = start - origin
    q_start = _quaternion(start_quaternion_wxyz, "start_quaternion_wxyz")
    positions = []
    quaternions = []
    for value in blend:
        theta = angle * float(value)
        rotation_q = axis_angle_to_quat(direction * theta)
        rotated = _quaternion_to_matrix(rotation_q) @ radial
        positions.append(origin + rotated + direction * translation * float(value))
        quaternions.append(quat_multiply(rotation_q, q_start) if rotate_tool else q_start)
    return MotionPath(
        np.asarray(positions),
        np.asarray(quaternions),
        timestamps,
        frame_id=frame_id,
    )


def via_point_path(
    positions: Sequence[np.ndarray],
    quaternions_wxyz: Sequence[np.ndarray],
    *,
    samples_per_segment: int = 25,
    duration_s: float = 2.0,
    frame_id: str = "world",
) -> MotionPath:
    """Join pose waypoints with C2-compatible, zero-boundary-velocity jerk segments.

    Every supplied waypoint is retained exactly. Duration is allocated by a
    combined translation/rotation chord score, with equal allocation when all
    waypoints are identical.
    """

    points = np.asarray(positions, dtype=np.float64)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 2:
        raise PrimitiveContractError("positions must contain at least two 3D waypoints")
    if quaternions.shape != (len(points), 4):
        raise PrimitiveContractError("quaternions_wxyz must contain one quaternion per waypoint")
    count = int(samples_per_segment)
    if count < 2:
        raise PrimitiveContractError("samples_per_segment must be at least two")
    duration = float(duration_s)
    if not np.isfinite(duration) or duration <= 0.0:
        raise PrimitiveContractError("duration_s must be finite and positive")
    normalized_quaternions = np.stack(
        [
            _quaternion(value, f"quaternions_wxyz[{index}]")
            for index, value in enumerate(quaternions)
        ]
    )
    linear_chords = np.linalg.norm(np.diff(points, axis=0), axis=1)
    angular_chords = np.array(
        [
            2.0
            * np.arccos(
                np.clip(abs(float(np.dot(start, goal))), 0.0, 1.0)
            )
            for start, goal in zip(
                normalized_quaternions[:-1], normalized_quaternions[1:]
            )
        ]
    )
    scores = linear_chords + 0.1 * angular_chords
    if float(np.sum(scores)) <= 1e-12:
        scores = np.ones_like(scores)
    segment_durations = duration * scores / np.sum(scores)

    path_positions = []
    path_quaternions = []
    path_timestamps = []
    offset = 0.0
    for index, segment_duration in enumerate(segment_durations):
        segment = minimum_jerk_path(
            points[index],
            normalized_quaternions[index],
            points[index + 1],
            normalized_quaternions[index + 1],
            samples=count,
            duration_s=float(segment_duration),
            frame_id=frame_id,
        )
        start = 0 if index == 0 else 1
        path_positions.append(segment.positions[start:])
        path_quaternions.append(segment.quaternions_wxyz[start:])
        path_timestamps.append(segment.timestamps_s[start:] + offset)
        offset += float(segment_duration)
    return MotionPath(
        np.concatenate(path_positions),
        np.concatenate(path_quaternions),
        np.concatenate(path_timestamps),
        frame_id=frame_id,
    )


def affordance_normal_path(
    start_position: np.ndarray,
    start_quaternion_wxyz: np.ndarray,
    normal: np.ndarray,
    distance_m: float,
    *,
    goal_quaternion_wxyz: Optional[np.ndarray] = None,
    samples: int = 50,
    duration_s: float = 2.0,
    frame_id: str = "world",
) -> MotionPath:
    """Generate approach or retreat along a normalized affordance direction."""

    start = _position(start_position, "start_position")
    direction = _unit_axis(normal, "normal")
    distance = float(distance_m)
    if not np.isfinite(distance):
        raise PrimitiveContractError("distance_m must be finite")
    goal_q = start_quaternion_wxyz if goal_quaternion_wxyz is None else goal_quaternion_wxyz
    return minimum_jerk_path(
        start,
        start_quaternion_wxyz,
        start + direction * distance,
        goal_q,
        samples=samples,
        duration_s=duration_s,
        frame_id=frame_id,
    )


def _quaternion_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Small local conversion that avoids exposing SciPy's xyzw convention."""

    w, x, y, z = _quaternion(quaternion_wxyz, "quaternion_wxyz")
    matrix: np.ndarray = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return matrix
