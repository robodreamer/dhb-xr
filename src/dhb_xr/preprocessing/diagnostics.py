"""
Trajectory quality diagnostics for DHB encoding.

Provides analysis tools to detect potential encoding issues:
- 180-degree direction reversals
- Zero/near-zero motion segments
- Initial direction alignment
- Overall trajectory quality assessment
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import Enum


class PreprocessingRecommendation(Enum):
    """Recommended preprocessing actions."""
    NONE = "none"
    ALIGN_TO_X = "align_to_x"
    SMOOTH_TRAJECTORY = "smooth_trajectory"
    REMOVE_ZERO_MOTION = "remove_zero_motion"
    INTERPOLATE_REVERSALS = "interpolate_reversals"


@dataclass
class TrajectoryDiagnostics:
    """Comprehensive trajectory quality report."""

    # Basic info
    num_samples: int = 0
    trajectory_length: float = 0.0

    # Reversal detection
    num_reversals: int = 0
    reversal_indices: List[int] = field(default_factory=list)
    min_tangent_dot_product: float = 1.0  # -1 indicates 180° reversal

    # Zero-motion detection
    num_zero_motion_segments: int = 0
    zero_motion_segments: List[Tuple[int, int]] = field(default_factory=list)
    total_zero_motion_frames: int = 0

    # Initial direction analysis
    initial_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    initial_direction_magnitude: float = 0.0
    alignment_angle_to_x: float = 0.0  # Angle between initial direction and +x

    # Rotation analysis
    max_angular_velocity: float = 0.0
    total_rotation_angle: float = 0.0

    # Quality metrics
    is_well_conditioned: bool = True
    recommended_preprocessing: List[PreprocessingRecommendation] = field(default_factory=list)

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"Trajectory Diagnostics ({self.num_samples} samples, {self.trajectory_length:.3f}m total)",
            f"  Reversals: {self.num_reversals} (min dot: {self.min_tangent_dot_product:.3f})",
            f"  Zero-motion segments: {self.num_zero_motion_segments} ({self.total_zero_motion_frames} frames)",
            f"  Initial direction: {self.initial_direction} (angle to +x: {np.degrees(self.alignment_angle_to_x):.1f}°)",
            f"  Well-conditioned: {self.is_well_conditioned}",
        ]
        if self.recommended_preprocessing:
            lines.append(f"  Recommended: {[r.value for r in self.recommended_preprocessing]}")
        return "\n".join(lines)


def detect_reversals(
    positions: np.ndarray,
    threshold: float = -0.5,
) -> Tuple[List[int], float]:
    """
    Detect indices where motion direction reverses significantly.

    A reversal is detected when consecutive tangent vectors have a
    dot product below the threshold (e.g., -0.5 means > 120° turn).

    Args:
        positions: (N, 3) position trajectory.
        threshold: Dot product threshold for reversal detection.
            -1.0 = exact 180° reversal only
            -0.5 = 120° or larger turn
            0.0 = 90° or larger turn

    Returns:
        Tuple of (reversal_indices, min_dot_product).
    """
    positions = np.asarray(positions)
    if len(positions) < 3:
        return [], 1.0

    # Compute tangent vectors
    tangents = np.diff(positions, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)

    # Normalize tangents (avoid division by zero)
    eps = 1e-10
    tangents_normalized = tangents / (norms + eps)

    # Compute consecutive dot products
    reversal_indices = []
    min_dot = 1.0

    for i in range(len(tangents_normalized) - 1):
        # Skip if either tangent is near-zero
        if norms[i] < eps or norms[i + 1] < eps:
            continue

        dot = np.dot(tangents_normalized[i], tangents_normalized[i + 1])
        min_dot = min(min_dot, dot)

        if dot < threshold:
            reversal_indices.append(i + 1)  # Index in original positions

    return reversal_indices, min_dot


def detect_zero_motion(
    positions: np.ndarray,
    threshold: float = 1e-6,
    min_segment_length: int = 1,
) -> List[Tuple[int, int]]:
    """
    Detect contiguous segments with zero or near-zero motion.

    Args:
        positions: (N, 3) position trajectory.
        threshold: Distance threshold for zero motion.
        min_segment_length: Minimum number of consecutive frames to report.

    Returns:
        List of (start_index, end_index) tuples for zero-motion segments.
    """
    positions = np.asarray(positions)
    if len(positions) < 2:
        return []

    # Compute step distances
    steps = np.diff(positions, axis=0)
    distances = np.linalg.norm(steps, axis=1)

    # Find zero-motion frames
    is_zero = distances < threshold

    # Group into contiguous segments
    segments = []
    in_segment = False
    segment_start = 0

    for i, zero in enumerate(is_zero):
        if zero and not in_segment:
            in_segment = True
            segment_start = i
        elif not zero and in_segment:
            in_segment = False
            segment_length = i - segment_start
            if segment_length >= min_segment_length:
                segments.append((segment_start, i))

    # Handle segment at end
    if in_segment:
        segment_length = len(is_zero) - segment_start
        if segment_length >= min_segment_length:
            segments.append((segment_start, len(is_zero)))

    return segments


def compute_initial_direction(
    positions: np.ndarray,
    num_samples: int = 5,
) -> Tuple[np.ndarray, float]:
    """
    Compute the initial motion direction.

    Uses first few samples to robustly estimate initial direction,
    skipping zero-motion initial frames if present.

    Args:
        positions: (N, 3) position trajectory.
        num_samples: Number of initial samples to consider.

    Returns:
        Tuple of (direction_unit_vector, magnitude).
    """
    positions = np.asarray(positions)
    eps = 1e-10

    # Try successive differences until we find non-zero motion
    for i in range(min(num_samples, len(positions) - 1)):
        diff = positions[i + 1] - positions[i]
        magnitude = np.linalg.norm(diff)
        if magnitude > eps:
            return diff / magnitude, magnitude

    # If all initial motion is zero, try overall direction
    if len(positions) > 1:
        diff = positions[-1] - positions[0]
        magnitude = np.linalg.norm(diff)
        if magnitude > eps:
            return diff / magnitude, magnitude

    # Fallback to +x
    return np.array([1.0, 0.0, 0.0]), 0.0


def compute_alignment_angle(direction: np.ndarray, target: np.ndarray = None) -> float:
    """
    Compute angle between direction and target (default +x).

    Returns angle in radians.
    """
    if target is None:
        target = np.array([1.0, 0.0, 0.0])

    direction = np.asarray(direction)
    target = np.asarray(target)

    # Normalize
    d_norm = np.linalg.norm(direction)
    t_norm = np.linalg.norm(target)

    if d_norm < 1e-10 or t_norm < 1e-10:
        return 0.0

    direction = direction / d_norm
    target = target / t_norm

    # Compute angle
    dot = np.clip(np.dot(direction, target), -1.0, 1.0)
    return np.arccos(dot)


def analyze_trajectory(
    positions: np.ndarray,
    quaternions: Optional[np.ndarray] = None,
    reversal_threshold: float = -0.5,
    zero_motion_threshold: float = 1e-6,
) -> TrajectoryDiagnostics:
    """
    Comprehensive trajectory quality analysis.

    Detects potential issues that could cause DHB encoding problems:
    - 180° direction reversals (cause Euler singularities)
    - Zero-motion segments (cause undefined tangent frames)
    - Misaligned initial direction (reconstruction error with default frames)

    Args:
        positions: (N, 3) position trajectory.
        quaternions: Optional (N, 4) wxyz quaternion trajectory.
        reversal_threshold: Dot product threshold for reversal detection.
        zero_motion_threshold: Distance threshold for zero motion.

    Returns:
        TrajectoryDiagnostics with analysis results and recommendations.
    """
    positions = np.asarray(positions)
    diagnostics = TrajectoryDiagnostics()

    # Basic info
    diagnostics.num_samples = len(positions)
    if len(positions) > 1:
        steps = np.diff(positions, axis=0)
        diagnostics.trajectory_length = np.sum(np.linalg.norm(steps, axis=1))

    # Reversal detection
    reversal_indices, min_dot = detect_reversals(positions, reversal_threshold)
    diagnostics.num_reversals = len(reversal_indices)
    diagnostics.reversal_indices = reversal_indices
    diagnostics.min_tangent_dot_product = min_dot

    # Zero-motion detection
    zero_segments = detect_zero_motion(positions, zero_motion_threshold)
    diagnostics.num_zero_motion_segments = len(zero_segments)
    diagnostics.zero_motion_segments = zero_segments
    diagnostics.total_zero_motion_frames = sum(end - start for start, end in zero_segments)

    # Initial direction
    init_dir, init_mag = compute_initial_direction(positions)
    diagnostics.initial_direction = init_dir
    diagnostics.initial_direction_magnitude = init_mag
    diagnostics.alignment_angle_to_x = compute_alignment_angle(init_dir)

    # Rotation analysis (if quaternions provided)
    if quaternions is not None:
        quaternions = np.asarray(quaternions)
        if len(quaternions) > 1:
            # Compute angular velocities
            from dhb_xr.core import geometry as geom
            angular_velocities = []
            for i in range(len(quaternions) - 1):
                R_prev = geom.quat_to_rot(quaternions[i])
                R_curr = geom.quat_to_rot(quaternions[i + 1])
                R_rel = R_curr @ R_prev.T
                axis_angle = geom.rot_to_axis_angle(R_rel)
                angular_velocities.append(np.linalg.norm(axis_angle))

            if angular_velocities:
                diagnostics.max_angular_velocity = max(angular_velocities)
                diagnostics.total_rotation_angle = sum(angular_velocities)

    # Determine if well-conditioned
    diagnostics.is_well_conditioned = (
        diagnostics.num_reversals == 0 and
        diagnostics.num_zero_motion_segments == 0 and
        diagnostics.alignment_angle_to_x < np.radians(30)  # Within 30° of +x
    )

    # Generate recommendations
    recommendations = []

    if diagnostics.alignment_angle_to_x > np.radians(10):
        recommendations.append(PreprocessingRecommendation.ALIGN_TO_X)

    if diagnostics.num_reversals > 0:
        recommendations.append(PreprocessingRecommendation.SMOOTH_TRAJECTORY)
        if diagnostics.min_tangent_dot_product < -0.8:
            recommendations.append(PreprocessingRecommendation.INTERPOLATE_REVERSALS)

    if diagnostics.num_zero_motion_segments > 0:
        recommendations.append(PreprocessingRecommendation.REMOVE_ZERO_MOTION)

    diagnostics.recommended_preprocessing = recommendations

    return diagnostics
