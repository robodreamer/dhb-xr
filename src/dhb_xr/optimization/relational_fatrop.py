"""Persistent matched FATROP retargeters with relational pose anchors.

This module isolates the representation question.  The ``dhb`` and
``cartesian`` variants share the same state, dynamics, anchor penalties,
boundary constraints, initialization, solver, and output contract.  They
differ only in whether the reusable OCP regularizes DHB invariants or sampled
Cartesian positions.  Intermediate orientation anchors currently use one
shared deterministic quaternion correction; the comparative claim is therefore
limited to linear-path representation until angular-state dynamics are added.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Literal, Sequence

import numpy as np

from dhb_xr.core import geometry as geom
from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.optimization.fatrop_solver import (
    FatropTrajectoryGenerator,
    _boundary_adjusted_position_reference,
    _boundary_adjusted_quaternion_reference,
    _rollout_linear_frames,
)
from dhb_xr.primitives.sources import RelationalPoseAnchor

RegularizationMode = Literal["dhb", "cartesian"]


class PersistentRelationalFatropRetargeter:
    """Reuse one parameterized FATROP graph across relational retarget requests."""

    def __init__(
        self,
        reference_positions: np.ndarray,
        reference_quaternions_wxyz: np.ndarray,
        *,
        samples: int,
        regularization: RegularizationMode,
        use_fatrop: bool = True,
        invariant_weight: float = 1.0,
        cartesian_weight: float = 1.0e6,
        anchor_weight: float = 1.0e8,
        max_iters: int = 300,
        verbose: bool = False,
    ) -> None:
        positions = np.asarray(reference_positions, dtype=np.float64)
        quaternions = np.asarray(reference_quaternions_wxyz, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) < 3:
            raise ValueError("reference_positions must have shape (N, 3), N >= 3")
        if quaternions.shape != (len(positions), 4):
            raise ValueError("reference_quaternions_wxyz must have shape (N, 4)")
        if regularization not in {"dhb", "cartesian"}:
            raise ValueError("regularization must be 'dhb' or 'cartesian'")
        if samples < 8:
            raise ValueError("samples must be at least eight")
        self.reference_positions = positions.copy()
        self.reference_quaternions_wxyz = _continuous_quaternions(quaternions)
        self.samples = int(samples)
        self.regularization = regularization
        self.solve_count = 0
        started = perf_counter()
        self.generator = FatropTrajectoryGenerator(
            N=self.samples,
            use_fatrop=use_fatrop,
            w_invariants=invariant_weight if regularization == "dhb" else 0.0,
            w_path=cartesian_weight if regularization == "cartesian" else 0.0,
            w_anchor_position=anchor_weight,
            enable_relational_anchors=True,
            compile_repeated_solve=True,
            max_iters=max_iters,
            verbose=verbose,
        )
        self.build_ms = (perf_counter() - started) * 1_000.0

    @property
    def source_id(self) -> str:
        solver = "fatrop" if self.generator.use_fatrop else "ipopt"
        return f"{self.regularization}-regularized:{solver}_relational"

    def generate(
        self,
        *,
        start_position: np.ndarray,
        start_quaternion_wxyz: np.ndarray,
        goal_position: np.ndarray,
        goal_quaternion_wxyz: np.ndarray,
        relational_anchors: Sequence[RelationalPoseAnchor],
    ) -> dict[str, Any]:
        """Solve one request without rebuilding the OCP graph."""

        start_position = np.asarray(start_position, dtype=np.float64)
        goal_position = np.asarray(goal_position, dtype=np.float64)
        start_quaternion = _unit_quaternion(start_quaternion_wxyz)
        goal_quaternion = _unit_quaternion(goal_quaternion_wxyz)
        anchors = tuple(relational_anchors)
        phases = [anchor.phase for anchor in anchors]
        if phases != sorted(phases) or len(phases) != len(set(phases)):
            raise ValueError("relational anchor phases must be unique and increasing")

        start_pose = {"position": start_position, "quaternion": start_quaternion}
        goal_pose = {"position": goal_position, "quaternion": goal_quaternion}
        position_reference = _boundary_adjusted_position_reference(
            self.reference_positions,
            start_pose,
            goal_pose,
            self.samples,
        )
        quaternion_reference = _boundary_adjusted_quaternion_reference(
            self.reference_quaternions_wxyz,
            start_pose,
            goal_pose,
            self.samples,
        )
        position_initial = _anchored_position_guess(position_reference, anchors)
        quaternion_output = _anchored_quaternion_reference(quaternion_reference, anchors)
        anchor_positions, anchor_mask, anchor_indices = _anchor_grid(
            position_reference,
            anchors,
        )

        encoding = encode_dhb_dr(
            position_reference,
            quaternion_reference,
            init_pose=start_pose,
            method=EncodingMethod.POSITION,
            use_default_initial_frames=False,
            dhb_method=DHBMethod.DOUBLE_REFLECTION,
        )
        demo_lin = _resample_rows(encoding["linear_motion_invariants"], self.samples)
        demo_ang = _resample_rows(encoding["angular_motion_invariants"], self.samples)
        start_rot = encoding["linear_frame_initial"][:3, :3]
        frame_rollout = _rollout_linear_frames(encoding["linear_frame_initial"], demo_lin)
        rotation_initial = (
            np.concatenate([start_rot[None, :, :], frame_rollout[:-1]], axis=0)
            if len(frame_rollout)
            else np.repeat(start_rot[None, :, :], self.samples, axis=0)
        )
        if len(rotation_initial) < self.samples:
            rotation_initial = np.concatenate(
                [
                    rotation_initial,
                    np.repeat(
                        rotation_initial[-1][None, :, :],
                        self.samples - len(rotation_initial),
                        axis=0,
                    ),
                ],
                axis=0,
            )

        result = self.generator.generate(
            demo_lin_invariants=demo_lin,
            demo_ang_invariants=demo_ang,
            start_pos=start_position,
            start_rot=start_rot,
            goal_pos=goal_position,
            goal_rot=rotation_initial[-1],
            init_positions=position_initial,
            init_rotations=rotation_initial,
            position_reference=position_reference,
            anchor_positions=anchor_positions,
            anchor_mask=anchor_mask,
        )
        self.solve_count += 1
        positions = np.asarray(result["positions"], dtype=np.float64).copy()
        positions[0] = start_position
        positions[-1] = goal_position
        quaternion_output[0] = start_quaternion
        quaternion_output[-1] = goal_quaternion
        anchor_errors = [
            float(np.linalg.norm(positions[index] - anchor.position))
            for index, anchor in zip(anchor_indices, anchors)
        ]
        return {
            "positions": positions,
            "quaternions": quaternion_output,
            "linear_invariants": result["linear_invariants"],
            "angular_invariants": result["angular_invariants"],
            "success": bool(result["success"]),
            "solver": "fatrop" if self.generator.use_fatrop else "ipopt",
            "regularization": self.regularization,
            "solve_time": float(result["solve_time"]),
            "ocp_build_ms": self.build_ms,
            "ocp_build_count": 1,
            "ocp_solve_count": self.solve_count,
            "anchor_indices": anchor_indices,
            "anchor_position_errors_m": anchor_errors,
            "max_anchor_position_error_m": max(anchor_errors, default=0.0),
            "orientation_anchor_policy": "shared_piecewise_quaternion_residual",
            "comparison_scope": "linear_representation_only",
        }


def _resample_rows(values: np.ndarray, samples: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == samples:
        return values.copy()
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, samples)
    return np.column_stack(
        [np.interp(target, source, values[:, axis]) for axis in range(values.shape[1])]
    )


def _anchor_grid(
    reference: np.ndarray,
    anchors: Sequence[RelationalPoseAnchor],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    positions = np.asarray(reference, dtype=np.float64).copy()
    mask = np.zeros(len(reference), dtype=np.float64)
    indices = []
    for anchor in anchors:
        index = int(round(anchor.phase * (len(reference) - 1)))
        index = min(max(index, 1), len(reference) - 2)
        positions[index] = anchor.position
        mask[index] = 1.0
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError("relational anchors map to duplicate solver indices")
    return positions, mask, indices


def _anchored_position_guess(
    reference: np.ndarray,
    anchors: Sequence[RelationalPoseAnchor],
) -> np.ndarray:
    phases = np.linspace(0.0, 1.0, len(reference))
    key_phases = np.array([0.0, *[anchor.phase for anchor in anchors], 1.0])
    key_targets = np.vstack([reference[0], *[anchor.position for anchor in anchors], reference[-1]])
    reference_at_keys = np.column_stack(
        [np.interp(key_phases, phases, reference[:, axis]) for axis in range(3)]
    )
    residuals = key_targets - reference_at_keys
    output = reference.copy()
    for segment in range(len(key_phases) - 1):
        lo, hi = key_phases[segment : segment + 2]
        selected = (phases >= lo) & (phases <= hi)
        local = np.clip((phases[selected] - lo) / max(hi - lo, 1.0e-12), 0.0, 1.0)
        blend = 10.0 * local**3 - 15.0 * local**4 + 6.0 * local**5
        correction = (1.0 - blend[:, None]) * residuals[segment] + blend[:, None] * residuals[
            segment + 1
        ]
        output[selected] = reference[selected] + correction
    _, _, anchor_indices = _anchor_grid(reference, anchors)
    for index, anchor in zip(anchor_indices, anchors):
        output[index] = anchor.position
    return output


def _anchored_quaternion_reference(
    reference: np.ndarray,
    anchors: Sequence[RelationalPoseAnchor],
) -> np.ndarray:
    output = _continuous_quaternions(reference)
    phases = np.linspace(0.0, 1.0, len(output))
    key_phases = [0.0, *[anchor.phase for anchor in anchors], 1.0]
    key_targets = [output[0], *[anchor.quaternion_wxyz for anchor in anchors], output[-1]]
    key_reference = [_quaternion_at_phase(output, phase) for phase in key_phases]
    residuals = [
        geom.quat_multiply(target, geom.quat_inv(base))
        for target, base in zip(key_targets, key_reference)
    ]
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    del identity
    for segment, (lo, hi) in enumerate(zip(key_phases[:-1], key_phases[1:])):
        selected = np.flatnonzero((phases >= lo) & (phases <= hi))
        for index in selected:
            local = np.clip((phases[index] - lo) / max(hi - lo, 1.0e-12), 0.0, 1.0)
            blend = 10.0 * local**3 - 15.0 * local**4 + 6.0 * local**5
            correction = geom.quat_slerp(residuals[segment], residuals[segment + 1], float(blend))
            output[index] = geom.quat_multiply(correction, reference[index])
            output[index] = _unit_quaternion(output[index])
    for anchor in anchors:
        index = int(round(anchor.phase * (len(output) - 1)))
        output[index] = anchor.quaternion_wxyz
    return _continuous_quaternions(output)


def _quaternion_at_phase(quaternions: np.ndarray, phase: float) -> np.ndarray:
    scaled = phase * (len(quaternions) - 1)
    lower = int(np.floor(scaled))
    upper = min(lower + 1, len(quaternions) - 1)
    return geom.quat_slerp(quaternions[lower], quaternions[upper], scaled - lower)


def _continuous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    output = np.asarray(quaternions, dtype=np.float64).copy()
    output[0] = _unit_quaternion(output[0])
    for index in range(1, len(output)):
        output[index] = geom.quat_ensure_continuous(
            _unit_quaternion(output[index]), output[index - 1]
        )
    return output


def _unit_quaternion(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise ValueError("quaternion must be finite and nonzero")
    return value / norm
