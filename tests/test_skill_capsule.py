"""Tests for provenance-preserving object-centric skill capsules."""

import numpy as np
import pytest

from dhb_xr.core import geometry as geom
from dhb_xr.primitives import (
    ActionTiming,
    EndEffectorCommand,
    EndEffectorCommandType,
    EvidenceLevel,
    MotionPath,
    ObjectCentricSkillCapsule,
    PrimitiveContractError,
    PrimitiveEvidence,
    RelationalPoseAnchor,
    ScheduledEndEffectorAction,
    SparseMotionCompletionRequest,
)
from dhb_xr.primitives.sources import PrimitiveRequest


def _paths(samples: int = 33) -> tuple[MotionPath, MotionPath]:
    phase = np.linspace(0.0, 1.0, samples)
    object_positions = np.column_stack(
        [0.4 + 0.1 * phase, -0.05 + 0.03 * phase, 0.45 + 0.02 * phase]
    )
    object_quaternions = np.stack(
        [geom.axis_angle_to_quat([0.0, 0.0, 0.4 * value]) for value in phase]
    )
    relative_positions = np.column_stack(
        [0.04 * np.sin(np.pi * phase), 0.08 * phase, 0.12 + 0.02 * np.sin(2 * np.pi * phase)]
    )
    relative_quaternions = np.stack(
        [
            geom.axis_angle_to_quat(
                [0.5 * value, 0.1 * np.sin(np.pi * value), 0.0]
            )
            for value in phase
        ]
    )
    tcp_positions = np.empty_like(relative_positions)
    tcp_quaternions = np.empty_like(relative_quaternions)
    for index in range(samples):
        tcp_positions[index] = (
            object_positions[index]
            + geom.quat_to_rot(object_quaternions[index]) @ relative_positions[index]
        )
        tcp_quaternions[index] = geom.quat_multiply(
            object_quaternions[index], relative_quaternions[index]
        )
    timestamps = 2.0 * phase
    return (
        MotionPath(tcp_positions, tcp_quaternions, timestamps, "world"),
        MotionPath(object_positions, object_quaternions, timestamps, "world"),
    )


def _capsule(tcp: MotionPath, obj: MotionPath) -> ObjectCentricSkillCapsule:
    return ObjectCentricSkillCapsule.capture(
        capsule_id="carry-through-aperture-v1",
        semantic_label="carry object through aperture",
        tcp_world=tcp,
        object_world=obj,
        relation_events={"pre_aperture": 0.30, "inside_aperture": 0.62, "settled": 1.0},
        end_effector_actions=(
            ScheduledEndEffectorAction(
                EndEffectorCommand(EndEffectorCommandType.HOLD),
                ActionTiming.DURING_MOTION,
            ),
        ),
        evidence=PrimitiveEvidence(
            EvidenceLevel.SIM_VERIFIED,
            "capsule-unit-test",
            source_revision="abc123",
            license_id="MIT",
        ),
        provenance={
            "source_uri": "artifact://capsule-unit-test",
            "source_revision": "abc123",
            "action_convention": "tcp_pose_wxyz+binary_gripper",
            "license_id": "MIT",
            "dataset": "synthetic-unit-test",
        },
        descriptor_scales=(8, 16),
    )


def test_capsule_preserves_object_relative_motion_and_round_trips(tmp_path):
    tcp, obj = _paths()
    capsule = _capsule(tcp, obj)

    assert capsule.descriptor.scales == (8, 16, 33)
    assert capsule.descriptor.frame_id == "world_relative_to_initial_object_pose"
    assert [item.relation_id for item in capsule.relations] == [
        "pre_aperture",
        "inside_aperture",
        "settled",
    ]
    destination = tmp_path / "capsule.json"
    capsule.save(destination)
    restored = ObjectCentricSkillCapsule.load(destination)

    np.testing.assert_allclose(restored.tcp_in_object.positions, capsule.tcp_in_object.positions)
    np.testing.assert_allclose(
        restored.descriptor.windows[-1].invariants,
        capsule.descriptor.windows[-1].invariants,
    )
    assert restored.provenance["action_convention"] == "tcp_pose_wxyz+binary_gripper"


def test_object_centric_descriptor_is_unchanged_by_shared_world_transform():
    tcp, obj = _paths()
    reference = _capsule(tcp, obj)
    world_rotation = geom.axis_angle_to_quat([0.2, -0.4, 0.7])
    rotation = geom.quat_to_rot(world_rotation)
    translation = np.array([0.7, -0.3, 0.2])

    def transformed(path: MotionPath) -> MotionPath:
        positions = path.positions @ rotation.T + translation
        quaternions = np.stack(
            [geom.quat_multiply(world_rotation, value) for value in path.quaternions_wxyz]
        )
        return MotionPath(positions, quaternions, path.timestamps_s, path.frame_id)

    moved = _capsule(transformed(tcp), transformed(obj))

    np.testing.assert_allclose(
        moved.tcp_in_object.positions,
        reference.tcp_in_object.positions,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        moved.tcp_in_object.quaternions_wxyz,
        reference.tcp_in_object.quaternions_wxyz,
        atol=1e-8,
    )
    for first, second in zip(reference.descriptor.windows, moved.descriptor.windows, strict=True):
        np.testing.assert_allclose(first.invariants, second.invariants, atol=1e-8)


def test_capsule_requires_portable_provenance():
    tcp, obj = _paths()
    with pytest.raises(PrimitiveContractError, match="source_revision"):
        ObjectCentricSkillCapsule.capture(
            capsule_id="missing-provenance",
            semantic_label="test",
            tcp_world=tcp,
            object_world=obj,
            relation_events={},
            end_effector_actions=(),
            evidence=PrimitiveEvidence(EvidenceLevel.PLANNED, "test"),
            provenance={
                "source_uri": "artifact://test",
                "action_convention": "tcp_wxyz",
                "license_id": "MIT",
            },
        )


def test_sparse_completion_request_exactly_attributes_anchor_relations():
    primitive_request = PrimitiveRequest(
        primitive_id="complete-aperture-motion",
        start_position=[0.4, 0.0, 0.5],
        start_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        goal_position=[0.6, 0.1, 0.5],
        goal_quaternion_wxyz=[0.707, 0.707, 0.0, 0.0],
        relational_anchors=(
            RelationalPoseAnchor(0.5, [0.5, 0.04, 0.55], [1.0, 0.0, 0.0, 0.0], "aperture"),
        ),
    )
    request = SparseMotionCompletionRequest(
        "carry-through-aperture-v1",
        primitive_request,
        ("aperture",),
        "mujoco_world",
    )
    assert request.observed_relation_ids == ("aperture",)

    with pytest.raises(PrimitiveContractError, match="exactly attribute"):
        SparseMotionCompletionRequest(
            "carry-through-aperture-v1",
            primitive_request,
            ("different",),
            "mujoco_world",
        )
