"""Dependency-light tests for the relational FATROP preparation boundary."""

import numpy as np
import pytest

from dhb_xr.primitives import RelationalPoseAnchor


def test_anchor_guess_hits_targets_and_preserves_boundaries():
    from dhb_xr.optimization.relational_fatrop import _anchored_position_guess

    phase = np.linspace(0.0, 1.0, 11)
    reference = np.column_stack([phase, np.zeros((11, 2))])
    anchor = RelationalPoseAnchor(
        0.5,
        np.array([0.5, 0.2, -0.1]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        "offset-aperture",
    )

    result = _anchored_position_guess(reference, (anchor,))

    np.testing.assert_allclose(result[0], reference[0])
    np.testing.assert_allclose(result[5], anchor.position)
    np.testing.assert_allclose(result[-1], reference[-1])


def test_anchor_grid_rejects_phases_that_alias_the_same_solver_index():
    from dhb_xr.optimization.relational_fatrop import _anchor_grid

    reference = np.zeros((10, 3))
    anchors = (
        RelationalPoseAnchor(0.20, np.ones(3), [1, 0, 0, 0], "a"),
        RelationalPoseAnchor(0.21, np.ones(3), [1, 0, 0, 0], "b"),
    )

    try:
        _anchor_grid(reference, anchors)
    except ValueError as error:
        assert "duplicate solver indices" in str(error)
    else:
        raise AssertionError("aliased relational anchors must be rejected")


def test_persistent_solver_reuses_one_graph_across_requests():
    pytest.importorskip("rockit")
    from dhb_xr.optimization.relational_fatrop import (
        PersistentRelationalFatropRetargeter,
    )

    samples = 12
    phase = np.linspace(0.0, 1.0, samples)
    positions = np.column_stack([phase, 0.05 * np.sin(np.pi * phase), np.zeros(samples)])
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (samples, 1))
    retargeter = PersistentRelationalFatropRetargeter(
        positions,
        quaternions,
        samples=samples,
        regularization="dhb",
        use_fatrop=True,
    )
    anchor = RelationalPoseAnchor(0.5, [0.5, 0.08, 0.0], [1, 0, 0, 0], "mid")
    kwargs = {
        "start_position": positions[0],
        "start_quaternion_wxyz": quaternions[0],
        "goal_position": positions[-1],
        "goal_quaternion_wxyz": quaternions[-1],
        "relational_anchors": (anchor,),
    }

    first = retargeter.generate(**kwargs)
    second = retargeter.generate(**kwargs)

    assert first["ocp_build_count"] == second["ocp_build_count"] == 1
    assert first["ocp_solve_count"] == 1
    assert second["ocp_solve_count"] == 2
    assert first["max_anchor_position_error_m"] < 0.008
