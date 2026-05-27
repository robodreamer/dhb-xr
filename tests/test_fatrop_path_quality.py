"""Path-quality regressions for FATROP trajectory adaptation."""

import numpy as np
import pytest


def _max_turn_angle(positions: np.ndarray) -> float:
    deltas = np.diff(np.asarray(positions, dtype=float), axis=0)
    angles = []
    for a, b in zip(deltas[:-1], deltas[1:]):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na > 1e-5 and nb > 1e-5:
            angles.append(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)))
    return float(max(angles, default=0.0))


def _max_second_difference_ratio(positions: np.ndarray) -> float:
    second = np.linalg.norm(
        np.diff(np.asarray(positions, dtype=float), n=2, axis=0), axis=1
    )
    if len(second) == 0:
        return 0.0
    median = float(np.median(second))
    return float(np.max(second) / max(median, 1e-9))


def _oxe_austin_ee_fixture() -> tuple[np.ndarray, np.ndarray]:
    positions = np.array(
        [
            [0.30702, -0.0, 0.48687],
            [0.30702, -0.0, 0.48687],
            [0.307018, -0.0, 0.486866],
            [0.30702, -2e-06, 0.486868],
            [0.307021, -0.0, 0.486869],
            [0.307018, 1e-06, 0.486863],
            [0.30702, -0.0, 0.486866],
            [0.307015, -2e-06, 0.486679],
            [0.306853, -2.6e-05, 0.485669],
            [0.306641, -6.7e-05, 0.484122],
            [0.306364, -0.000179, 0.482189],
            [0.306255, -0.000283, 0.480401],
            [0.306452, -0.000385, 0.479053],
            [0.307105, -0.000572, 0.477196],
            [0.308209, -0.000879, 0.474718],
            [0.309474, -0.001311, 0.471914],
            [0.310227, -0.00206, 0.46858],
            [0.310631, -0.00285, 0.46534],
            [0.310755, -0.003691, 0.462539],
            [0.310593, -0.004498, 0.459859],
            [0.310382, -0.005309, 0.457178],
            [0.310105, -0.006163, 0.454047],
            [0.30992, -0.006834, 0.450939],
            [0.30968, -0.007551, 0.447104],
            [0.309501, -0.008319, 0.44286],
            [0.309528, -0.009087, 0.4392],
            [0.309563, -0.010066, 0.434682],
            [0.309801, -0.011145, 0.429764],
            [0.3102, -0.01225, 0.425125],
            [0.310407, -0.013431, 0.419708],
            [0.311013, -0.014555, 0.413956],
            [0.311405, -0.01584, 0.407247],
            [0.311638, -0.017294, 0.400525],
            [0.311996, -0.018809, 0.393845],
            [0.312457, -0.020197, 0.387225],
            [0.312923, -0.021381, 0.380964],
            [0.313311, -0.022291, 0.374701],
            [0.313516, -0.022838, 0.369975],
            [0.313676, -0.023324, 0.364014],
            [0.313483, -0.023798, 0.358043],
            [0.313239, -0.024354, 0.352305],
            [0.312695, -0.025149, 0.346996],
            [0.312273, -0.026101, 0.342741],
            [0.311972, -0.026944, 0.339444],
            [0.311596, -0.027674, 0.336285],
            [0.31099, -0.028334, 0.333407],
            [0.310177, -0.029105, 0.330144],
            [0.309423, -0.029993, 0.327129],
            [0.308864, -0.030817, 0.324979],
            [0.308387, -0.031917, 0.322978],
            [0.3078, -0.033268, 0.320841],
            [0.30725, -0.034609, 0.318619],
            [0.306611, -0.036047, 0.315992],
            [0.306204, -0.037475, 0.313497],
            [0.30577, -0.038761, 0.311188],
            [0.304984, -0.039848, 0.308659],
            [0.30425, -0.040638, 0.306167],
            [0.303143, -0.041433, 0.303173],
            [0.302237, -0.042245, 0.30036],
            [0.300896, -0.04318, 0.297047],
            [0.299463, -0.04407, 0.293755],
            [0.298383, -0.044782, 0.290931],
            [0.297166, -0.045617, 0.287626],
            [0.295771, -0.046318, 0.284383],
            [0.294363, -0.04681, 0.281319],
            [0.292896, -0.047137, 0.278608],
            [0.291437, -0.047303, 0.276106],
            [0.289393, -0.047406, 0.273389],
            [0.287441, -0.047425, 0.270683],
            [0.285116, -0.047608, 0.267462],
            [0.282895, -0.047882, 0.264329],
            [0.280333, -0.04817, 0.261051],
            [0.277958, -0.048358, 0.25793],
            [0.276256, -0.048357, 0.255538],
            [0.273863, -0.048321, 0.252756],
            [0.271605, -0.048226, 0.250402],
            [0.270534, -0.048204, 0.249052],
            [0.269779, -0.048479, 0.24874],
            [0.269206, -0.048699, 0.248483],
            [0.268766, -0.048937, 0.248336],
        ],
        dtype=float,
    )
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(positions), 1))
    return positions, quats


def test_fatrop_tracks_boundary_adjusted_path_without_zigzag():
    pytest.importorskip("rockit")
    from dhb_xr.core import geometry as geom
    from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

    t = np.linspace(0.0, 1.0, 40)
    pos_data = np.column_stack(
        [
            0.4 + 0.15 * t,
            -0.05 + 0.04 * np.sin(np.pi * t),
            0.35 - 0.12 * t,
        ]
    )
    quat_data = np.asarray(
        [geom.axis_angle_to_quat(np.array([0.0, 0.0, 0.2 * ti])) for ti in t],
        dtype=float,
    )
    goal_delta = np.array([0.16, 0.12, -0.04])

    result = generate_trajectory_fatrop(
        pos_data=pos_data,
        quat_data=quat_data,
        pose_target_init={"position": pos_data[0], "quaternion": quat_data[0]},
        pose_target_final={
            "position": pos_data[-1] + goal_delta,
            "quaternion": quat_data[-1],
        },
        traj_length=40,
        use_fatrop=True,
        verbose=False,
    )

    positions = np.asarray(result["positions"], dtype=float)
    np.testing.assert_allclose(positions[0], pos_data[0], atol=1e-8)
    np.testing.assert_allclose(positions[-1], pos_data[-1] + goal_delta, atol=1e-8)
    assert _max_turn_angle(positions) < 2.0


def test_fatrop_converges_without_kink_for_real_teleop_like_path():
    pytest.importorskip("rockit")
    from dhb_xr.optimization.fatrop_solver import generate_trajectory_fatrop

    pos_data, quat_data = _oxe_austin_ee_fixture()
    goal_delta = np.array([0.16, 0.12, -0.04])

    result = generate_trajectory_fatrop(
        pos_data=pos_data,
        quat_data=quat_data,
        pose_target_init={"position": pos_data[0], "quaternion": quat_data[0]},
        pose_target_final={
            "position": pos_data[-1] + goal_delta,
            "quaternion": quat_data[-1],
        },
        traj_length=len(pos_data),
        use_fatrop=True,
        verbose=False,
    )

    positions = np.asarray(result["positions"], dtype=float)
    assert result["success"] is True
    assert result["ocp_success"] is True
    assert result["path_quality_repaired"] is False
    assert result["solver"] == "fatrop"
    assert _max_second_difference_ratio(positions) < 8.0
    assert _max_turn_angle(positions) < 1.4


def test_public_cusadi_backend_matches_fatrop_for_real_teleop_like_path():
    pytest.importorskip("rockit")
    from dhb_xr.core import geometry as geom
    from dhb_xr import optimization

    pos_data, quat_data = _oxe_austin_ee_fixture()
    t = np.linspace(0.0, 1.0, len(pos_data))
    quat_data = np.asarray(
        [
            geom.axis_angle_to_quat(
                np.array(
                    [
                        0.16 * np.sin(np.pi * ti),
                        -0.08 * ti,
                        0.28 * ti,
                    ]
                )
            )
            for ti in t
        ],
        dtype=float,
    )
    goal_delta = np.array([0.16, 0.12, -0.04])
    start_pose = {"position": pos_data[0], "quaternion": quat_data[0]}
    goal_pose = {
        "position": pos_data[-1] + goal_delta,
        "quaternion": quat_data[-1],
    }

    fatrop = optimization.generate_trajectory(
        pos_data=pos_data,
        quat_data=quat_data,
        pose_target_init=start_pose,
        pose_target_final=goal_pose,
        traj_length=len(pos_data),
        backend="fatrop",
        allow_fallback=False,
    )
    cusadi = optimization.generate_trajectory(
        pos_data=pos_data,
        quat_data=quat_data,
        pose_target_init=start_pose,
        pose_target_final=goal_pose,
        traj_length=len(pos_data),
        backend="cusadi",
        allow_fallback=False,
    )

    assert cusadi["solver"] == "cusadi"
    assert cusadi["optimizer_solver"] == "cusadi-adapt"
    assert cusadi["requested_backend"] == "cusadi"
    assert cusadi["cusadi_adaptation"] == "casadi-function"
    assert cusadi["cusadi_decode"] in {"cpu", "gpu"}
    assert np.isfinite(cusadi["cusadi_decoded_pos_data"]).all()
    assert np.isfinite(cusadi["cusadi_decoded_quat_data"]).all()
    assert cusadi["path_quality_repaired"] is False
    np.testing.assert_allclose(
        cusadi["adapted_pos_data"],
        fatrop["adapted_pos_data"],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        cusadi["adapted_quat_data"],
        fatrop["adapted_quat_data"],
        atol=1e-8,
    )
