import numpy as np

from scripts.starvla_dhb_predict_smoke import _add_probability_summary


def test_predict_smoke_labels_one_logit_contact_state_as_close_probability():
    summary = {}
    traj = {
        "contact_mode": "gripper_state",
        "pred_contact_probs": np.array([[[0.2], [0.8]]], dtype=np.float32),
    }

    _add_probability_summary(summary, traj)

    assert summary["pred_contact_mode"] == "gripper_state"
    assert summary["pred_close_probs_head"] == [0.20000000298023224, 0.800000011920929]
    assert "pred_open_probs_head" not in summary


def test_predict_smoke_keeps_two_logit_contact_state_open_close_order():
    summary = {}
    traj = {
        "contact_mode": "gripper_state",
        "pred_contact_probs": np.array(
            [[[0.8, 0.2], [0.3, 0.7]]],
            dtype=np.float32,
        ),
    }

    _add_probability_summary(summary, traj)

    assert summary["pred_open_probs_head"] == [0.800000011920929, 0.30000001192092896]
    assert summary["pred_close_probs_head"] == [0.20000000298023224, 0.699999988079071]


def test_predict_smoke_labels_one_logit_gripper_state_as_close_probability():
    summary = {}
    traj = {
        "pred_gripper_state_probs": np.array([[[0.4], [0.9]]], dtype=np.float32),
    }

    _add_probability_summary(summary, traj)

    assert summary["pred_close_probs_head"] == [0.4000000059604645, 0.8999999761581421]
    assert "pred_open_probs_head" not in summary
