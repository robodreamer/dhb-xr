import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STARVLA = ROOT / "vendor" / "starVLA"
if str(STARVLA) not in sys.path:
    sys.path.insert(0, str(STARVLA))

from starVLA.dataloader.policy_trace_contact_datasets import PolicyTraceContactDataset


def _write_policy_trace_fixture(tmp_path: Path) -> Path:
    npz_path = tmp_path / "chunk_0000_t0000.npz"
    np.savez_compressed(
        npz_path,
        agentview_image=np.full((16, 16, 3), 127, dtype=np.uint8),
        wrist_image=np.full((16, 16, 3), 64, dtype=np.uint8),
        state_window=np.arange(32, dtype=np.float32).reshape(4, 8),
        ee_pos_live=np.zeros(3, dtype=np.float32),
        chunk_positions=np.zeros((4, 3), dtype=np.float32),
        close_probs=np.zeros(4, dtype=np.float32),
        open_probs=np.zeros(4, dtype=np.float32),
    )
    summary_path = tmp_path / "rollout_summary.json"
    summary_path.write_text(
        json.dumps({"task_language": "move the bowl to the plate"}),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "rollout_json": str(summary_path),
                "chunk_index": 0,
                "t_start": 0,
                "n_use": 4,
                "npz_path": str(npz_path),
                "target_close": [0.0, 1.0, 0.0, 0.0],
                "train_mask": [0.0, 1.0, 1.0, 0.0],
                "hard_negative_mask": [0.0, 0.0, 1.0, 0.0],
                "labels": ["background", "positive", "hard_negative", "background"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_policy_trace_contact_dataset_packs_qwendhb_sample(tmp_path):
    manifest_path = _write_policy_trace_fixture(tmp_path)
    dataset = PolicyTraceContactDataset(
        {
            "policy_trace_contact_manifest_jsonl": [str(manifest_path)],
            "action_window_size": 8,
            "action_dim": 7,
            "state_history_window": 4,
            "state_dim": 8,
            "policy_trace_image_size": 32,
            "policy_trace_include_wrist": True,
        }
    )

    sample = dataset[0]

    assert sample["lang"] == "move the bowl to the plate"
    assert len(sample["image"]) == 2
    assert sample["image"][0].size == (32, 32)
    assert sample["action"].shape == (8, 7)
    assert sample["state"].shape == (4, 8)
    assert sample["dhb.close_contact"].shape == (8, 1)
    assert sample["dhb.action_valid_mask"].shape == (8,)
    np.testing.assert_allclose(sample["dhb.close_contact"][:4, 0], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(sample["dhb.action_valid_mask"][:4], [0.0, 1.0, 1.0, 0.0])


def test_policy_trace_contact_dataset_filters_unmasked_rows(tmp_path):
    manifest_path = _write_policy_trace_fixture(tmp_path)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "rollout_json": "",
                    "chunk_index": 1,
                    "t_start": 4,
                    "n_use": 4,
                    "npz_path": str(tmp_path / "missing.npz"),
                    "target_close": [0.0, 0.0, 0.0, 0.0],
                    "train_mask": [0.0, 0.0, 0.0, 0.0],
                }
            )
            + "\n"
        )

    dataset = PolicyTraceContactDataset(
        {
            "policy_trace_contact_manifest_jsonl": str(manifest_path),
            "action_window_size": 8,
            "policy_trace_require_train_mask": True,
        }
    )

    assert len(dataset) == 1


def test_policy_trace_contact_dataset_merges_base_statistics(tmp_path):
    manifest_path = _write_policy_trace_fixture(tmp_path)
    base_stats_path = tmp_path / "base_stats.json"
    base_stats_path.write_text(
        json.dumps(
            {
                "franka": {
                    "action": {
                        "min": [0.0],
                        "max": [1.0],
                        "mask": [True],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "dataset_statistics.json"
    dataset = PolicyTraceContactDataset(
        {
            "policy_trace_contact_manifest_jsonl": str(manifest_path),
            "action_window_size": 8,
            "policy_trace_base_statistics_json": str(base_stats_path),
        }
    )

    dataset.save_dataset_statistics(out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["franka"]["action"]["max"] == [1.0]
    assert payload["policy_trace_contact"]["num_rows"] == 1
    assert payload["policy_trace_contact"]["supervised_steps"] == 2
    assert payload["policy_trace_contact"]["positive_steps"] == 1
