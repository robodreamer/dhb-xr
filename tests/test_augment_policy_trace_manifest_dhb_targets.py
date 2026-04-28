from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "augment_policy_trace_manifest_dhb_targets.py"
SPEC = importlib.util.spec_from_file_location("augment_policy_trace_manifest_dhb_targets", SCRIPT_PATH)
augment = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(augment)


def test_augments_manifest_with_chunk_local_dhb_targets(tmp_path: Path) -> None:
    npz_path = tmp_path / "trace_chunk.npz"
    np.savez_compressed(
        npz_path,
        state_window=np.array(
            [
                [0.0, 0.0, 0.90, 0.0, 0.0, 0.0, 0.0, 0.04],
                [0.1, 0.2, 0.95, 0.0, 0.0, 0.0, 0.0, 0.00],
            ],
            dtype=np.float32,
        ),
    )
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps(
            {
                "npz_path": str(npz_path),
                "target_action_normalized": True,
                "target_action": [
                    [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0],
                    [-0.5, 0.2, 0.4, 0.0, 0.0, 0.0, 1.0],
                ],
                "train_mask": [1.0, 1.0],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "action": {
                    "min": [-0.01, -0.01, -0.01, -1.0, -1.0, -1.0, -1.0],
                    "max": [0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 1.0],
                    "mask": [True, True, True, True, True, True, True],
                }
            }
        ),
        encoding="utf-8",
    )
    out_path = tmp_path / "manifest_dhb.jsonl"

    summary = augment.augment_manifest(
        manifest_path,
        out_path,
        action_stats_json=stats_path,
    )

    assert summary["rows"] == 1
    assert summary["invariant_steps"] == 2
    row = json.loads(out_path.read_text(encoding="utf-8").strip())
    linear = np.asarray(row["target_linear_invariants"], dtype=np.float32)
    angular = np.asarray(row["target_angular_invariants"], dtype=np.float32)
    assert row["target_dhb_invariants_chunk_local"] is True
    assert linear.shape == (2, 4)
    assert angular.shape == (2, 4)
    assert np.isfinite(linear).all()
    assert np.isfinite(angular).all()
    assert float(np.abs(linear).sum()) > 0.0
