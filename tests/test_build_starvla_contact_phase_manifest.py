import json
import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_starvla_contact_phase_manifest.py"
_SPEC = importlib.util.spec_from_file_location("build_starvla_contact_phase_manifest", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_manifest = _MODULE.build_manifest
infer_contact_phase = _MODULE.infer_contact_phase


def test_infer_contact_phase_from_contact_labels():
    row = {
        "labels": ["background", "positive", "hard_negative", "release"],
        "target_close": [0.0, 1.0, 0.0, 0.0],
        "train_mask": [1.0, 1.0, 1.0, 1.0],
    }

    assert infer_contact_phase(row) == [0, 1, 0, 2]


def test_infer_contact_phase_prefers_gripper_state_for_closed_transport():
    row = {
        "labels": ["attached_transport_positive", "attached_transport_release"],
        "target_gripper_state": [[1.0], [0.0]],
        "target_contact_release_phase": [0, 2],
    }

    assert infer_contact_phase(row) == [1, 2]


def test_infer_contact_phase_accepts_scalar_fields():
    row = {"target_contact_release_phase": 2, "train_mask": 1.0}

    assert infer_contact_phase(row) == [2]


def test_build_manifest_repeats_release_rows(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "npz_path": "a.npz",
                        "labels": ["background", "positive"],
                        "target_close": [0.0, 1.0],
                        "train_mask": [1.0, 1.0],
                    }
                ),
                json.dumps(
                    {
                        "npz_path": "b.npz",
                        "labels": ["attached_transport_release"],
                        "target_contact_release_phase": [2],
                        "train_mask": [1.0],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"

    summary = build_manifest([source], output, release_repeat=3, closed_repeat=1)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["rows"] == 4
    assert len(rows) == 4
    assert rows[0]["target_contact_phase"] == [0, 1]
    assert rows[1]["target_contact_phase"] == [2]
    assert summary["phase_counts"] == {"0": 1, "1": 1, "2": 3}
