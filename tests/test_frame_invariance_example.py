"""Regression test for the minimal frame-invariance overview example."""

import importlib.util
from pathlib import Path

EXAMPLE_PATH = Path(__file__).resolve().parents[1] / "examples" / "frame_invariance.py"
SPEC = importlib.util.spec_from_file_location("dhb_xr_frame_invariance", EXAMPLE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {EXAMPLE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
run_example = MODULE.run_example


def test_overview_moves_world_coordinates_but_preserves_signature() -> None:
    metrics = run_example()

    assert metrics["world_position_rms_delta_m"] > 0.1
    assert metrics["initial_frame_position_max_delta_m"] < 1e-12
    assert metrics["dhb_signature_max_delta"] < 1e-10
