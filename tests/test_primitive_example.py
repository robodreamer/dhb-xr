"""Smoke test for the documented primitive-program example."""

import os
import subprocess
import sys
from pathlib import Path


def test_primitive_program_example_reaches_declared_terminal_state():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, str(root / "examples" / "primitive_program.py")],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "architecture demonstration" in result.stdout
    assert "status=succeeded" in result.stdout
    for node in ("approach", "grasp", "lift", "release", "retreat"):
        assert f"node={node:>8}" in result.stdout
