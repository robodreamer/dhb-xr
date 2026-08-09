"""Dependency boundaries for the typed primitive runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_guarded_executor_import_does_not_eagerly_import_scipy():
    code = (
        "import sys; "
        "import dhb_xr.primitives as p; "
        "assert p.PrimitiveProgramExecutor.__module__ == "
        "'dhb_xr.primitives.execution'; "
        "assert 'scipy' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [
                    str(Path(__file__).resolve().parents[1] / "src"),
                    os.environ.get("PYTHONPATH", ""),
                ]
            ),
        },
        check=False,
    )
    assert result.returncode == 0, result.stderr
