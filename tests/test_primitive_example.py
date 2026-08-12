"""Smoke test for the documented primitive-program example."""

import importlib.util
import json
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


def test_primitive_benchmark_writes_report_and_figure(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    report = tmp_path / "primitive-benchmark.json"
    figure = tmp_path / "primitive-benchmark.svg"
    subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "benchmark_primitive_portfolio.py"),
            "--trials",
            "2",
            "--output",
            str(report),
            "--figure",
            str(figure),
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text())
    assert payload["schema_version"] == 1
    assert len(payload["rows"]) == 4
    assert set(payload["illustration"]["paths"]) == {
        "analytic:minimum_jerk",
        "analytic:via_point",
    }
    assert figure.read_text().startswith("<?xml")


def test_relational_claim_gate_enforces_easy_control_regression():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "benchmark_relational_retargeting",
        root / "examples" / "benchmark_relational_retargeting.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = []
    for trial in range(20):
        for condition, dhb_accepted, raw_accepted in (
            ("hard_relational_shift", True, False),
            ("easy_control", False, True),
        ):
            rows.extend(
                [
                    {
                        "condition": condition,
                        "trial": trial,
                        "source": "dhb-regularized:fatrop_relational",
                        "accepted": dhb_accepted,
                    },
                    {
                        "condition": condition,
                        "trial": trial,
                        "source": "reference:relational_anchor_se3",
                        "accepted": raw_accepted,
                    },
                ]
            )

    result = module._claim_gate(rows, "fatrop")

    assert result["acceptance_delta"] == 1.0
    assert result["easy_control_acceptance_delta"] == -1.0
    assert not result["passed"]


def test_retarget_isolation_report_serializes_numpy_gate_metrics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    report = tmp_path / "retarget-isolation.json"
    subprocess.run(
        [
            sys.executable,
            str(root / "examples" / "benchmark_retarget_isolation.py"),
            "--trials",
            "1",
            "--samples",
            "12",
            "--dhb-backend",
            "interpolation",
            "--output",
            str(report),
        ],
        cwd=root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(report.read_text())
    assert payload["schema_version"] == 1
    assert len(payload["rows"]) == 6
