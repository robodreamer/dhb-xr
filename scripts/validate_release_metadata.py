#!/usr/bin/env python3
"""Validate synchronized DHB-XR release metadata and the release tag."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _match_once(pattern: str, text: str, description: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {description}, found {len(matches)}")
    return str(matches[0])


def validate_release_metadata(
    project_root: Path = PROJECT_ROOT,
    *,
    expected_tag: str | None = None,
) -> str:
    """Return the release version after validating every published version surface."""

    pyproject = (project_root / "pyproject.toml").read_text()
    project_block = _match_once(
        r"(?s)^\[project\]\n(?P<block>.*?)(?=^\[|\Z)",
        pyproject,
        "[project] table",
    )
    package_version = _match_once(
        r'^version\s*=\s*["\']([^"\']+)["\']\s*$',
        project_block,
        "project version",
    )

    package_init = (project_root / "src" / "dhb_xr" / "__init__.py").read_text()
    runtime_version = _match_once(
        r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
        package_init,
        "dhb_xr.__version__",
    )

    examples_init = (project_root / "examples" / "__init__.py").read_text()
    examples_version = _match_once(
        r'^__version__\s*=\s*["\']([^"\']+)["\']\s*$',
        examples_init,
        "examples.__version__",
    )

    mkdocs = (project_root / "mkdocs.yml").read_text()
    docs_version = _match_once(
        r'^  version:\s*["\']([^"\']+)["\']\s*$',
        mkdocs,
        "MkDocs extra.version",
    )
    site_version = _match_once(
        r"^site_description:.*\(v([^\)]+)\)\s*$",
        mkdocs,
        "MkDocs site-description version",
    )

    versions = {
        "pyproject": package_version,
        "runtime": runtime_version,
        "examples": examples_version,
        "mkdocs": docs_version,
        "site_description": site_version,
    }
    if len(set(versions.values())) != 1:
        rendered = ", ".join(f"{name}={value}" for name, value in versions.items())
        raise ValueError(f"version mismatch: {rendered}")

    changelog = (project_root / "CHANGELOG.md").read_text()
    docs_changelog = (project_root / "docs" / "changelog.md").read_text()
    if docs_changelog != changelog:
        raise ValueError("docs/changelog.md must exactly match CHANGELOG.md")
    heading_pattern = re.compile(
        rf"^## \[{re.escape(package_version)}\] - " r"(?P<date>\d{4}-\d{2}-\d{2})$",
        re.MULTILINE,
    )
    headings = list(heading_pattern.finditer(changelog))
    if len(headings) != 1:
        raise ValueError(
            f"CHANGELOG.md must contain exactly one dated {package_version} release entry"
        )
    release_date = headings[0].group("date")
    try:
        date.fromisoformat(release_date)
    except ValueError as exc:
        raise ValueError(f"CHANGELOG.md contains invalid release date {release_date}") from exc

    expected = f"v{package_version}"
    if expected_tag is not None and expected_tag != expected:
        raise ValueError(f"tag mismatch: expected {expected}, received {expected_tag}")
    return package_version


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="Require an exact v<project-version> tag")
    args = parser.parse_args()
    try:
        print(validate_release_metadata(expected_tag=args.tag))
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
