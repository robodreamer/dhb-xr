"""Release metadata and version-management regression tests."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"dhb_xr_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validate_release_metadata = _load_script("validate_release_metadata").validate_release_metadata
set_version = _load_script("version").set_version
bump_version = _load_script("version").bump_version


def _write_release_metadata(
    root: Path,
    *,
    package_version: str = "0.5.0",
    runtime_version: str = "0.5.0",
    docs_version: str = "0.5.0",
    release_date: str | None = "2026-08-09",
) -> None:
    (root / "src" / "dhb_xr").mkdir(parents=True)
    (root / "examples").mkdir()
    (root / "docs").mkdir()
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "dhb_xr"\nversion = "{package_version}"\n'
    )
    (root / "src" / "dhb_xr" / "__init__.py").write_text(f'__version__ = "{runtime_version}"\n')
    (root / "examples" / "__init__.py").write_text(f'__version__ = "{runtime_version}"\n')
    (root / "mkdocs.yml").write_text(
        f'site_description: SE(3) tools (v{docs_version})\nextra:\n  version: "{docs_version}"\n'
    )
    heading = f"## [{package_version}] - {release_date}\n" if release_date else ""
    changelog = f"# Changelog\n\n## [Unreleased]\n\n{heading}"
    (root / "CHANGELOG.md").write_text(changelog)
    (root / "docs" / "changelog.md").write_text(changelog)


def test_validate_release_metadata_accepts_synchronized_release(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path)

    assert validate_release_metadata(tmp_path, expected_tag="v0.5.0") == "0.5.0"


@pytest.mark.parametrize(
    ("runtime_version", "docs_version"),
    [("0.4.1", "0.5.0"), ("0.5.0", "0.4.1")],
)
def test_validate_release_metadata_rejects_version_mismatch(
    tmp_path: Path,
    runtime_version: str,
    docs_version: str,
) -> None:
    _write_release_metadata(
        tmp_path,
        runtime_version=runtime_version,
        docs_version=docs_version,
    )

    with pytest.raises(ValueError, match="version mismatch"):
        validate_release_metadata(tmp_path)


def test_validate_release_metadata_rejects_stale_examples_version(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path)
    (tmp_path / "examples" / "__init__.py").write_text('__version__ = "0.4.0"\n')

    with pytest.raises(ValueError, match="version mismatch"):
        validate_release_metadata(tmp_path)


def test_validate_release_metadata_requires_dated_changelog_entry(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path, release_date=None)

    with pytest.raises(ValueError, match="exactly one dated 0.5.0"):
        validate_release_metadata(tmp_path)


def test_validate_release_metadata_rejects_invalid_date(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path, release_date="2026-02-30")

    with pytest.raises(ValueError, match="invalid release date"):
        validate_release_metadata(tmp_path)


def test_validate_release_metadata_rejects_stale_docs_changelog(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path)
    (tmp_path / "docs" / "changelog.md").write_text("stale\n")

    with pytest.raises(ValueError, match="must exactly match"):
        validate_release_metadata(tmp_path)


def test_validate_release_metadata_rejects_wrong_tag(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path)

    with pytest.raises(ValueError, match="tag mismatch"):
        validate_release_metadata(tmp_path, expected_tag="v0.5.1")


def test_set_version_updates_every_version_surface(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path)

    set_version("0.6.0", project_root=tmp_path)

    assert 'version = "0.6.0"' in (tmp_path / "pyproject.toml").read_text()
    assert '__version__ = "0.6.0"' in (tmp_path / "src" / "dhb_xr" / "__init__.py").read_text()
    assert '__version__ = "0.6.0"' in (tmp_path / "examples" / "__init__.py").read_text()
    mkdocs = (tmp_path / "mkdocs.yml").read_text()
    assert "(v0.6.0)" in mkdocs
    assert 'version: "0.6.0"' in mkdocs


def test_bump_version_uses_requested_project_root(tmp_path: Path) -> None:
    _write_release_metadata(tmp_path)

    assert bump_version("minor", project_root=tmp_path) == "0.6.0"
    assert 'version = "0.6.0"' in (tmp_path / "pyproject.toml").read_text()
