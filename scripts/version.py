#!/usr/bin/env python3
"""Version management script for dhb_xr package.

Usage:
    python scripts/version.py                    # Show current version
    python scripts/version.py 0.2.0              # Set version to 0.2.0
    python scripts/version.py --bump patch       # Bump patch version (0.2.0 -> 0.2.1)
    python scripts/version.py --bump minor       # Bump minor version (0.2.0 -> 0.3.0)
    python scripts/version.py --bump major       # Bump major version (0.2.0 -> 1.0.0)
"""

import argparse
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _replace_once(
    content: str,
    pattern: str,
    replacement: str,
    description: str,
) -> str:
    updated, replacements = re.subn(
        pattern,
        replacement,
        content,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise ValueError(f"Could not find exactly one {description}")
    return updated


def get_version(project_root: Path = PROJECT_ROOT) -> str:
    """Get current version from pyproject.toml."""
    content = (project_root / "pyproject.toml").read_text()

    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)

    raise ValueError("Could not find version in pyproject.toml")


def set_version(new_version: str, project_root: Path = PROJECT_ROOT) -> None:
    """Set every package and documentation version surface."""
    pyproject_path = project_root / "pyproject.toml"
    init_path = project_root / "src" / "dhb_xr" / "__init__.py"
    examples_init_path = project_root / "examples" / "__init__.py"
    mkdocs_path = project_root / "mkdocs.yml"

    # Replace only the project metadata version, not similarly named tool tasks.
    content = _replace_once(
        pyproject_path.read_text(),
        r'(\[project\]\s*\nname\s*=\s*["\']dhb_xr["\']\s*\n)version\s*=\s*["\'][^"\']+["\']',
        rf'\1version = "{new_version}"',
        "[project] version",
    )
    pyproject_path.write_text(content)

    init_content = _replace_once(
        init_path.read_text(),
        r'^__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        "dhb_xr.__version__",
    )
    init_path.write_text(init_content)

    examples_content = _replace_once(
        examples_init_path.read_text(),
        r'^__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        "examples.__version__",
    )
    examples_init_path.write_text(examples_content)

    mkdocs_content = _replace_once(
        mkdocs_path.read_text(),
        r"^(site_description:.*)\(v[^)]+\)(\s*)$",
        rf"\g<1>(v{new_version})\g<2>",
        "version in the MkDocs site description",
    )
    mkdocs_content = _replace_once(
        mkdocs_content,
        r'^  version:\s*["\'][^"\']+["\']\s*$',
        f'  version: "{new_version}"',
        "MkDocs extra.version",
    )
    mkdocs_path.write_text(mkdocs_content)

    print(f"✓ Version updated to {new_version}")


def bump_version(component: str, project_root: Path = PROJECT_ROOT) -> str:
    """Bump version component."""
    current = get_version(project_root)
    major, minor, patch = map(int, current.split("."))

    if component == "major":
        major += 1
        minor = 0
        patch = 0
    elif component == "minor":
        minor += 1
        patch = 0
    elif component == "patch":
        patch += 1
    else:
        raise ValueError(f"Invalid component: {component}")

    new_version = f"{major}.{minor}.{patch}"
    set_version(new_version, project_root)
    return new_version


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Version management for dhb_xr")
    parser.add_argument(
        "version",
        nargs="?",
        help="Version to set (e.g., 0.2.0)",
    )
    parser.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        help="Bump version component",
    )

    args = parser.parse_args()

    if args.bump:
        new_version = bump_version(args.bump)
        print(f"✓ Bumped {args.bump} version to {new_version}")
    elif args.version:
        set_version(args.version)
    else:
        # Just show current version
        print(get_version())


if __name__ == "__main__":
    main()
