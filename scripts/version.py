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
import sys
from pathlib import Path


def get_version():
    """Get current version from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    with open(pyproject_path, "r") as f:
        content = f.read()

    match = re.search(r'^version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)

    raise ValueError("Could not find version in pyproject.toml")


def set_version(new_version: str):
    """Set version in pyproject.toml and the package __version__."""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    init_path = Path(__file__).parent.parent / "src" / "dhb_xr" / "__init__.py"

    with open(pyproject_path, "r") as f:
        content = f.read()

    # Replace only the project metadata version, not similarly named tool tasks.
    content = re.sub(
        r'(\[project\]\s*\nname\s*=\s*["\']dhb_xr["\']\s*\n)version\s*=\s*["\'][^"\']+["\']',
        rf'\1version = "{new_version}"',
        content,
        count=1,
        flags=re.MULTILINE,
    )

    with open(pyproject_path, "w") as f:
        f.write(content)

    with open(init_path, "r") as f:
        init_content = f.read()
    init_content = re.sub(
        r'^__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"',
        init_content,
        count=1,
        flags=re.MULTILINE,
    )
    with open(init_path, "w") as f:
        f.write(init_content)

    print(f"✓ Version updated to {new_version}")


def bump_version(component: str):
    """Bump version component."""
    current = get_version()
    major, minor, patch = map(int, current.split('.'))

    if component == 'major':
        major += 1
        minor = 0
        patch = 0
    elif component == 'minor':
        minor += 1
        patch = 0
    elif component == 'patch':
        patch += 1
    else:
        raise ValueError(f"Invalid component: {component}")

    new_version = f"{major}.{minor}.{patch}"
    set_version(new_version)
    return new_version


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Version management for dhb_xr")
    parser.add_argument(
        'version',
        nargs='?',
        help='Version to set (e.g., 0.2.0)'
    )
    parser.add_argument(
        '--bump',
        choices=['major', 'minor', 'patch'],
        help='Bump version component'
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
