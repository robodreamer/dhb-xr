"""CLI utilities for DHB-XR.

Currently provides:
- `dhb_xr-examples`: locate or copy example scripts for pip-installed users.
"""

import argparse
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Optional


def _find_examples_dir() -> Optional[Path]:
    """Find the examples directory in the installed package."""
    try:
        import dhb_xr

        # Get the package directory
        dhb_xr_pkg_dir = Path(dhb_xr.__file__).parent

        # Check multiple possible locations (in order of likelihood)
        candidates = [
            # Relative to package parent (for editable installs from source)
            dhb_xr_pkg_dir.parent.parent / "examples",
            # At site-packages root (if installed via package data)
            dhb_xr_pkg_dir.parent / "examples",
            # In the dhb_xr package itself
            dhb_xr_pkg_dir / "examples",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # Verify it has example files
                if any(candidate.glob("*.py")):
                    return candidate
    except Exception:
        pass

    # Fallback: check all sys.path entries
    for site_packages in sys.path:
        try:
            site_path = Path(site_packages)
            if not site_path.exists():
                continue

            # Check various locations
            candidates = [
                site_path / "examples",
                site_path.parent / "examples",
            ]

            for candidate in candidates:
                if candidate.exists() and candidate.is_dir():
                    if any(candidate.glob("*.py")):
                        return candidate
        except Exception:
            continue

    return None


def examples_cmd(argv: Optional[list[str]] = None) -> int:
    """Entry point for `dhb_xr-examples`."""
    parser = argparse.ArgumentParser(
        prog="dhb_xr-examples",
        description=(
            "Locate or copy DHB-XR example scripts.\n\n"
            "Examples are included in the pip package and can be run directly.\n"
            "Use --copy to copy them to a local directory for editing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--copy",
        nargs="?",
        const="dhb_xr_examples",
        metavar="DEST",
        type=str,
        help=(
            "Copy examples to DEST. "
            "If DEST is omitted, copies to ./dhb_xr_examples. "
            "Example: `dhb_xr-examples --copy` or `dhb_xr-examples --copy ./examples`."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available example scripts.",
    )

    args = parser.parse_args(argv)

    examples_dir = _find_examples_dir()

    if examples_dir is None:
        print("ERROR: Could not find examples directory in installed package.", file=sys.stderr)
        print("Examples may not be included in this installation.", file=sys.stderr)
        print("Try: pip install --force-reinstall dhb_xr[examples]", file=sys.stderr)
        return 1

    if args.list:
        print(f"Examples directory: {examples_dir}")
        print("\nAvailable examples:")
        for py_file in sorted(examples_dir.glob("*.py")):
            if py_file.name != "__init__.py":
                print(f"  - {py_file.name}")

        integration_dir = examples_dir / "integration"
        if integration_dir.exists() and integration_dir.is_dir():
            integration_files = [f for f in integration_dir.glob("*.py") if f.name != "__init__.py"]
            if integration_files:
                print("  📂 integration/:")
                for py_file in sorted(integration_files):
                    print(f"      - {py_file.name}")

        return 0

    if args.copy:
        dest = Path(args.copy).expanduser().resolve()
    else:
        dest = Path.cwd() / "dhb_xr_examples"

    try:
        # Ensure examples_dir is resolved to absolute path and verify it exists
        try:
            examples_dir_resolved = examples_dir.resolve()
        except (OSError, RuntimeError):
            examples_dir_resolved = examples_dir

        if not examples_dir_resolved.exists():
            if examples_dir.exists():
                examples_dir_resolved = examples_dir
            else:
                print(f"ERROR: Examples directory does not exist: {examples_dir_resolved}", file=sys.stderr)
                return 1

        if not examples_dir_resolved.is_dir():
            print(f"ERROR: Examples path exists but is not a directory: {examples_dir_resolved}", file=sys.stderr)
            return 1

        if dest.exists():
            if not dest.is_dir():
                print(f"ERROR: {dest} exists but is not a directory", file=sys.stderr)
                return 1
            response = input(f"Directory {dest} already exists. Overwrite? [y/N]: ")
            if response.lower() != "y":
                print("Cancelled.")
                return 0
            shutil.rmtree(dest)

        # Copy examples
        shutil.copytree(str(examples_dir_resolved), str(dest))

        # Remove any compiled Python files
        for pyc_file in dest.rglob("__pycache__"):
            shutil.rmtree(pyc_file)
        for pyc_file in dest.rglob("*.pyc"):
            pyc_file.unlink()

        print(f"✓ Copied examples to {dest}")
        print(f"\nTo run an example:")
        print(f"  cd {dest}")
        print("  pip install dhb_xr  # Install core library"
        print("  python basic_encoding.py  # Run an example"
        # If the user is in a Pixi project, also show the pixi-friendly form.
        if os.environ.get("PIXI_PROJECT_ROOT") or os.environ.get("PIXI_ENVIRONMENT_NAME"):
            print("  # or")
            print("  pixi run python basic_encoding.py")
        return 0

    except Exception as e:
        print(f"ERROR: Failed to copy examples: {e}", file=sys.stderr)
        return 1

    # Default: just show location
    print(f"Examples directory: {examples_dir}")
    print("\nTo run an example:")
    print(f"  {shlex.quote(sys.executable)} {examples_dir}/basic_encoding.py")
    print("\nTo copy examples to a local directory:")
    print("  dhb_xr-examples --copy")
    return 0


def main() -> None:
    """Main CLI entry point."""
    # Check if we're being called as examples command
    if len(sys.argv) > 1 and sys.argv[1] == "examples":
        sys.argv = sys.argv[1:]  # Remove 'examples' from argv
        sys.argv[0] = "dhb_xr-examples"  # Update program name
        raise SystemExit(examples_cmd())
    else:
        # Default behavior - show help
        print("DHB-XR CLI utilities")
        print()
        print("Available commands:")
        print("  dhb_xr-examples  Locate or copy example scripts")
        print()
        print("For help on a specific command, run:")
        print("  dhb_xr-examples --help")


if __name__ == "__main__":
    main()