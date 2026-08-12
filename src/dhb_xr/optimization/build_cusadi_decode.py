"""Build optional dhb_xr-owned CUDA decode libraries from bundled sources."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

from dhb_xr.optimization.cusadi_runtime import (
    cusadi_decode_library_filename,
    default_cusadi_decode_cache_dir,
)
from dhb_xr.optimization.cusadi_solver import (
    PACKAGE_CUSADI_CODEGEN_DIR,
    SUPPORTED_CUSADI_DECODE_SAMPLES,
)


def _validate_horizons(horizons: list[int]) -> list[int]:
    unsupported = sorted(set(horizons) - set(SUPPORTED_CUSADI_DECODE_SAMPLES))
    if unsupported:
        raise ValueError(
            "Unsupported CusADi decode horizon(s): "
            f"{unsupported}. Supported: {SUPPORTED_CUSADI_DECODE_SAMPLES}"
        )
    return horizons


def _cmake_text(horizons: list[int]) -> str:
    sources = []
    libraries = []
    for horizon in horizons:
        target = f"fn_dhb_decode_linear_{horizon}"
        sources.append(f"set({target.upper()}_SOURCE codegen/{target}.cu)")
        libraries.append(f"add_library({target} SHARED ${{{target.upper()}_SOURCE}})")

    return dedent(
        f"""
        cmake_minimum_required(VERSION 3.15)
        project(dhb_xr_cusadi_decode LANGUAGES CXX)

        include(CheckLanguage)
        check_language(CUDA)
        if(NOT CMAKE_CUDA_COMPILER)
            message(FATAL_ERROR
                "CUDA compiler not found. Install the CUDA toolkit or use CPU decode."
            )
        endif()
        enable_language(CUDA)

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 75 86)
        endif()

        set(CMAKE_CXX_STANDARD 11)
        set(CUDA_NVCC_FLAGS ${{CUDA_NVCC_FLAGS}}; -O3 --use_fast_math)
        include_directories(${{CMAKE_CURRENT_SOURCE_DIR}})

        {os.linesep.join(sources)}

        {os.linesep.join(libraries)}
        """
    ).strip()


def prepare_cusadi_decode_build(
    *,
    horizons: list[int],
    output_dir: str,
    source_dir: str = PACKAGE_CUSADI_CODEGEN_DIR,
) -> Path:
    """Create a self-contained CMake project for supported decode kernels."""
    horizons = _validate_horizons(horizons)
    output_path = Path(output_dir).expanduser().resolve()
    codegen_path = output_path / "codegen"
    codegen_path.mkdir(parents=True, exist_ok=True)

    for horizon in horizons:
        source = Path(source_dir) / f"fn_dhb_decode_linear_{horizon}.cu"
        if not source.is_file():
            raise FileNotFoundError(f"Bundled CusADi CUDA source not found: {source}")
        shutil.copy2(source, codegen_path / source.name)

    (output_path / "CMakeLists.txt").write_text(_cmake_text(horizons) + "\n")
    return output_path


def build_cusadi_decode_libraries(
    *,
    horizons: list[int],
    output_dir: str | None = None,
    dry_run: bool = False,
    cuda_architectures: str | None = None,
    cuda_host_compiler: str | None = None,
    fresh: bool = False,
) -> list[Path]:
    """Build fixed-horizon decode libraries into a dhb_xr-owned cache directory."""
    output_dir = output_dir or default_cusadi_decode_cache_dir()
    project_dir = prepare_cusadi_decode_build(horizons=horizons, output_dir=output_dir)
    build_dir = project_dir / "build"
    expected = [build_dir / cusadi_decode_library_filename(horizon) for horizon in horizons]

    if dry_run:
        return expected

    build_dir.mkdir(parents=True, exist_ok=True)
    configure_command = ["cmake"]
    if fresh:
        configure_command.append("--fresh")
    configure_command.extend(["-S", str(project_dir), "-B", str(build_dir)])
    if cuda_architectures:
        configure_command.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}")
    if cuda_host_compiler:
        configure_command.append(f"-DCMAKE_CUDA_HOST_COMPILER={cuda_host_compiler}")
    subprocess.run(configure_command, check=True)
    subprocess.run(["cmake", "--build", str(build_dir), "-j"], check=True)
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"CusADi decode build completed but outputs are missing: {missing}")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build optional dhb_xr CUDA decode libraries from bundled sources."
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(SUPPORTED_CUSADI_DECODE_SAMPLES),
        help=f"Fixed sample horizons to build. Supported: {SUPPORTED_CUSADI_DECODE_SAMPLES}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Build/cache directory. Defaults to $DHB_XR_CUSADI_CACHE or ~/.cache/dhb_xr/cusadi.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the build project and print expected outputs without compiling.",
    )
    parser.add_argument(
        "--cuda-architectures",
        default=None,
        help=(
            "CMake CUDA architecture list (for example '120' for Blackwell or "
            "'89;120'). Defaults to the builder's portable 75/86 targets."
        ),
    )
    parser.add_argument(
        "--cuda-host-compiler",
        default=None,
        help="Optional host C++ compiler path passed to CMake (for example /usr/bin/g++-13).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard the existing CMake cache before configuring.",
    )
    args = parser.parse_args()

    outputs = build_cusadi_decode_libraries(
        horizons=args.horizons,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        cuda_architectures=args.cuda_architectures,
        cuda_host_compiler=args.cuda_host_compiler,
        fresh=args.fresh,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
