"""Tests for the optional C++ extension (nanobind). Skip if not built."""

import pytest

try:
    from dhb_xr import _dhb_xr_cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


@pytest.mark.skipif(not _CPP_AVAILABLE, reason="C++ extension not built (run pixi run build-cpp)")
def test_cpp_version():
    from dhb_xr import cpp_version
    assert cpp_version is not None
    assert cpp_version() == 1


@pytest.mark.skipif(not _CPP_AVAILABLE, reason="C++ extension not built (run pixi run build-cpp)")
def test_cpp_add():
    from dhb_xr import _dhb_xr_cpp
    assert _dhb_xr_cpp.add(1.0, 2.0) == 3.0
    assert _dhb_xr_cpp.add(-1.0, 1.0) == 0.0


@pytest.mark.skipif(not _CPP_AVAILABLE, reason="C++ extension not built (run pixi run build-cpp)")
def test_cpp_module_has_functions():
    from dhb_xr import _dhb_xr_cpp
    assert hasattr(_dhb_xr_cpp, "cpp_version")
    assert hasattr(_dhb_xr_cpp, "add")
    assert callable(_dhb_xr_cpp.cpp_version)
    assert callable(_dhb_xr_cpp.add)


def test_cpp_version_import_without_extension():
    """When extension is not built, cpp_version is None but import succeeds."""
    from dhb_xr import cpp_version
    # If extension was built, cpp_version is callable; else None
    if cpp_version is not None:
        assert callable(cpp_version)
    else:
        assert cpp_version is None
