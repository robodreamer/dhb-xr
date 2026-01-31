# C++ extension (optional)

Minimal nanobind-based C++ extension for dhb_xr. Placeholder for future performance-critical encode/decode.

## Build (from repo root)

Requires: CMake 3.15+, Python 3.10+ dev, nanobind (conda-forge or pip).

```bash
# With pixi (add nanobind to dev feature)
pixi install
pixi run build-cpp

# Or manually with conda (nanobind from pip does not provide CMake config; use conda)
conda install -c conda-forge cmake ninja python nanobind
cd dhb_xr
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
cp src/dhb_xr/_cpp/_dhb_xr_cpp*.so ../src/dhb_xr/
```

## Pixi task

From pyproject.toml: `build-cpp` runs CMake and builds the extension into `src/dhb_xr/`.

## Use from Python

```python
try:
    from dhb_xr._dhb_xr_cpp import cpp_version, add
    print(cpp_version())  # 1
    print(add(1.0, 2.0))  # 3.0
except ImportError:
    print("C++ extension not built")
```

## Extending

To add DHB encode/decode in C++: implement the recurrence in `dhb_cpp.cpp` (or new files), expose with `m.def(...)`, and call from Python in `encoder`/`decoder` when the extension is available.
