# dhb_xr CusADi Decode Assets

This directory contains fixed-horizon CasADi decode functions for the optional
CusADi GPU decode path. These files are packaged with `dhb_xr` so callers do not
need a separate CusADi checkout just to resolve decode artifacts.

Supported sample horizons:

- 50
- 80
- 100
- 150
- 200

GPU execution also needs a compiled CUDA shared library for the same horizon.
Compilation is explicit and local-machine specific:

```bash
python -m dhb_xr.optimization.build_cusadi_decode --horizons 50 80 100 150 200
```

By default this writes a small CMake build project and libraries under
`$DHB_XR_CUSADI_CACHE`, or `~/.cache/dhb_xr/cusadi` when the environment
variable is unset. Runtime lookup checks that cache first and falls back to an
external CusADi build directory only when present.

Do not make CUDA compilation mandatory at install time. The CPU decode path must
remain available when CUDA, CMake, a compiler, or prebuilt libraries are absent.
