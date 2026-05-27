# Installation

DHB-XR installs as a normal Python package. Start with the smallest install that
matches your workflow, then add optional backends when you need trajectory
optimization, GPU decode, tokenization, or database retrieval.

## Fastest Path

```bash
python -m pip install dhb_xr
python -c "import dhb_xr; print(dhb_xr.__version__)"
```

This is enough for DHB-DR/DHB-QR encoding and decoding with NumPy.

## Choose an Install Path

| Goal | Install | Notes |
| --- | --- | --- |
| Encode/decode trajectories | `python -m pip install dhb_xr` | Core NumPy/SciPy path. |
| Boundary-adapt trajectories | `python -m pip install "dhb_xr[fatrop]"` | Enables the FATROP/Rockit path used by `generate_trajectory(..., backend="auto")` when available. |
| Legacy CasADi/IPOPT optimization | `python -m pip install "dhb_xr[optimization]"` | Useful for explicit `backend="ipopt"` or exploratory fallback. |
| Fixed-horizon GPU decode | `python -m pip install "dhb_xr[cusadi]"` | Adds CasADi and PyTorch. CUDA libraries are built locally only when requested. |
| VLA tokenization | `python -m pip install "dhb_xr[tokenization]"` | Adds PyTorch and Einops for DHB-Token models. |
| Motion database | `python -m pip install "dhb_xr[database]"` | Adds FAISS CPU retrieval support. |
| Examples and notebooks | `python -m pip install "dhb_xr[examples]"` | Adds Jupyter and notebook tooling. |
| Everything | `python -m pip install "dhb_xr[all]"` | Convenience install for broad experimentation. |

Smoke-test the optimization API after installing optional solver dependencies:

```bash
python -m pip install "dhb_xr[fatrop]"
python -c "from dhb_xr.optimization import generate_trajectory; print(generate_trajectory)"
```

## Optional CusADi GPU Decode

DHB-XR ships fixed-horizon CasADi artifacts and generated CUDA source for sample
horizons `50`, `80`, `100`, `150`, and `200`. You do not need an external CusADi
checkout for those supported decode horizons.

Requirements:

- NVIDIA GPU and driver for GPU execution.
- CUDA toolkit with `nvcc` for local library compilation.
- CUDA-enabled PyTorch if you want GPU execution rather than CPU fallback.

Build the decode libraries into the dhb_xr cache:

```bash
python -m pip install "dhb_xr[cusadi]"
dhb_xr-build-cusadi-decode --horizons 50 80 100 150 200
```

The default output is `$DHB_XR_CUSADI_CACHE/build` when `DHB_XR_CUSADI_CACHE` is
set, otherwise `~/.cache/dhb_xr/cusadi/build`.

For a no-write smoke test:

```bash
dhb_xr-build-cusadi-decode --horizons 50 80 100 150 200 --dry-run
```

At runtime, `backend="cusadi"` uses exact-horizon GPU decode when a matching
compiled library is available. CPU decode remains the default fallback. Use
`cusadi_decode="gpu_required"` or `cusadi_decode_fallback="error"` when missing
CUDA assets should fail loudly.

## Development Setup

Pixi is the recommended development environment because it pins docs, build,
test, and optional CUDA tooling in one place.

```bash
# Install pixi: https://pixi.sh
curl -fsSL https://pixi.sh/install.sh | bash

git clone https://github.com/robodreamer/dhb-xr.git
cd dhb-xr
pixi install
pixi run test
```

Useful development commands:

```bash
pixi run build              # editable install
pixi run docs               # build MkDocs site
pixi run test-casadi        # optimization and CusADi tests
pixi run build-wheel        # build a wheel
pixi run notebook           # launch notebooks
```

CUDA development uses a separate Linux environment so non-Linux solves do not
pull CUDA-only packages:

```bash
pixi install -e cuda
pixi run -e cuda check-cuda
pixi run -e cuda python -m dhb_xr.optimization.build_cusadi_decode --dry-run
```

## Troubleshooting

If `backend="auto"` does not select FATROP, verify that the optional solver stack
imports:

```bash
python -c "from dhb_xr.optimization import generate_trajectory_fatrop; print(generate_trajectory_fatrop)"
```

If CusADi decode reports a missing library, build the matching fixed horizon:

```bash
python -m dhb_xr.optimization.build_cusadi_decode --horizons 100
```

If CUDA is not available to PyTorch:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

On systems without CUDA, leave CusADi decode in its default mode and DHB-XR will
use CPU decode fallback.
