# Installation

## Basic Installation

Install DHB-XR using pip:

```bash
pip install dhb_xr
```

## Optional Dependencies

DHB-XR supports several optional features that require additional dependencies:

### GPU Support (PyTorch)

For GPU-accelerated encoding/decoding and tokenization:

```bash
pip install dhb_xr[gpu]
```

### Optimization (CasADi)

For trajectory optimization and adaptation:

```bash
pip install dhb_xr[optimization]
```

### Motion Database

For trajectory storage and retrieval:

```bash
pip install dhb_xr[database]
```

### All Features

Install all optional dependencies:

```bash
pip install dhb_xr[all]
```

## Development Installation

For development and testing:

```bash
git clone https://github.com/robodreamer/dhb_xr.git
cd dhb_xr
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.10
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.5

Optional:

- PyTorch >= 2.0 (for GPU support)
- CasADi >= 3.5 (for optimization)
- FAISS >= 1.7 (for motion database)
- DTAIDistance >= 3.0 (for trajectory similarity)