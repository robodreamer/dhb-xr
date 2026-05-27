# Development

## Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/robodreamer/dhb-xr.git
cd dhb_xr

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Or using Pixi (recommended)
pixi install
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_encoder.py

# Run with coverage
pytest tests/ --cov=src/dhb_xr --cov-report=html

# Run tests for specific backends
pixi run test-casadi  # CasADi optimization tests
```

## Optional CusADi Decode Libraries

The repository includes fixed-horizon `.casadi` artifacts and generated CUDA
source for CusADi decode. Normal installs use compiled binaries from the dhb_xr
cache when they exist and fall back to CPU decode otherwise.

```bash
# Build all supported fixed-horizon libraries into the default cache
pixi run python -m dhb_xr.optimization.build_cusadi_decode --horizons 50 80 100 150 200

# Inspect paths without writing files
pixi run python -m dhb_xr.optimization.build_cusadi_decode --horizons 100 --dry-run
```

Set `DHB_XR_CUSADI_CACHE` to use a project-local cache during development or
deployment testing.

## Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/dhb_xr
```

## Building Documentation

```bash
# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Project Structure

```
dhb_xr/
├── src/dhb_xr/
│   ├── core/           # Core types and geometry
│   ├── encoder/        # DHB encoding functions
│   ├── decoder/        # DHB decoding functions
│   ├── optimization/   # Trajectory optimization
│   ├── database/       # Motion storage and retrieval
│   ├── tokenization/   # VQ-VAE tokenization
│   ├── losses/         # Imitation learning losses
│   ├── visualization/  # Plotting utilities
│   └── utils/          # Additional utilities
├── tests/              # Unit and integration tests
├── examples/           # Example scripts
├── docs/               # Documentation
└── notebooks/          # Jupyter notebooks
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Release Process

### Automated PyPI Publishing

1. Update version: `pixi run version --bump patch` or `pixi run version 0.4.0`
2. Update `CHANGELOG.md`
3. Commit and push changes
4. Create and push a matching git tag, for example `git tag v0.4.0 && git push origin v0.4.0`
5. The tag-triggered release workflow builds and publishes to PyPI

### Manual Publishing

```bash
pixi run setup-pypirc
pixi run build-dist
pixi run upload-testpypi
python -m pip install -i https://test.pypi.org/simple/ dhb_xr
pixi run upload-pypi
```

PyPI and TestPyPI API tokens are stored in `~/.pypirc` or environment variables.
