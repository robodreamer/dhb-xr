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

### Merge-triggered PyPI publishing

Releases are prepared in a pull request and start automatically after that PR is
merged to `main` with a changed `pyproject.toml` version.

1. Update every version surface with `pixi run version --bump patch` or
   `pixi run version 0.6.0`.
2. Move the relevant notes from `[Unreleased]` into a dated heading such as
   `## [0.6.0] - 2026-08-10`, then run `cp CHANGELOG.md docs/changelog.md`.
3. Validate locally with `pixi run validate-release`, `pixi run pytest -q`, and
   `pixi run mkdocs build --strict`.
4. Open and merge the release PR. The **Prepare Release** workflow repeats those
   checks, creates the annotated `v<version>` tag, and dispatches the
   **Publish Package** workflow on that immutable tag.
5. The publish workflow validates that the tag and metadata agree, builds and
   checks the wheel and source distribution, publishes them to PyPI, and only
   then creates the GitHub release.

The explicit dispatch is intentional: a tag pushed with the repository's
`GITHUB_TOKEN` does not start a second workflow. Both workflows are idempotent;
an existing version tag is not recreated, and a failed publish can be retried
by running **Publish Package** on that tag.

### One-time repository configuration

- The repository may keep its default workflow permission read-only. The
  workflow grants only its tag/dispatch job explicit `contents: write` and
  `actions: write` access.
- Create a GitHub environment named `pypi`. Optional required reviewers can add
  a human approval gate immediately before publication.
- In the PyPI project settings, configure a trusted publisher with owner
  `robodreamer`, repository `dhb-xr`, workflow `publish.yml`, and environment
  `pypi`. No long-lived PyPI token is required by the automated workflow.

### Manual fallback

```bash
pixi run setup-pypirc
pixi run build-dist
pixi run upload-testpypi
python -m pip install -i https://test.pypi.org/simple/ dhb_xr
pixi run upload-pypi
```

PyPI and TestPyPI API tokens are stored in `~/.pypirc` or environment variables.
The fallback is independent of the trusted-publishing workflow and should only
be used when GitHub Actions is unavailable.
