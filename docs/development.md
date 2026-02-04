# Development

## Setting up Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/dhb_xr.git
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

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. Push to main branch
5. GitHub Actions will build and publish to PyPI