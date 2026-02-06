#!/bin/bash
# Wrapper script for uploading to PyPI with credentials from ~/.pypirc

set -e

REPO="${1:-pypi}"

# Use config file if it exists, otherwise use environment variables
if [ -f "$HOME/.pypirc" ]; then
    if [ "$REPO" = "testpypi" ]; then
        echo "Uploading distribution to TestPyPI..."
        twine upload --repository testpypi --config-file "$HOME/.pypirc" dist/*
    else
        echo "Uploading distribution to PyPI..."
        twine upload --config-file "$HOME/.pypirc" dist/*
    fi
else
    echo "Warning: ~/.pypirc not found. Using environment variables if set."
    echo "Set TWINE_USERNAME=__token__ and TWINE_PASSWORD=<token> if needed."
    if [ "$REPO" = "testpypi" ]; then
        twine upload --repository testpypi dist/*
    else
        twine upload dist/*
    fi
fi