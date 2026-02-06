#!/bin/bash
# Wrapper script for uploading to TestPyPI with credentials from ~/.pypirc

set -e

if [ -f "$HOME/.pypirc" ]; then
    echo "Uploading distribution to TestPyPI..."
    twine upload --repository testpypi --config-file "$HOME/.pypirc" --verbose dist/* 2>&1 || {
        echo ""
        echo "Note: If you see 'File already exists', the version was already uploaded."
        echo "Bump the version with: pixi run version --bump patch"
        exit 1
    }
else
    echo "Warning: ~/.pypirc not found. Using environment variables if set."
    echo "Set TWINE_USERNAME=__token__ and TWINE_PASSWORD=<token> if needed."
    twine upload --repository testpypi --verbose dist/*
fi