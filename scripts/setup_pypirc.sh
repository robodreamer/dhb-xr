#!/bin/bash
# Setup ~/.pypirc for PyPI uploads

set -e

PYPI_RC="$HOME/.pypirc"

if [ -f "$PYPI_RC" ]; then
    echo "~/.pypirc already exists. Backing up to ~/.pypirc.backup"
    cp "$PYPI_RC" "${PYPI_RC}.backup"
fi

cat > "$PYPI_RC" << 'EOF'
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password =

[testpypi]
username = __token__
password =
EOF

echo "✓ Created ~/.pypirc"
echo ""
echo "Next steps:"
echo "1. Get your API tokens:"
echo "   - TestPyPI: https://test.pypi.org/manage/account/#api-tokens"
echo "   - PyPI: https://pypi.org/manage/account/#api-tokens"
echo ""
echo "2. Edit ~/.pypirc and fill in the password fields:"
echo "   vim ~/.pypirc"
echo ""
echo "3. Make the file readable only by you:"
echo "   chmod 600 ~/.pypirc"