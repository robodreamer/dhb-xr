#!/usr/bin/env bash
# Re-clone all vendor dependencies for DHB-XR.
# Run from repo root: bash vendor/setup.sh

set -e
VENDOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

clone_or_update() {
    local name="$1"
    local url="$2"
    local pin="$3"
    local dir="$VENDOR_DIR/$name"

    if [ -d "$dir/.git" ]; then
        echo "[$name] already exists — fetching updates..."
        git -C "$dir" fetch --quiet
    else
        echo "[$name] cloning from $url..."
        git clone "$url" "$dir"
    fi

    if [ -n "$pin" ]; then
        echo "[$name] pinning to $pin..."
        git -C "$dir" checkout "$pin" --quiet 2>/dev/null || echo "  (pin not found — using HEAD)"
    fi
    echo "[$name] ready at $(git -C $dir rev-parse --short HEAD)"
}

clone_or_update "starVLA"               "https://github.com/starVLA/starVLA"               "f8394a8"
clone_or_update "vla-evaluation-harness" "https://github.com/allenai/vla-evaluation-harness" "ae23527"

echo ""
echo "All vendor deps ready in $VENDOR_DIR"
echo "See vendor/README.md for DHB integration notes."
