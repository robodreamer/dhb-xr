#!/usr/bin/env bash
set -euo pipefail

echo "== Host =="
hostname
whoami
hostname -I || true

echo
echo "== Git =="
git status --short --branch

echo
echo "== SSH listener =="
if command -v ss >/dev/null 2>&1; then
    ss -tlnp | grep -E ':(22|2222)\b' || true
else
    echo "ss not available"
fi

echo
echo "== NVIDIA PCI devices =="
if command -v lspci >/dev/null 2>&1; then
    lspci | grep -Ei 'nvidia|vga|3d|display' || true
else
    echo "lspci not available"
fi

echo
echo "== NVIDIA kernel modules =="
lsmod | grep -E 'nvidia|nouveau' || true

echo
echo "== nvidia-smi =="
if command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi; then
        echo
        echo "nvidia-smi failed. Checking whether the NVIDIA module exists for this kernel..."
        modinfo nvidia 2>&1 || true
        if test -f /var/run/reboot-required; then
            echo
            cat /var/run/reboot-required
            cat /var/run/reboot-required.pkgs 2>/dev/null || true
        fi
    fi
else
    echo "nvidia-smi not available"
fi

echo
echo "== Python CUDA =="
if command -v pixi >/dev/null 2>&1; then
    pixi run python - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}")
else:
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"cuda_device_{idx}: {torch.cuda.get_device_name(idx)}")
PY
else
    python - <<'PY'
try:
    import torch
except Exception as exc:
    print(f"torch import failed: {exc}")
else:
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            print(f"cuda_device_{idx}: {torch.cuda.get_device_name(idx)}")
PY
fi
