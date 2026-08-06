#!/usr/bin/env bash
set -euo pipefail

DHB_XR_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DHB_XR_PROJECT_DIR"

pixi install -e cuda

if [[ -n "${DHB_XR_CUDA_ARCHITECTURES:-}" ]]; then
  CUDA_ARCHITECTURES="$DHB_XR_CUDA_ARCHITECTURES"
else
  CUDA_ARCHITECTURES="$(pixi run -e cuda python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in the DHB-XR cuda environment")
major, minor = torch.cuda.get_device_capability()
print(f"{major}{minor}")
PY
)"
fi

echo "Building DHB-XR CusADi kernels for CUDA architecture ${CUDA_ARCHITECTURES}"
PYTHONPATH=src pixi run -e cuda python \
  -m dhb_xr.optimization.build_cusadi_decode \
  --horizons 50 80 100 150 200 \
  --cuda-architectures "$CUDA_ARCHITECTURES" \
  --fresh

PYTHONPATH=src pixi run -e cuda python - <<'PY'
import casadi as ca
import numpy as np
import rockit  # noqa: F401
import torch

from dhb_xr.optimization.cusadi_solver import (
    CusadiDecodeSpec,
    CusadiTrajectoryOptimizer,
)

if not ca.has_nlpsol("fatrop"):
    raise SystemExit("CasADi FATROP plugin is unavailable")

horizon = 50
optimizer = CusadiTrajectoryOptimizer(
    batch_size=1,
    decode_spec=CusadiDecodeSpec(
        mode="gpu_required",
        horizon=horizon,
        fallback="error",
    ),
)
linear = np.zeros((1, horizon - 1, 4), dtype=np.float64)
angular = np.zeros_like(linear)
poses = [
    {
        "position": np.zeros(3, dtype=np.float64),
        "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    }
]
positions, quaternions = optimizer.forward(
    linear,
    angular,
    poses,
    drop_padded=False,
)
torch.cuda.synchronize()
if optimizer.last_decode_backend != "gpu":
    raise SystemExit(f"Strict CusADi decode used {optimizer.last_decode_backend!r}")
if optimizer.last_decode_metadata["cusadi_quaternion_decode"] != "gpu":
    raise SystemExit("Strict CusADi decode left quaternion reconstruction on CPU")
if positions.shape != (1, horizon, 3) or quaternions.shape != (1, horizon, 4):
    raise SystemExit("Strict CusADi decode returned unexpected output shapes")

print(
    "Validated accelerated backends: "
    f"torch={torch.__version__}, cuda={torch.version.cuda}, "
    f"gpu={torch.cuda.get_device_name()}, cusadi=gpu, fatrop=available"
)
PY
