"""
Cusadi-based GPU-parallel trajectory optimization (optional).

This module provides GPU-accelerated batch decoding of DHB invariants using CusADi.

Setup (one-time):
    1. Build optional GPU decode libraries:
       python -m dhb_xr.optimization.build_cusadi_decode --horizons 50 80 100 150 200
    2. The .casadi and generated .cu decode inputs are bundled with dhb_xr.
       Compilation is explicit and writes to $DHB_XR_CUSADI_CACHE or ~/.cache/dhb_xr/cusadi.

Benchmark results (RTX A2000, 50-step trajectories):
    Batch 100:   43x speedup (CPU 34ms vs GPU 0.8ms)
    Batch 1000: 199x speedup (CPU 342ms vs GPU 1.7ms)
    Batch 2000: 387x speedup (CPU 685ms vs GPU 1.8ms)

Without CUDA/CasADi/compiled decode libraries: CusadiTrajectoryOptimizer.forward()
falls back to NumPy batched decode.
"""

from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from dhb_xr.core.geometry_torch import (
    axis_angle_to_quat_torch,
    euler_xyz_to_rot_torch,
    quat_multiply_torch,
)
from dhb_xr.core.types import DHBMethod, EncodingMethod
from dhb_xr.decoder.dhb_dr import decode_dhb_dr
from dhb_xr.encoder.dhb_dr import encode_dhb_dr
from dhb_xr.optimization.cusadi_runtime import (
    DhbCusadiFunction,
    default_cusadi_decode_library_dir,
    resolve_cusadi_decode_library,
)
from dhb_xr.optimization.fatrop_solver import (
    _boundary_adjusted_position_reference,
    _boundary_adjusted_quaternion_reference,
    _decode_adapted_quaternions,
    _path_quality_metrics,
)

# Check for CasADi
try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

# dhb_xr owns the narrow runtime loader for generated decode kernels. The
# external CusADi package is only needed when regenerating generic kernels.
HAS_CUSADI = True
CusadiFunction = DhbCusadiFunction

# Check for PyTorch with CUDA
try:
    import torch

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*NVML.*")
        HAS_TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH_CUDA = False

# Default CusADi paths
PACKAGE_CUSADI_FUNCTION_DIR = os.path.join(
    os.path.dirname(__file__),
    "cusadi_functions",
)
PACKAGE_CUSADI_CODEGEN_DIR = os.path.join(
    os.path.dirname(__file__),
    "cusadi_codegen",
)
CUSADI_ROOT = os.environ.get("CUSADI_ROOT", "/home/andypark/Projects/repos/cusadi")
SUPPORTED_CUSADI_DECODE_SAMPLES = (50, 80, 100, 150, 200)
DEFAULT_DECODE_SAMPLES = 80
DEFAULT_DECODE_CONTROLS = DEFAULT_DECODE_SAMPLES - 1
DEFAULT_FN_PATH = os.path.join(
    PACKAGE_CUSADI_FUNCTION_DIR,
    f"fn_dhb_decode_linear_{DEFAULT_DECODE_SAMPLES}.casadi",
)
DEFAULT_CUSADI_FUNCTION_DIR = PACKAGE_CUSADI_FUNCTION_DIR
EXTERNAL_CUSADI_FUNCTION_DIR = os.path.join(CUSADI_ROOT, "src/casadi_functions")
DEFAULT_CUSADI_LIBRARY_DIR = default_cusadi_decode_library_dir()
EXTERNAL_CUSADI_LIBRARY_DIR = os.path.join(CUSADI_ROOT, "build")

# Global cached CusADi function
_cusadi_fn_cache: Dict[Tuple[str, str, int], Any] = {}
_boundary_adapt_fn_cache: Dict[int, Any] = {}


@dataclass(frozen=True)
class CusadiDecodeArtifact:
    """Resolved fixed-horizon CusADi decode artifact."""

    horizon: int
    path: str
    library_path: Optional[str] = None


@dataclass(frozen=True)
class CusadiDecodeSpec:
    """Policy for selecting a fixed-horizon CusADi decode artifact."""

    mode: Literal["auto", "cpu", "gpu_required"] = "auto"
    horizon: Union[int, Literal["auto"]] = "auto"
    fallback: Literal["cpu", "error"] = "cpu"
    artifact_dir: Optional[str] = None
    artifact_path: Optional[str] = None
    library_dir: Optional[str] = None
    library_path: Optional[str] = None

    def __post_init__(self):
        if self.mode not in {"auto", "cpu", "gpu_required"}:
            raise ValueError("cusadi decode mode must be auto, cpu, or gpu_required")
        if self.fallback not in {"cpu", "error"}:
            raise ValueError("cusadi decode fallback must be cpu or error")


def cusadi_decode_filename(samples: int) -> str:
    """Canonical filename for a fixed-horizon linear decode artifact."""
    return f"fn_dhb_decode_linear_{samples}.casadi"


def infer_cusadi_decode_horizon(path: str) -> Optional[int]:
    """Infer the sample horizon from a canonical CusADi decode artifact path."""
    match = re.search(r"fn_dhb_decode_linear_(\d+)\.casadi$", os.path.basename(path))
    return int(match.group(1)) if match else None


def resolve_cusadi_decode_artifact(
    spec: CusadiDecodeSpec,
    traj_length: int,
) -> Optional[CusadiDecodeArtifact]:
    """Resolve an exact-horizon decode artifact for the requested trajectory length."""
    if spec.mode == "cpu":
        return None
    if spec.horizon != "auto" and int(spec.horizon) < 2:
        raise ValueError("cusadi decode horizon must be at least 2 samples")

    requested_horizon = traj_length if spec.horizon == "auto" else int(spec.horizon)
    if spec.artifact_path is not None:
        path = os.path.abspath(spec.artifact_path)
        horizon = infer_cusadi_decode_horizon(path) or requested_horizon
    else:
        artifact_dir = spec.artifact_dir or DEFAULT_CUSADI_FUNCTION_DIR
        path = os.path.join(artifact_dir, cusadi_decode_filename(requested_horizon))
        horizon = requested_horizon

    if not os.path.exists(path):
        return None
    library_path = resolve_cusadi_decode_library(
        horizon,
        library_dir=spec.library_dir,
        library_path=spec.library_path,
        extra_search_dirs=(EXTERNAL_CUSADI_LIBRARY_DIR,),
    )
    return CusadiDecodeArtifact(horizon=horizon, path=path, library_path=library_path)


def build_boundary_adaptation_function(traj_length: int) -> Any:
    """Build the fixed-horizon boundary adaptation function used by CusADi."""
    if not HAS_CASADI:
        raise RuntimeError("CasADi is required to build CusADi adaptation functions")
    if traj_length in _boundary_adapt_fn_cache:
        return _boundary_adapt_fn_cache[traj_length]

    pos_ref = ca.SX.sym("pos_ref", traj_length * 3)
    start_delta = ca.SX.sym("start_delta", 3)
    goal_delta = ca.SX.sym("goal_delta", 3)
    out = []
    denom = max(traj_length - 1, 1)
    for i in range(traj_length):
        blend = float(i) / denom
        p_i = pos_ref[3 * i : 3 * i + 3]
        adjusted = p_i + (1.0 - blend) * start_delta + blend * goal_delta
        out.append(adjusted)
    trajectory = ca.horzcat(*out).T
    fn = ca.Function(
        f"fn_dhb_boundary_adapt_{traj_length}",
        [pos_ref, start_delta, goal_delta],
        [trajectory],
    )
    _boundary_adapt_fn_cache[traj_length] = fn
    return fn


def generate_trajectory_cusadi_adapt(
    pos_data: np.ndarray,
    quat_data: np.ndarray,
    pose_target_init: Dict[str, np.ndarray],
    pose_target_final: Dict[str, np.ndarray],
    traj_length: int,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
) -> Dict[str, Any]:
    """Generate a boundary-adapted trajectory through a CusADi-exportable function."""
    if not HAS_CASADI:
        raise RuntimeError("CasADi is required for the CusADi adaptation backend")

    pos_reference = _boundary_adjusted_position_reference(
        pos_data,
        {
            "position": np.asarray(pos_data[0], dtype=np.float64),
            "quaternion": np.asarray(quat_data[0], dtype=np.float64),
        },
        {
            "position": np.asarray(pos_data[-1], dtype=np.float64),
            "quaternion": np.asarray(quat_data[-1], dtype=np.float64),
        },
        traj_length,
    )
    start_delta = np.asarray(pose_target_init["position"], dtype=np.float64) - np.asarray(
        pos_reference[0], dtype=np.float64
    )
    goal_delta = np.asarray(pose_target_final["position"], dtype=np.float64) - np.asarray(
        pos_reference[-1], dtype=np.float64
    )
    fn = build_boundary_adaptation_function(traj_length)
    solve_start = time.perf_counter()
    positions = np.asarray(fn(pos_reference.reshape(-1), start_delta, goal_delta), dtype=np.float64)
    solve_time = time.perf_counter() - solve_start
    if positions.shape != (traj_length, 3):
        positions = positions.reshape((traj_length, 3), order="F")
    positions[0] = np.asarray(pose_target_init["position"], dtype=np.float64)
    positions[-1] = np.asarray(pose_target_final["position"], dtype=np.float64)

    quaternions = _boundary_adjusted_quaternion_reference(
        quat_data,
        pose_target_init,
        pose_target_final,
        traj_length,
    )
    encoded = encode_dhb_dr(
        positions,
        quaternions,
        init_pose={"position": positions[0], "quaternion": quaternions[0]},
        method=EncodingMethod.POSITION,
        use_default_initial_frames=False,
        dhb_method=dhb_method,
    )
    adapted_lin = encoded["linear_motion_invariants"][: max(len(positions) - 1, 0)]
    adapted_ang = encoded["angular_motion_invariants"][: max(len(positions) - 1, 0)]
    quaternions = _decode_adapted_quaternions(
        adapted_lin,
        adapted_ang,
        pose_target_init,
        pose_target_final,
        len(positions),
        angular_frame_initial=encoded["angular_frame_initial"],
        dhb_method=dhb_method,
    )

    return {
        "positions": positions,
        "quaternions": quaternions,
        "adapted_pos_data": positions,
        "adapted_quat_data": quaternions,
        "linear_invariants": adapted_lin,
        "angular_invariants": adapted_ang,
        "linear_motion_invariant": adapted_lin,
        "angular_motion_invariant": adapted_ang,
        "linear_frame_initial": encoded["linear_frame_initial"],
        "angular_frame_initial": encoded["angular_frame_initial"],
        "solve_time": solve_time,
        "success": True,
        "ocp_success": None,
        "path_quality_repaired": False,
        "path_quality_metrics": _path_quality_metrics(positions),
        "solver": "cusadi",
        "optimizer_solver": "cusadi-adapt",
        "cusadi_adaptation": "casadi-function",
    }


def get_cusadi_function(
    casadi_path: str = DEFAULT_FN_PATH,
    batch_size: int = 1000,
    library_path: Optional[str] = None,
) -> Optional[Any]:
    """Get or create a cached CusADi function for GPU decode."""
    if not HAS_CUSADI or not HAS_CASADI:
        return None

    cache_key = (casadi_path, library_path or "", batch_size)
    if cache_key in _cusadi_fn_cache:
        return _cusadi_fn_cache[cache_key]

    if not os.path.exists(casadi_path) or library_path is None:
        return None

    try:
        fn = ca.Function.load(casadi_path)
        cusadi_fn = CusadiFunction(fn, batch_size, library_path)
        _cusadi_fn_cache[cache_key] = cusadi_fn
        return cusadi_fn
    except Exception as e:
        print(f"Warning: Failed to load CusADi function: {e}")
        return None


def batched_decode_dhb_dr_gpu(
    linear_invariants_batch: np.ndarray,
    angular_invariants_batch: np.ndarray,
    initial_poses: List[Dict[str, np.ndarray]],
    casadi_path: str = DEFAULT_FN_PATH,
    library_path: Optional[str] = None,
    drop_padded: bool = True,
    initial_linear_frames: Optional[np.ndarray] = None,
    initial_angular_frames: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated batch decode using CusADi.

    This function uses CusADi to run the DHB decode on GPU in parallel across
    all trajectories in the batch.

    Requirements:
        - CasADi and PyTorch installed
        - CUDA available and PyTorch with CUDA support
        - Compiled dhb_xr decode library for the requested fixed horizon

    Args:
        linear_invariants_batch: (B, T, 4) linear invariants
        angular_invariants_batch: (B, T, 4) angular invariants
        initial_poses: List of B dicts with 'position' (3,) and 'quaternion' (4,) wxyz
        casadi_path: Path to compiled CasADi function
        drop_padded: Whether to drop padded frames, matching CPU decode
        initial_linear_frames: Optional (B, 3, 3) rotation matrices for the
            translational DHB frame. Defaults to identity, matching the NumPy
            decoder. This is deliberately independent of tool orientation.
        initial_angular_frames: Optional (B, 3, 3) natural angular frames for
            quaternion reconstruction. Defaults to identity.

    Returns:
        Tuple of (positions (B, N, 3), quaternions (B, N, 4))

    Raises:
        RuntimeError: If CusADi or CUDA is not available
    """
    if not HAS_TORCH_CUDA:
        raise RuntimeError("CUDA not available. Install PyTorch with CUDA support.")
    if not HAS_CUSADI:
        raise RuntimeError("CusADi not installed. Clone and install from github.com/se-hwan/cusadi")

    batch_size = linear_invariants_batch.shape[0]

    # Get or create CusADi function
    cusadi_fn = get_cusadi_function(casadi_path, batch_size, library_path=library_path)
    if cusadi_fn is None:
        raise RuntimeError(
            f"Could not load CusADi function from {casadi_path} with library {library_path}"
        )

    # Prepare inputs for CusADi
    # The fn_dhb_decode_linear function expects:
    # - i0: flattened linear invariants (T*4,) per sample
    # - i1: initial position (3,) per sample
    # - i2: initial rotation matrix (9,) flattened per sample

    # Stack initial poses
    init_pos = np.array([p["position"] for p in initial_poses])  # (B, 3)

    if initial_linear_frames is None:
        init_rot = np.tile(np.eye(3, dtype=np.float64), (batch_size, 1, 1))
    else:
        init_rot = np.asarray(initial_linear_frames, dtype=np.float64)
        if init_rot.shape != (batch_size, 3, 3):
            raise ValueError(
                f"initial_linear_frames must have shape ({batch_size}, 3, 3), got {init_rot.shape}"
            )

    # Check function signature - fn_dhb_decode_linear expects:
    # i0: linear invariants flattened (nnz_in for first input)
    # i1: initial position (3,)
    # i2: initial rotation matrix flattened (9,)
    n_in = cusadi_fn.fn_casadi.n_in()
    nnz_in = [cusadi_fn.fn_casadi.nnz_in(i) for i in range(n_in)]
    required_controls = nnz_in[0] // 4
    provided_controls = linear_invariants_batch.reshape(batch_size, -1).shape[1] // 4
    if provided_controls < required_controls:
        raise RuntimeError(
            "Compiled CusADi decode requires "
            f"{required_controls} controls, got {provided_controls}. "
            "Regenerate the decode function for this horizon or use CPU decode."
        )

    # Flatten linear invariants to match expected input size
    lin_flat = linear_invariants_batch.reshape(batch_size, -1)[:, : nnz_in[0]]

    # Convert to torch tensors on GPU (must be contiguous and double)
    lin_t = torch.from_numpy(lin_flat.astype(np.float64)).cuda().contiguous()
    pos_t = torch.from_numpy(init_pos.astype(np.float64)).cuda().contiguous()
    # CasADi vectorizes matrices in column-major order while contiguous PyTorch
    # tensors are row-major. Transpose the final two axes so the flat device
    # buffer represents the requested rotation under CasADi's reshape.
    init_rot_casadi = np.swapaxes(init_rot, -1, -2).copy()
    rot_t = torch.from_numpy(init_rot_casadi).cuda().contiguous()

    # Run CusADi function - evaluate takes a LIST of input tensors
    cusadi_fn.evaluate([lin_t, pos_t, rot_t])

    # Get dense output (positions)
    positions = cusadi_fn.getDenseOutput(0).cpu().numpy()  # (B, N, 3)

    # Reshape if needed (output might be (B, N*3) or (B, N, 3))
    if positions.ndim == 2:
        num_samples = positions.shape[1] // 3
        positions = positions.reshape(batch_size, num_samples, 3)
    if drop_padded:
        positions = positions[:, 2:]

    # The compiled linear decode returns the initial sample plus one output per
    # control, so a zero terminal invariant lets the angular decoder emit the
    # same horizon.
    quat_angular = angular_invariants_batch[:, :required_controls]
    if positions.shape[1] == required_controls + 1 - (2 if drop_padded else 0):
        terminal_angular = np.zeros((batch_size, 1, quat_angular.shape[2]), dtype=np.float64)
        quat_angular = np.concatenate([quat_angular, terminal_angular], axis=1)
    quat_batch = batched_decode_dhb_dr_quaternions_gpu(
        quat_angular,
        initial_poses,
        drop_padded=drop_padded,
        initial_angular_frames=initial_angular_frames,
    )

    return positions, quat_batch


def batched_decode_dhb_dr_quaternions_gpu(
    angular_invariants_batch: np.ndarray,
    initial_poses: List[Dict[str, np.ndarray]],
    *,
    drop_padded: bool = True,
    initial_angular_frames: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Decode DHB-DR quaternion trajectories in parallel across a CUDA batch.

    Time integration remains sequential, but every step is vectorized across
    the candidate batch. This removes the previous Python/NumPy loop over
    trajectories that dominated the otherwise-fast linear CusADi kernel.
    """
    if not HAS_TORCH_CUDA or torch is None:
        raise RuntimeError("CUDA PyTorch is required for GPU quaternion decode")
    if (
        axis_angle_to_quat_torch is None
        or euler_xyz_to_rot_torch is None
        or quat_multiply_torch is None
    ):
        raise RuntimeError("PyTorch geometry helpers are unavailable")

    angular = torch.as_tensor(
        angular_invariants_batch,
        dtype=torch.float64,
        device="cuda",
    )
    if angular.ndim != 3 or angular.shape[-1] != 4:
        raise ValueError("DHB-DR angular invariants must have shape (B, T, 4)")
    batch_size, num_samples, _ = angular.shape
    if len(initial_poses) != batch_size:
        raise ValueError("initial_poses length must match the invariant batch")
    if num_samples == 0:
        return np.empty((batch_size, 0, 4), dtype=np.float64)

    initial_quaternions = torch.as_tensor(
        np.asarray([pose["quaternion"] for pose in initial_poses]),
        dtype=torch.float64,
        device="cuda",
    )
    quaternion = initial_quaternions / torch.linalg.vector_norm(
        initial_quaternions, dim=-1, keepdim=True
    ).clamp_min(1e-12)
    trajectory = torch.empty(
        (batch_size, num_samples, 4),
        dtype=torch.float64,
        device="cuda",
    )
    trajectory[:, 0] = quaternion
    if initial_angular_frames is None:
        angular_frame = (
            torch.eye(3, dtype=torch.float64, device="cuda")
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
    else:
        initial_angular_frames = np.asarray(initial_angular_frames, dtype=np.float64)
        if initial_angular_frames.shape != (batch_size, 3, 3):
            raise ValueError(
                "initial_angular_frames must have shape "
                f"({batch_size}, 3, 3), got {initial_angular_frames.shape}"
            )
        angular_frame = torch.as_tensor(
            initial_angular_frames,
            dtype=torch.float64,
            device="cuda",
        )
    frame_updates = euler_xyz_to_rot_torch(angular[:, :, 1:4])

    for sample_index in range(num_samples):
        rotation_vector = angular_frame[:, :, 0] * angular[:, sample_index, 0:1]
        delta = axis_angle_to_quat_torch(-rotation_vector)
        quaternion = quat_multiply_torch(quaternion, delta)
        quaternion = quaternion / torch.linalg.vector_norm(
            quaternion, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        if sample_index < num_samples - 1:
            previous = trajectory[:, sample_index]
            sign = torch.where(
                torch.sum(quaternion * previous, dim=-1, keepdim=True) < 0.0,
                -torch.ones_like(quaternion[:, :1]),
                torch.ones_like(quaternion[:, :1]),
            )
            quaternion = quaternion * sign
            trajectory[:, sample_index + 1] = quaternion
        angular_frame = angular_frame @ frame_updates[:, sample_index]

    if drop_padded:
        trajectory = trajectory[:, 2:]
    return trajectory.cpu().numpy()


def batched_decode_dhb_dr(
    linear_invariants_batch: np.ndarray,
    angular_invariants_batch: np.ndarray,
    initial_poses: List[Dict[str, np.ndarray]],
    method: EncodingMethod = EncodingMethod.POSITION,
    dhb_method: DHBMethod = DHBMethod.DOUBLE_REFLECTION,
    drop_padded: bool = True,
    use_gpu: bool = False,
    casadi_path: str = DEFAULT_FN_PATH,
    initial_linear_frames: Optional[np.ndarray] = None,
    initial_angular_frames: Optional[np.ndarray] = None,
) -> tuple:
    """
    Decode multiple trajectories in batch.

    Args:
        linear_invariants_batch: (B, T, 4) or list of (T, 4)
        angular_invariants_batch: (B, T, 4) or list of (T, 4)
        initial_poses: list of B dicts with 'position' (3,) and 'quaternion' (4,) wxyz.
        method: 'pos' or 'vel' for invariant interpretation
        dhb_method: DHBMethod enum (DOUBLE_REFLECTION or ORIGINAL)
        drop_padded: Whether to drop padded frames
        use_gpu: If True and CUDA available, use CusADi GPU acceleration
        casadi_path: Path to compiled CasADi function for GPU decode
        initial_linear_frames: Optional translational DHB frame rotations for
            GPU and CPU decode. Defaults to identity.
        initial_angular_frames: Optional natural angular frame rotations for
            GPU and CPU quaternion reconstruction. Defaults to identity.

    Returns:
        (positions_batch, quaternions_batch): (B, N, 3), (B, N, 4).
    """
    # Try GPU decode if requested
    if use_gpu and dhb_method == DHBMethod.DOUBLE_REFLECTION:
        if HAS_TORCH_CUDA and HAS_CUSADI:
            try:
                return batched_decode_dhb_dr_gpu(
                    linear_invariants_batch,
                    angular_invariants_batch,
                    initial_poses,
                    casadi_path,
                    library_path=resolve_cusadi_decode_library(
                        infer_cusadi_decode_horizon(casadi_path) or DEFAULT_DECODE_SAMPLES,
                        extra_search_dirs=(EXTERNAL_CUSADI_LIBRARY_DIR,),
                    ),
                    drop_padded=drop_padded,
                    initial_linear_frames=initial_linear_frames,
                    initial_angular_frames=initial_angular_frames,
                )
            except Exception as e:
                print(f"GPU decode failed, falling back to CPU: {e}")
        else:
            missing = []
            if not HAS_TORCH_CUDA:
                missing.append("PyTorch CUDA")
            if not HAS_CUSADI:
                missing.append("CusADi")
            print(f"GPU decode unavailable (missing: {', '.join(missing)}), using CPU")
    elif use_gpu:
        print("GPU decode supports DHB double-reflection invariants only, using CPU")

    # CPU decode (NumPy loop)
    if isinstance(linear_invariants_batch, np.ndarray) and linear_invariants_batch.ndim == 3:
        batch_size = linear_invariants_batch.shape[0]
        lin_list = [linear_invariants_batch[b] for b in range(batch_size)]
        ang_list = [angular_invariants_batch[b] for b in range(batch_size)]
    else:
        lin_list = list(linear_invariants_batch)
        ang_list = list(angular_invariants_batch)
        batch_size = len(lin_list)
    assert len(initial_poses) == batch_size and len(ang_list) == batch_size
    linear_frames = None
    if initial_linear_frames is not None:
        linear_frames = np.asarray(initial_linear_frames, dtype=np.float64)
        if linear_frames.shape != (batch_size, 3, 3):
            raise ValueError(
                "initial_linear_frames must have shape "
                f"({batch_size}, 3, 3), got {linear_frames.shape}"
            )
    angular_frames = None
    if initial_angular_frames is not None:
        angular_frames = np.asarray(initial_angular_frames, dtype=np.float64)
        if angular_frames.shape != (batch_size, 3, 3):
            raise ValueError(
                "initial_angular_frames must have shape "
                f"({batch_size}, 3, 3), got {angular_frames.shape}"
            )
    pos_list = []
    quat_list = []
    for b in range(batch_size):
        decoded = decode_dhb_dr(
            lin_list[b],
            ang_list[b],
            initial_poses[b],
            method=method,
            dhb_method=dhb_method,
            drop_padded=drop_padded,
            initial_linear_frame=(linear_frames[b] if linear_frames is not None else None),
            initial_angular_frame=(angular_frames[b] if angular_frames is not None else None),
        )
        positions = decoded["positions"]
        pos_list.append(positions)
        quat_list.append(decoded["quaternions"])
    return np.stack(pos_list), np.stack(quat_list)


class CusadiTrajectoryOptimizer:
    """
    Batched trajectory decode / optimization.

    - If cusadi is not installed: forward() uses batched_decode_dhb_dr (NumPy loop).
    - If cusadi is installed and a compiled decode function is provided:
      forward() can use CusadiFunction for GPU batch (set decode_casadi_path).
    """

    def __init__(
        self,
        batch_size: int = 1000,
        dhb_method: DHBMethod | str = DHBMethod.DOUBLE_REFLECTION,
        decode_casadi_path: Optional[str] = None,
        decode_spec: Optional[CusadiDecodeSpec] = None,
    ):
        self.batch_size = batch_size
        self.dhb_method = (
            dhb_method
            if isinstance(dhb_method, DHBMethod)
            else (
                DHBMethod.DOUBLE_REFLECTION
                if dhb_method == "double_reflection"
                else DHBMethod.ORIGINAL
            )
        )
        self.decode_casadi_path = decode_casadi_path
        self.decode_spec = decode_spec or CusadiDecodeSpec(artifact_path=decode_casadi_path)
        self._cusadi_fn: Any = None
        self.last_decode_backend = "unavailable"
        self.last_decode_metadata: Dict[str, Any] = {
            "cusadi_decode": "unavailable",
            "cusadi_gpu": False,
            "cusadi_quaternion_decode": "unavailable",
            "cusadi_decode_requested_horizon": None,
            "cusadi_decode_artifact_horizon": None,
            "cusadi_decode_artifact_path": None,
            "cusadi_decode_library_path": None,
            "cusadi_decode_fallback_reason": None,
        }

    def forward(
        self,
        linear_invariants: np.ndarray,
        angular_invariants: np.ndarray,
        initial_poses: List[Dict[str, np.ndarray]],
        method: EncodingMethod = EncodingMethod.POSITION,
        drop_padded: bool = True,
        initial_linear_frames: Optional[np.ndarray] = None,
        initial_angular_frames: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Batched decode: (linear_inv, angular_inv, initial_poses) -> (positions, quaternions).

        linear_invariants: (B, T, 4), angular_invariants: (B, T, 4),
        initial_poses: list of B dicts. Returns (B, N, 3), (B, N, 4).
        Uses NumPy batched_decode_dhb_dr. For GPU, build a .casadi decode and use
        CusadiFunction.evaluate() directly (see export_casadi_decode and cusadi repo).
        """
        sample_horizon = linear_invariants.shape[1] + 1
        requested_horizon = (
            sample_horizon if self.decode_spec.horizon == "auto" else int(self.decode_spec.horizon)
        )
        self.last_decode_metadata = {
            "cusadi_decode": "unavailable",
            "cusadi_gpu": False,
            "cusadi_quaternion_decode": "unavailable",
            "cusadi_decode_requested_horizon": requested_horizon,
            "cusadi_decode_artifact_horizon": None,
            "cusadi_decode_artifact_path": None,
            "cusadi_decode_library_path": None,
            "cusadi_decode_fallback_reason": None,
        }

        fallback_reason = None
        artifact = None
        if self.decode_spec.mode == "cpu":
            fallback_reason = "cpu_requested"
        elif self.dhb_method != DHBMethod.DOUBLE_REFLECTION:
            fallback_reason = "unsupported_dhb_method"
        elif not HAS_TORCH_CUDA:
            fallback_reason = "cuda_unavailable"
        elif not HAS_CUSADI:
            fallback_reason = "cusadi_unavailable"
        else:
            try:
                artifact = resolve_cusadi_decode_artifact(
                    self.decode_spec,
                    sample_horizon,
                )
            except Exception as exc:
                fallback_reason = "artifact_resolution_failed"
                if self.decode_spec.mode == "gpu_required" or self.decode_spec.fallback == "error":
                    raise RuntimeError("Failed to resolve CusADi decode artifact") from exc
            if artifact is None and fallback_reason is None:
                fallback_reason = "artifact_not_found"

        if artifact is not None:
            if artifact.library_path is None:
                fallback_reason = "library_not_found"
                self.last_decode_metadata.update(
                    {
                        "cusadi_decode_artifact_horizon": artifact.horizon,
                        "cusadi_decode_artifact_path": artifact.path,
                        "cusadi_decode_library_path": None,
                    }
                )
                if self.decode_spec.mode == "gpu_required" or self.decode_spec.fallback == "error":
                    raise RuntimeError(
                        "CusADi GPU decode unavailable: library_not_found. Run "
                        "python -m dhb_xr.optimization.build_cusadi_decode "
                        f"--horizons {artifact.horizon}"
                    )
                self.last_decode_backend = "cpu"
            else:
                try:
                    self.last_decode_metadata.update(
                        {
                            "cusadi_decode_artifact_horizon": artifact.horizon,
                            "cusadi_decode_artifact_path": artifact.path,
                            "cusadi_decode_library_path": artifact.library_path,
                        }
                    )
                    decoded = batched_decode_dhb_dr_gpu(
                        linear_invariants,
                        angular_invariants,
                        initial_poses,
                        casadi_path=artifact.path,
                        library_path=artifact.library_path,
                        drop_padded=drop_padded,
                        initial_linear_frames=initial_linear_frames,
                        initial_angular_frames=initial_angular_frames,
                    )
                    self.last_decode_backend = "gpu"
                    self.last_decode_metadata.update(
                        {
                            "cusadi_decode": "gpu",
                            "cusadi_gpu": True,
                            "cusadi_quaternion_decode": "gpu",
                            "cusadi_decode_fallback_reason": None,
                        }
                    )
                    return decoded
                except Exception as exc:
                    fallback_reason = "artifact_load_failed"
                    if (
                        self.decode_spec.mode == "gpu_required"
                        or self.decode_spec.fallback == "error"
                    ):
                        raise RuntimeError("CusADi GPU decode failed") from exc
                    self.last_decode_backend = "cpu"

        if self.decode_spec.mode == "gpu_required" or self.decode_spec.fallback == "error":
            raise RuntimeError(f"CusADi GPU decode unavailable: {fallback_reason or 'unknown'}")

        self.last_decode_backend = "cpu"
        self.last_decode_metadata.update(
            {
                "cusadi_decode": "cpu",
                "cusadi_gpu": False,
                "cusadi_quaternion_decode": "cpu",
                "cusadi_decode_fallback_reason": fallback_reason,
            }
        )
        return batched_decode_dhb_dr(
            linear_invariants,
            angular_invariants,
            initial_poses,
            method=method,
            dhb_method=self.dhb_method,
            drop_padded=drop_padded,
            initial_linear_frames=initial_linear_frames,
            initial_angular_frames=initial_angular_frames,
        )
