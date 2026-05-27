"""Local runtime support for dhb_xr's pre-generated CusADi decode kernels."""

from __future__ import annotations

import ctypes
import os
from typing import Any, Optional

try:
    import torch
except ImportError:  # pragma: no cover - exercised by optional dependency installs
    torch = None


def default_cusadi_decode_cache_dir() -> str:
    """Directory where optional locally compiled decode libraries are stored."""
    return os.path.abspath(
        os.environ.get(
            "DHB_XR_CUSADI_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "dhb_xr", "cusadi"),
        )
    )


def default_cusadi_decode_library_dir() -> str:
    """Directory where CMake places optional locally compiled decode libraries."""
    return os.path.join(default_cusadi_decode_cache_dir(), "build")


def cusadi_decode_library_filename(samples: int) -> str:
    """Canonical shared-library filename for a fixed-horizon decode kernel."""
    return f"libfn_dhb_decode_linear_{samples}.so"


def infer_cusadi_decode_library_horizon(path: str) -> Optional[int]:
    """Infer the sample horizon from a canonical decode shared-library path."""
    basename = os.path.basename(path)
    prefix = "libfn_dhb_decode_linear_"
    suffix = ".so"
    if not basename.startswith(prefix) or not basename.endswith(suffix):
        return None
    try:
        return int(basename[len(prefix) : -len(suffix)])
    except ValueError:
        return None


def resolve_cusadi_decode_library(
    samples: int,
    *,
    library_dir: Optional[str] = None,
    library_path: Optional[str] = None,
    extra_search_dirs: tuple[str, ...] = (),
) -> Optional[str]:
    """Resolve a compiled decode shared library without requiring CusADi's repo layout."""
    if library_path is not None:
        path = os.path.abspath(library_path)
        return path if os.path.exists(path) else None

    filename = cusadi_decode_library_filename(samples)
    search_dirs = []
    if library_dir is not None:
        search_dirs.append(library_dir)
    search_dirs.append(default_cusadi_decode_library_dir())
    search_dirs.extend(extra_search_dirs)

    for directory in search_dirs:
        path = os.path.abspath(os.path.join(directory, filename))
        if os.path.exists(path):
            return path
    return None


class DhbCusadiFunction:
    """Small ctypes/PyTorch wrapper for the generated decode kernel ABI.

    The generated CUDA libraries expose ``evaluate(inputs, work, outputs, batch)``.
    This wrapper is intentionally narrow and only owns runtime invocation; code
    generation and compilation live behind an explicit build command.
    """

    def __init__(self, fn_casadi: Any, num_instances: int, library_path: str):
        if torch is None:
            raise RuntimeError("PyTorch is required to load a dhb_xr CusADi decode library")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to load a dhb_xr CusADi decode library")
        if not os.path.exists(library_path):
            raise RuntimeError(f"CusADi decode library not found: {library_path}")

        self.fn_casadi = fn_casadi
        self.fn_name = fn_casadi.name()
        self.num_instances = num_instances
        self.library_path = os.path.abspath(library_path)
        self._device = "cuda"
        self._fn_library = ctypes.CDLL(self.library_path)
        self._fn_library.evaluate.restype = ctypes.c_float
        self.eval_time = 0.0
        self._setup()

    def evaluate(self, inputs):
        self._clear_tensors()
        self._prepare_input_tensors(inputs)
        self.eval_time = self._fn_library.evaluate(
            self._fn_input,
            self._fn_work,
            self._fn_output,
            self.num_instances,
        )

    def getDenseOutput(self, out_idx=None):  # noqa: N802 - matches CusADi API
        env_idx = torch.tensor(
            range(self.num_instances),
            device=self._device,
        ).repeat_interleave(self.fn_casadi.nnz_out(out_idx))
        row_idx = torch.tensor(
            self.fn_casadi.sparsity_out(out_idx).get_triplet()[0],
            device=self._device,
        ).repeat(self.num_instances)
        col_idx = torch.tensor(
            self.fn_casadi.sparsity_out(out_idx).get_triplet()[1],
            device=self._device,
        ).repeat(self.num_instances)
        dense_shape = (
            self.num_instances,
            self.fn_casadi.size1_out(out_idx),
            self.fn_casadi.size2_out(out_idx),
        )
        return torch.sparse_coo_tensor(
            torch.vstack((env_idx, row_idx, col_idx)),
            self.outputs_sparse[out_idx].reshape(-1),
            dense_shape,
        ).to_dense()

    def _setup(self):
        self._input_tensors = [
            torch.zeros(
                (self.num_instances, self.fn_casadi.nnz_in(i)),
                device=self._device,
                dtype=torch.double,
            ).contiguous()
            for i in range(self.fn_casadi.n_in())
        ]
        self._output_tensors = [
            torch.zeros(
                self.num_instances,
                self.fn_casadi.nnz_out(i),
                device=self._device,
                dtype=torch.double,
            ).contiguous()
            for i in range(self.fn_casadi.n_out())
        ]
        self._output_tensors_dense = [
            torch.zeros(
                (
                    self.num_instances,
                    self.fn_casadi.size1_out(i),
                    self.fn_casadi.size2_out(i),
                ),
                device=self._device,
                dtype=torch.double,
            ).contiguous()
            for i in range(self.fn_casadi.n_out())
        ]
        self._work_tensor = torch.zeros(
            (self.num_instances, self.fn_casadi.sz_w()),
            device=self._device,
            dtype=torch.double,
        ).contiguous()
        self._input_ptrs = torch.zeros(
            self.fn_casadi.n_in(), device=self._device, dtype=torch.int64
        ).contiguous()
        self._output_ptrs = torch.zeros(
            self.fn_casadi.n_out(), device=self._device, dtype=torch.int64
        ).contiguous()

        for i in range(self.fn_casadi.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(self.fn_casadi.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self._fn_input = self._cast_pointer(self._input_ptrs.data_ptr(), "int")
        self._fn_output = self._cast_pointer(self._output_ptrs.data_ptr(), "int")
        self._fn_work = self._cast_pointer(self._work_tensor.data_ptr(), "double")
        self.inputs_sparse = self._input_tensors
        self.outputs_sparse = self._output_tensors
        self.outputs_dense = self._output_tensors_dense

    def _prepare_input_tensors(self, inputs):
        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        self._fn_input = self._cast_pointer(self._input_ptrs.data_ptr(), "int")
        self.inputs_sparse = self._input_tensors

    def _clear_tensors(self):
        for tensor in self._output_tensors:
            tensor.zero_()
        self._work_tensor.zero_()

    @staticmethod
    def _cast_pointer(ptr, kind: str):
        if kind == "int":
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))
        if kind == "double":
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
        return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
