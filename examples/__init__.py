"""
DHB-XR Examples Package

This package contains comprehensive examples demonstrating DHB-XR usage across
various robotics and VLA applications, including both Python scripts and Jupyter notebooks.

Quick Start Examples:
- basic_encoding: Core DHB-DR/DHB-QR encoding and decoding
- trajectory_adaptation: Retargeting trajectories to new poses
- dhb_qr_comparison: Comparing DHB-DR vs DHB-QR representations
- gpu_batch_optimization: Large-scale trajectory optimization
- vla_tokenization: DHB-Token for discrete action sequences
- motion_database: Similarity search and trajectory retrieval
- imitation_learning: Invariant-based imitation learning losses
- dhb_ti_time_invariant: Time-invariant reparameterization

VLA Integration Examples:
- libero_full_demo: Complete LIBERO simulation with DHB adaptation
- libero_pro_dhb_demo: LIBERO-PRO perturbation robustness testing
- libero_swap_demo: Spatial swap demonstration with naive vs DHB comparison
- robocasa_libero_dhb_pipeline: RoboCASA and LIBERO data processing pipeline

Tutorial Notebooks:
- tutorial_dhb_basics.ipynb: Basic encoding/decoding concepts
- tutorial_invariance_demo.ipynb: SE(3) invariance demonstration
- tutorial_umi_xdof_motion_analysis.ipynb: Real-world UMI dataset analysis
- demo_dhb_features_and_use_cases.ipynb: Comprehensive feature overview
- tutorial_vla_integration.ipynb: VLA integration with LIBERO/RoboCASA

Usage:
    from examples import *
    # or import specific examples:
    from examples.basic_encoding import run_example

    # Run notebooks with:
    # jupyter notebook
    # pixi run notebook
"""

__version__ = "0.4.0"

_EXAMPLE_EXPORTS = {
    "run_basic_encoding": (".basic_encoding", "run_example"),
    "run_trajectory_adaptation": (".trajectory_adaptation", "run_example"),
    "run_dhb_comparison": (".dhb_qr_comparison", "run_example"),
}


def __getattr__(name):
    """Lazily import example entrypoints so subpackages can be imported alone."""
    if name not in _EXAMPLE_EXPORTS:
        raise AttributeError(name)
    module_name, attr_name = _EXAMPLE_EXPORTS[name]
    from importlib import import_module

    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

def __dir__():
    """List all available examples."""
    return sorted([*globals(), *_EXAMPLE_EXPORTS])
