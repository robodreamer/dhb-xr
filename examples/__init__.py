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
    from dhb_xr_examples import *
    # or import specific examples:
    from dhb_xr_examples.basic_encoding import run_example

    # Run notebooks with:
    # jupyter notebook
    # pixi run notebook
"""

__version__ = "0.2.0"

# Convenience imports for common examples
from .basic_encoding import run_example as run_basic_encoding
from .trajectory_adaptation import run_example as run_trajectory_adaptation
from .dhb_qr_comparison import run_example as run_dhb_comparison

def __dir__():
    """List all available examples."""
    return [
        'run_basic_encoding',
        'run_trajectory_adaptation',
        'run_dhb_comparison',
        # Add more as needed
    ]