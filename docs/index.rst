DHB-XR
======

DHB Extended Representations: SE(3) invariant trajectory encoding for robotics and VLA (Vision-Language-Action).

Features
--------

- **DHB-DR** (Double-Reflection): Euler-based invariants (4 values per component).
- **DHB-QR** (Quaternion-Relative): Quaternion-based invariants (5 values per component), no gimbal lock.
- **Trajectory adaptation**: Resample and decode with new initial pose; optional batched optimization.
- **Tokenization**: VQ-VAE / RVQ for DHB-Token (VLA action representation).
- **Motion database**: Storage, similarity (L2, DTW), and retrieval.
- **Imitation learning**: Invariant matching, SE(3) geodesic, and hybrid losses.

Quick start
-----------

.. code-block:: bash

   pip install dhb_xr
   # or with extras: pip install dhb_xr[gpu,database,optimization]

.. code-block:: python

   from dhb_xr import encode_dhb_dr, decode_dhb_dr
   from dhb_xr.core.types import DHBMethod

   # Encode trajectory (positions Nx3, quaternions Nx4 wxyz)
   out = encode_dhb_dr(positions, quaternions, method="pos", use_default_initial_frames=True, dhb_method=DHBMethod.DOUBLE_REFLECTION)
   # Decode back to poses
   decoded = decode_dhb_dr(out["linear_motion_invariants"], out["angular_motion_invariants"], out["initial_pose"], method="pos", dhb_method=DHBMethod.DOUBLE_REFLECTION, drop_padded=True)

Examples
--------

See ``docs/examples_walkthrough.md`` in the repository for a walkthrough of the example scripts and how to run them.

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
