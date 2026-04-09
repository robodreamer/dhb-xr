#!/usr/bin/env python
"""StarVLA + DHB-XR Training Pipeline Validation (Experiment 2 — CoRL 2026).

This script validates the DHB action head integration with StarVLA WITHOUT
running actual training. It:

1. Verifies DHBActionHeader can be imported and instantiated standalone.
2. Verifies get_action_model() works (StarVLA factory convention).
3. Runs a minimal forward pass (B=2, seq_len=20, input_dim=2048) and checks
   that the training loss is a finite scalar.
4. Explains the registration gap: why WM4A_OFT currently hard-wires MLP
   and what needs patching for DHB.
5. Shows the full accelerate launch command for Exp 2 of the paper.

Usage:
    cd /home/andypark/Projects/repos/dhb_xr_paper
    pixi run python examples/integration/starvla_dhb_training.py

No GPU required. No model weights downloaded. No actual training happens.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# Allow running from repo root without installing packages.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_STARVLA_PATH = _REPO_ROOT / "vendor" / "starVLA"
_DHB_SRC_PATH = _REPO_ROOT / "src"

for p in [str(_STARVLA_PATH), str(_DHB_SRC_PATH)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ── 1. Import validation ─────────────────────────────────────────────────────

def check_imports() -> bool:
    print("\n=== 1. Import validation ===")
    try:
        from starVLA.model.modules.action_model.DHBActionHeader import (
            DHBActionHeader,
            get_action_model,
            HAS_DHB,
        )
        print(f"  DHBActionHeader ............. OK")
        print(f"  get_action_model ............ OK")
        print(f"  dhb_xr backend (HAS_DHB) .... {HAS_DHB}")
        return True
    except ImportError as e:
        print(f"  FAILED: {e}")
        return False


# ── 2. Instantiation via factory (StarVLA convention) ────────────────────────

def _make_config(**action_model_kwargs):
    """Build a minimal config namespace that mirrors configs/libero_dhbxr.yaml.

    Uses types.SimpleNamespace so no OmegaConf dependency is needed in the
    pixi environment (OmegaConf lives in the vendor/starVLA requirements but
    not in the dhb_xr pixi env).
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        framework=SimpleNamespace(
            name="WM4A_DHB",
            backbone="qwen2vl_2b",
            action_model=SimpleNamespace(**action_model_kwargs),
        )
    )


def check_factory_instantiation() -> bool:
    print("\n=== 2. Factory instantiation (get_action_model) ===")
    from starVLA.model.modules.action_model.DHBActionHeader import get_action_model

    # Build a minimal config that mirrors configs/libero_dhbxr.yaml
    cfg = _make_config(
        action_model_type="DHB",
        action_hidden_dim=2048,
        hidden_dim=1024,
        action_dim=7,
        invariant_dim=8,
        num_oat_tokens=8,
        future_action_window_size=7,
        past_action_window_size=0,
        use_oat_quantizer=True,
        lambda_invariant=1.0,
        lambda_action=1.0,
    )

    try:
        model = get_action_model(config=cfg)
        chunk_len = 7 + 0 + 1  # future + past + 1
        print(f"  DHBActionHeader via factory .. OK")
        print(f"  chunk_len ................... {model.chunk_len}  (expected {chunk_len})")
        print(f"  action_dim .................. {model.action_dim}")
        print(f"  invariant_dim ............... {model.invariant_dim}")
        print(f"  OAT quantizer ............... {'enabled' if model.oat is not None else 'fallback MLP'}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ── 3. Minimal forward pass ───────────────────────────────────────────────────

def check_forward_pass() -> bool:
    print("\n=== 3. Minimal forward pass (B=2, seq_len=20, input_dim=2048) ===")
    import torch
    from starVLA.model.modules.action_model.DHBActionHeader import DHBActionHeader

    # --- 3a. Without OAT (fast, no dhb_xr dependency) ---
    print("  3a. Without OAT quantizer:")
    model_no_oat = DHBActionHeader(
        input_dim=2048,
        hidden_dim=1024,
        action_dim=7,
        invariant_dim=8,
        chunk_len=8,
        use_oat_quantizer=False,
    )
    hidden = torch.randn(2, 20, 2048)
    actions_target = torch.randn(2, 8, 7)
    loss = model_no_oat(hidden, actions_target=actions_target)
    assert loss.isfinite(), "Loss is NaN/Inf!"
    print(f"     loss = {loss.item():.6f}  (finite: {loss.isfinite().item()})")

    # --- 3b. With OAT (requires dhb_xr in src/) ---
    from starVLA.model.modules.action_model.DHBActionHeader import HAS_DHB
    if HAS_DHB:
        print("  3b. With OAT quantizer (dhb_xr backend):")
        model_oat = DHBActionHeader(
            input_dim=2048,
            hidden_dim=1024,
            action_dim=7,
            invariant_dim=8,
            num_oat_tokens=8,
            chunk_len=8,
            use_oat_quantizer=True,
        )
        loss_oat = model_oat(hidden, actions_target=actions_target)
        assert loss_oat.isfinite(), "OAT loss is NaN/Inf!"
        print(f"     loss = {loss_oat.item():.6f}  (finite: {loss_oat.isfinite().item()})")
    else:
        print("  3b. OAT skipped — dhb_xr not importable from sys.path")

    # --- 3c. Inference (predict_action) ---
    print("  3c. predict_action inference:")
    with torch.no_grad():
        pred = model_no_oat.predict_action(hidden)
    assert pred.shape == (2, 8, 7), f"Unexpected shape: {pred.shape}"
    print(f"     output shape = {tuple(pred.shape)}  (expected (2, 8, 7))")

    return True


# ── 4. Registration gap analysis ─────────────────────────────────────────────

def explain_registration_gap() -> None:
    print("\n=== 4. Registration gap: what needs patching for full StarVLA training ===")
    print("""
  StarVLA's framework registry works like this:
    - Each framework class (QwenOFT, WM4A_OFT, …) is registered via
      @FRAMEWORK_REGISTRY.register("FrameworkName").
    - The framework's __init__ hardcodes which action head it imports:
        WM4A_OFT  → from …MLP_ActionHeader  import get_action_model
        QwenFast  → from …fast_ActionHeader import get_action_model
        QwenOFT   → from …MLP_ActionHeader  import get_action_model

  The config field `action_model_type: "DHB"` is a label, but the actual
  head class is determined by the import in the framework file, NOT the
  string.  MLP_ActionHeader.get_action_model() ignores action_model_type.

  STATUS: DHBActionHeader.get_action_model() exists and is StarVLA-compatible,
  but there is NO framework class that imports it yet.

  REQUIRED PATCH (one of two options):

  Option A — Add a new framework class WM4A_DHB (recommended, clean):
    File: vendor/starVLA/starVLA/model/framework/WM4A/WM4A_DHB.py
    - Copy WM4A_OFT.py
    - Replace: from …MLP_ActionHeader import get_action_model
      with:     from …DHBActionHeader  import get_action_model
    - Change @FRAMEWORK_REGISTRY.register("WM4A_OFT") → ("WM4A_DHB")
    - Update configs/libero_dhbxr.yaml:  framework.name: "WM4A_DHB"
    - Change forward() to pass gt_invariants when available

  Option B — Add DHB dispatch to WM4A_OFT (quick but messy):
    In WM4A_OFT.__init__(), after self.config is set:
      if self.config.framework.action_model.action_model_type == "DHB":
          from …DHBActionHeader import get_action_model as get_dhb_model
          self.action_model = get_dhb_model(config=self.config)
      else:
          self.action_model = get_action_model(config=self.config)

  For the paper, Option A is preferred (clean ablation boundary).
""")


# ── 5. Full training command ──────────────────────────────────────────────────

def show_training_command() -> None:
    print("\n=== 5. Full training command for CoRL 2026 Experiment 2 ===")
    print("""
  Prerequisites:
    1. Apply WM4A_DHB framework patch (see section 4 above).
    2. Update configs/libero_dhbxr.yaml: framework.name: "WM4A_DHB"
    3. Ensure LIBERO data is in LeRobot format OR write a thin adapter.
       (Current LIBERO HDF5 structure:
          data/demo_N/actions  (T, 7)
          data/demo_N/obs/agentview_rgb  (T, 128, 128, 3)
          data/demo_N/obs/ee_pos / ee_ori / gripper_states)

  Training command (single-node, 4 GPU):
  ──────────────────────────────────────────────────────────────────────────
  cd /home/andypark/Projects/repos/dhb_xr_paper/vendor/starVLA

  accelerate launch \\
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \\
    --num_processes 4 \\
    starVLA/training/train_starvla.py \\
    --config_yaml configs/libero_dhbxr.yaml \\
    --framework.name WM4A_DHB \\
    --framework.action_model.use_oat_quantizer true \\
    --datasets.vla_data.data_root_dir /home/andypark/Projects/data/libero \\
    --datasets.vla_data.data_mix libero_spatial \\
    --datasets.vla_data.per_device_batch_size 16 \\
    --trainer.max_train_steps 50000 \\
    --trainer.save_interval 5000 \\
    --trainer.logging_frequency 100 \\
    --trainer.eval_interval 1000 \\
    --run_root_dir ./results/Checkpoints \\
    --run_id exp2_dhbxr_libero_spatial_50demo \\
    --wandb_project dhb_xr_corl2026 \\
    --wandb_entity <your_entity>
  ──────────────────────────────────────────────────────────────────────────

  Ablation variant (no OAT quantizer):
    Add: --framework.action_model.use_oat_quantizer false
    Change run_id: exp2_dhbxr_no_oat_libero_spatial_50demo

  Baseline variant (MLP head, same backbone):
    Change: --framework.name WM4A_OFT
    Change run_id: exp2_mlp_baseline_libero_spatial_50demo
""")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("StarVLA + DHB-XR Training Pipeline Validation")
    print("CoRL 2026 — Experiment 2: LIBERO-Spatial with Spatial Perturbations")
    print("=" * 70)

    results = {}
    results["imports"] = check_imports()
    results["factory"] = check_factory_instantiation()
    results["forward"] = check_forward_pass()

    explain_registration_gap()
    show_training_command()

    print("\n=== Summary ===")
    all_ok = True
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {name:20s} {status}")

    if all_ok:
        print("\nAll validation checks passed.")
        print("Next step: implement WM4A_DHB framework class (see section 4).")
    else:
        print("\nSome checks failed. See output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
