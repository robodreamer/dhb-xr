# Vendor Dependencies

Third-party repos cloned here for DHB-XR CoRL 2026 development.
**These are NOT committed to git** (see `.gitignore`). Re-clone with:

```bash
cd vendor/
git clone https://github.com/starVLA/starVLA
git clone https://github.com/allenai/vla-evaluation-harness
```

---

## starVLA

| | |
|---|---|
| **Repo** | https://github.com/starVLA/starVLA |
| **Pinned commit** | `f8394a8` (2026-04-12) |
| **Purpose** | StarVLA training framework — WM4A backbone with modular action heads (FAST, OFT, GROIT, DIT). Used to train StarVLA-DHB variant. |
| **DHB integration** | `starVLA/model/modules/action_model/DHBActionHeader.py` (add), `WM4A_DHB.py` framework class (add) |
| **Config** | `configs/libero_dhbxr.yaml` (already in `configs/` of this repo) |
| **HuggingFace weights** | `StarVLA/Qwen2.5-VL-OFT-LIBERO-4in1`, `StarVLA/Qwen2.5-VL-FAST-LIBERO-4in1` |

### DHB patch needed (1-hour task)

Create `vendor/starVLA/starVLA/model/framework/WM4A/WM4A_DHB.py`:
```python
# Copy WM4A_OFT.py, change two lines:
@FRAMEWORK_REGISTRY.register("WM4A_DHB")   # was "WM4A_OFT"
class WM4A_DHB(WM4A_OFT):
    pass  # inherits everything; action head swapped via config
```

Then launch training:
```bash
cd vendor/starVLA
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes 4 \
    starVLA/training/train_starvla.py \
    --config_file ../../configs/model_servers/dhbxr.yaml \
    --framework.name WM4A_DHB
```

See `notes/supplementary/05_future_work_technical.md` for full training spec.

---

## vla-evaluation-harness

| | |
|---|---|
| **Repo** | https://github.com/allenai/vla-evaluation-harness |
| **Pinned commit** | `ae23527` (2026-04-14) |
| **Purpose** | AllenAI standardized VLA evaluation harness. Used to run DHB-XR as a model server against LIBERO and RoboCasa benchmarks. |
| **DHB integration** | `src/dhb_xr/integration/vla/harness_server.py` implements `PredictModelServer` interface |
| **Config** | `configs/model_servers/dhbxr.yaml` |

### Usage

```bash
# Install harness
cd vendor/vla-evaluation-harness
pip install -e .

# Start DHB model server
vla-eval serve --config configs/model_servers/dhbxr.yaml

# Run benchmark
vla-eval run --model dhbxr --benchmark libero_spatial --n-episodes 50
```

See `src/dhb_xr/integration/vla/harness_server.py` for full server implementation.

---

## Re-clone script

If this folder is missing (e.g. after a fresh clone of dhb_xr):

```bash
bash vendor/setup.sh
```
