# StarVLA Quantized Smoke Results

Date: 2026-04-17

## Goal

Establish a minimal, reproducible baseline showing that default StarVLA VLM frameworks can run on the laptop GPU when the Qwen backbone is loaded in `8-bit` mode with CPU offload.

## Why This Was Needed

The stock fake-batch framework smokes for `QwenGR00T` and `QwenOFT` previously failed with CUDA OOM on the RTX 3500 laptop GPU when using full-precision or bf16 model loading.

The released StarVLA LIBERO checkpoint evaluation path already worked and produced rollout videos, so the remaining bottleneck was framework-level VRAM pressure during local model loading and forward.

## Local Changes Used

- Installed `bitsandbytes` in the Pixi environment.
- Added quantization support to:
  - `vendor/starVLA/starVLA/model/modules/vlm/QWen2_5.py`
  - `vendor/starVLA/starVLA/model/modules/vlm/QWen3.py`
- Patched framework runtime device placement so the action head follows the hidden-state device during quantized execution:
  - `vendor/starVLA/starVLA/model/framework/VLM4A/QwenGR00T.py`
  - `vendor/starVLA/starVLA/model/framework/VLM4A/QwenOFT.py`

## Quantization Settings

The successful runs used:

- `load_in_8bit = True`
- `llm_int8_enable_fp32_cpu_offload = True`
- `device_map = "auto"`
- base model: local `Qwen2.5-VL-3B-Instruct`

## Commands

### QwenGR00T

```bash
./.pixi-home/bin/pixi run python - <<'PY'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

repo = Path('vendor/starVLA').resolve()
sys.path.insert(0, str(repo))
from starVLA.model.framework.VLM4A.QwenGR00T import Qwen_GR00T

cfg = OmegaConf.load(str(repo / 'examples/Franka/train_files/starvla_cotrain_franka_single.yaml'))
cfg.framework.qwenvl.base_vlm = str((repo / 'playground/Pretrained_models/Qwen2.5-VL-3B-Instruct').resolve())
cfg.framework.qwenvl.load_in_8bit = True
cfg.framework.qwenvl.llm_int8_enable_fp32_cpu_offload = True
cfg.framework.qwenvl.device_map = 'auto'
model = Qwen_GR00T(cfg)
image = Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8))
sample = {
    'action': np.random.uniform(-1,1,size=(16,7)).astype(np.float16),
    'image': [image],
    'lang': 'This is a fake instruction for testing.',
}
out = model([sample, sample])
print('action_loss', float(out['action_loss']))
pred = model.predict_action([sample])
print('pred_shape', pred['normalized_actions'].shape)
PY
```

Observed result:

- `action_loss 1.4320616722106934`
- `pred_shape (1, 16, 7)`

### QwenOFT

```bash
./.pixi-home/bin/pixi run python - <<'PY'
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from omegaconf import OmegaConf

repo = Path('vendor/starVLA').resolve()
sys.path.insert(0, str(repo))
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT

cfg = OmegaConf.load(str(repo / 'examples/Franka/train_files/starvla_cotrain_franka_single.yaml'))
cfg.framework.name = 'QwenOFT'
cfg.framework.qwenvl.base_vlm = str((repo / 'playground/Pretrained_models/Qwen2.5-VL-3B-Instruct').resolve())
cfg.framework.qwenvl.load_in_8bit = True
cfg.framework.qwenvl.llm_int8_enable_fp32_cpu_offload = True
cfg.framework.qwenvl.device_map = 'auto'
model = Qwenvl_OFT(cfg)
image = Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8))
sample = {
    'action': np.random.uniform(-1,1,size=(16,7)).astype(np.float16),
    'image': [image],
    'lang': 'This is a fake instruction for testing.',
}
out = model([sample, sample])
print('action_loss', float(out['action_loss']))
pred = model.predict_action([sample])
print('pred_shape', pred['normalized_actions'].shape)
PY
```

Observed result:

- `action_loss 0.832609236240387`
- `pred_shape (1, 16, 7)`

## Interpretation

This does not prove that full StarVLA training is comfortable on this GPU. It does show that:

- the default Qwen VLM frameworks are not fundamentally blocked on laptop VRAM anymore
- `8-bit` quantization is sufficient for local framework smoke validation
- the next StarVLA bottlenecks are likely training-scale concerns, not basic model loading or forward

## Practical Use

Use this setup for:

- local framework smoke tests
- integration debugging
- minimal custom framework validation before larger runs

Do not treat it as the final training recipe for DHB or WM4A.
