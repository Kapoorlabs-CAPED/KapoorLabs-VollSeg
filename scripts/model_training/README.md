# Model training

PyTorch Lightning training entry points for U-Net and StarDist, reading
the streaming H5 files produced under [`../train_data_generation/`](../train_data_generation/).
Mirrors the `lightning-care.py` / `lightning-roi.py` shape used in
KapoorLabs-Lightning.

## Layout

```
model_training/
├── lightning-unet.py             U-Net training (uses vollseg.UNetTrainer)
├── lightning-stardist.py         StarDist training (uses vollseg.StarDistTrainer)
├── scenario_train_unet.py        hydra dataclass schema
├── scenario_train_stardist.py    hydra dataclass schema
├── conf/
│   ├── scenario_train_unet.yaml
│   ├── scenario_train_stardist.yaml
│   ├── parameters/
│   │   ├── unet_default.yaml         architecture + training knobs
│   │   └── stardist_default.yaml
│   └── train_data_paths/
│       ├── xenopus_default.yaml      edit-locally template
│       ├── xenopus_jeanzay.yaml      jean-zay paths
│       └── xenopus_gwdg.yaml         gwdg/grete paths
└── slurm_*.sh                        jeanzay + gwdg submission scripts
```

## Usage

```bash
# Local
python lightning-unet.py
python lightning-stardist.py

# Per-cluster path overrides
python lightning-stardist.py train_data_paths=xenopus_jeanzay
python lightning-unet.py    train_data_paths=xenopus_gwdg

# Tweak any field
python lightning-stardist.py \
    parameters.batch_size=8 \
    parameters.augment=false \
    parameters.n_rays=64
```

## What gets written

**U-Net:**
- `<log_path>/<experiment_name>/last.ckpt` (Lightning checkpoint)
- `<log_path>/<experiment_name>.json` (architecture knobs for predict-time rebuild)

**StarDist:**
- `<log_path>/<experiment_name>/last.ckpt`
- `<log_path>/<experiment_name>.rays.npy` (ray geometry sidecar — required at predict)
- `<log_path>/<experiment_name>.json`

## Predict from a trained model

```python
import numpy as np
from kapoorlabs_vollseg import StarDistSegmenter, UNetSegmenter

# StarDist
seg = StarDistSegmenter.from_checkpoint(
    "<log_path>/<experiment_name>/last.ckpt",
    rays=np.load("<log_path>/<experiment_name>.rays.npy"),
)

# U-Net (read architecture knobs from the JSON sidecar)
import json
with open("<log_path>/<experiment_name>.json") as f:
    arch = json.load(f)
seg = UNetSegmenter.from_checkpoint(
    "<log_path>/<experiment_name>/last.ckpt",
    depth=arch["unet_depth"],
    num_channels_init=arch["num_channels_init"],
)
```

## SLURM

```bash
sbatch slurm_train_unet_jeanzay.sh
sbatch slurm_train_unet_gwdg.sh
sbatch slurm_train_stardist_jeanzay.sh
sbatch slurm_train_stardist_gwdg.sh
```

GPU jobs (1×A100). The Jean-Zay scripts request the `gpu_p5` A100
partition under the `lzc` account. The GWDG (Grete) scripts use the
`grete:shared` partition. Adjust as needed.
