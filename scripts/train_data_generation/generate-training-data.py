"""Generate the unified SmartPatches H5 for both U-Net and StarDist training.

Same patches, same H5 — different keys read by each trainer.

Output H5::

    /train/raw    (N, *patch_shape)   float32      # always
    /train/label  (N, *patch_shape)   int32        # always (StarDist target source)
    /train/mask   (N, *patch_shape)   uint8        # only if binary_mask_dir was set
    /val/raw, /val/label, /val/mask?              same shape, fewer rows

The val split is foreground-only (no paste augmentation). The
StarDist dataset reads ``raw + label`` and derives ``(prob, dist)``
targets on the fly. The U-Net dataset reads ``raw + mask`` if mask is
present, else falls back to deriving binary from ``label``.
"""

from __future__ import annotations

import os

import hydra
from hydra.core.config_store import ConfigStore

from kapoorlabs_vollseg.data import generate_smart_patches_h5

from scenario_generate import GenScenario


ConfigStore.instance().store(name="GenScenario", node=GenScenario)


@hydra.main(config_path="conf", config_name="scenario_generate", version_base="1.3")
def main(config: GenScenario):
    base = config.train_data_paths.base_data_dir
    raw_dir = os.path.join(base, config.train_data_paths.raw_dir)
    label_dir = os.path.join(base, config.train_data_paths.label_dir)
    h5_path = os.path.join(base, config.train_data_paths.h5_file)

    binary_mask_dir = None
    if config.train_data_paths.binary_mask_dir:
        binary_mask_dir = os.path.join(base, config.train_data_paths.binary_mask_dir)

    p = config.parameters
    counts = generate_smart_patches_h5(
        raw_dir=raw_dir,
        label_dir=label_dir,
        output_h5=h5_path,
        binary_mask_dir=binary_mask_dir,
        patch_shape=tuple(p.patch_shape),
        val_files=p.val_files,
        lower_ratio_fore_to_back=p.lower_ratio_fore_to_back,
        upper_ratio_fore_to_back=p.upper_ratio_fore_to_back,
        erosion_iterations=p.erosion_iterations,
        max_foreground_patches_per_image=p.max_foreground_patches_per_image,
        paste_augmentation=p.paste_augmentation,
        max_paste_patches_per_image=p.max_paste_patches_per_image,
        seed=p.seed,
        overwrite=p.overwrite,
    )
    print(f"Counts: {counts}")


if __name__ == "__main__":
    main()
