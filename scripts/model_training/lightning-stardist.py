"""Lightning training entry point for StarDist (PyTorch first-class).

Reads an H5 produced by ``generate-stardist-training-data.py`` (streaming
``raw + label`` patches) and trains via :class:`kapoorlabs_vollseg.StarDistTrainer`.
Per-batch ``(prob, dist)`` targets are derived inside the dataset from
the (possibly augmented) labels.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from kapoorlabs_vollseg import StarDistTrainer
from kapoorlabs_vollseg.stardist import (
    Compose,
    InputGaussianNoise,
    InputPercentileNormalize,
    RandomFlip,
    RandomRot90,
    StarDistH5Dataset,
    rays_2d,
    rays_3d_golden_spiral,
    stardist_collate,
)

from scenario_train_stardist import StarDistTrainScenario


ConfigStore.instance().store(name="StarDistTrainScenario", node=StarDistTrainScenario)


def _build_rays(p):
    if p.conv_dims == 2:
        return rays_2d(p.n_rays)
    aniso = tuple(p.anisotropy) if p.anisotropy else None
    return rays_3d_golden_spiral(p.n_rays, anisotropy=aniso)


def _build_transform(p):
    transforms = [InputPercentileNormalize(pmin=p.pmin, pmax=p.pmax)]
    if p.augment:
        transforms.extend([RandomFlip(p=0.5), RandomRot90(p=0.5)])
        if p.gaussian_noise_std > 0:
            transforms.append(InputGaussianNoise(std=p.gaussian_noise_std, p=0.3))
    return Compose(transforms)


@hydra.main(
    config_path="conf", config_name="scenario_train_stardist", version_base="1.3"
)
def main(config: StarDistTrainScenario):
    base = config.train_data_paths.base_data_dir
    h5_path = os.path.join(base, config.train_data_paths.h5_file)
    log_path = config.train_data_paths.log_path
    experiment = config.train_data_paths.experiment_name
    Path(log_path).mkdir(parents=True, exist_ok=True)

    p = config.parameters
    rays = _build_rays(p)

    train_tf = _build_transform(p)
    val_tf = InputPercentileNormalize(pmin=p.pmin, pmax=p.pmax)

    train_ds = StarDistH5Dataset(h5_path, split="train", rays=rays, transform=train_tf)
    val_ds = StarDistH5Dataset(h5_path, split="val", rays=rays, transform=val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=p.batch_size,
        shuffle=True,
        num_workers=p.num_workers,
        collate_fn=stardist_collate,
        persistent_workers=p.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=p.batch_size,
        shuffle=False,
        num_workers=p.num_workers,
        collate_fn=stardist_collate,
        persistent_workers=p.num_workers > 0,
    )

    trainer_obj = StarDistTrainer(
        model_name=experiment,
        model_dir=log_path,
        rays=rays,
        epochs=p.epochs,
        batch_size=p.batch_size,
        learning_rate=p.learning_rate,
        conv_dims=p.conv_dims,
        in_channels=p.in_channels,
        depth=p.unet_depth,
        num_channels_init=p.num_channels_init,
        use_batch_norm=p.use_batch_norm,
        loss_lam=p.loss_lam,
        n_tiles=list(p.n_tiles),
        tile_overlap=p.tile_overlap,
        accelerator=p.accelerator,
        devices=p.devices,
        precision=p.train_precision,
        strategy=p.strategy,
    )
    trainer_obj.fit(train_dataloader=train_loader, val_dataloader=val_loader)
    print(f"Done. Model + rays sidecar in {log_path}/")


if __name__ == "__main__":
    main()
