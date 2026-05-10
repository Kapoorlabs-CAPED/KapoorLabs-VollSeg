"""Lightning training entry point for U-Net (PyTorch first-class).

Reads an H5 produced by ``generate-unet-training-data.py`` (streaming
``raw + mask`` patches) and trains via :class:`kapoorlabs_vollseg.UNetTrainer`. The
shape mirrors KapoorLabs-Lightning's ``lightning-roi.py`` — same Hydra
schema convention — but uses our in-repo PyTorch trainer instead of
``CareInception``.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch
from hydra.core.config_store import ConfigStore
from torch.utils.data import DataLoader

from kapoorlabs_vollseg import UNetTrainer
from kapoorlabs_vollseg._lightning import H5UNetDataset, unet_collate

from scenario_train_unet import UNetTrainScenario


ConfigStore.instance().store(name="UNetTrainScenario", node=UNetTrainScenario)


def _percentile_normalize(raw: torch.Tensor, pmin: float, pmax: float) -> torch.Tensor:
    flat = raw.flatten()
    lo = torch.quantile(flat, pmin / 100.0)
    hi = torch.quantile(flat, pmax / 100.0)
    return (raw - lo) / (hi - lo + 1e-8)


def _make_transform(pmin: float, pmax: float, augment: bool, gaussian_noise_std: float):
    """Returns a (raw, mask) -> (raw, mask) callable."""

    def transform(raw, mask):
        raw = _percentile_normalize(raw, pmin, pmax)
        if augment:
            # Per-axis flip, joint to raw + mask.
            for axis in range(raw.dim()):
                if torch.rand(1).item() < 0.5:
                    raw = torch.flip(raw, dims=[axis])
                    mask = torch.flip(mask, dims=[axis])
            # YX-plane 90° rotation (k random in 0..3).
            if raw.dim() >= 2 and torch.rand(1).item() < 0.5:
                k = int(torch.randint(0, 4, (1,)).item())
                if k > 0:
                    raw = torch.rot90(raw, k=k, dims=[-2, -1])
                    mask = torch.rot90(mask, k=k, dims=[-2, -1])
            if gaussian_noise_std > 0 and torch.rand(1).item() < 0.3:
                raw = raw + torch.randn_like(raw) * gaussian_noise_std
        return raw, mask

    return transform


@hydra.main(config_path="conf", config_name="scenario_train_unet", version_base="1.3")
def main(config: UNetTrainScenario):
    base = config.train_data_paths.base_data_dir
    h5_path = os.path.join(base, config.train_data_paths.h5_file)
    log_path = config.train_data_paths.log_path
    experiment = config.train_data_paths.experiment_name
    Path(log_path).mkdir(parents=True, exist_ok=True)

    p = config.parameters
    train_tf = _make_transform(p.pmin, p.pmax, p.augment, p.gaussian_noise_std)
    val_tf = _make_transform(p.pmin, p.pmax, augment=False, gaussian_noise_std=0.0)

    train_ds = H5UNetDataset(h5_path, split="train", transform=train_tf)
    val_ds = H5UNetDataset(h5_path, split="val", transform=val_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=p.batch_size,
        shuffle=True,
        num_workers=p.num_workers,
        collate_fn=unet_collate,
        persistent_workers=p.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=p.batch_size,
        shuffle=False,
        num_workers=p.num_workers,
        collate_fn=unet_collate,
        persistent_workers=p.num_workers > 0,
    )

    trainer_obj = UNetTrainer(
        model_name=experiment,
        model_dir=log_path,
        epochs=p.epochs,
        batch_size=p.batch_size,
        learning_rate=p.learning_rate,
        conv_dims=p.conv_dims,
        in_channels=p.in_channels,
        num_classes=p.num_classes,
        unet_depth=p.unet_depth,
        num_channels_init=p.num_channels_init,
        use_batch_norm=p.use_batch_norm,
        n_tiles=list(p.n_tiles),
        tile_overlap=p.tile_overlap,
        accelerator=p.accelerator,
        devices=p.devices,
        precision=p.train_precision,
        strategy=p.strategy,
    )
    trainer_obj.fit(train_dataloader=train_loader, val_dataloader=val_loader)
    print(f"Done. Checkpoints in {log_path}/")


if __name__ == "__main__":
    main()
