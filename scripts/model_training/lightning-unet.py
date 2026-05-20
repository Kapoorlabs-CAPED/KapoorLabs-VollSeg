"""U-Net training — pure :class:`TrainingPipeline` invocation.

Same shape as ``lightning-stardist`` / ``lightning-care``: hydra config
→ sequence of ``setup_*`` calls → ``train()``. No per-task trainer
façade, no hand-rolled dataloaders — the pipeline owns transforms,
datasets, dataloaders, optimizer, scheduler, module, callbacks, logger,
and the Lightning fit loop.

Reads an H5 produced by ``generate-unet-training-data.py`` — streaming
``raw + mask`` patches.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from kapoorlabs_vollseg.training import TrainingPipeline

from scenario_train_unet import UNetTrainScenario


ConfigStore.instance().store(name="UNetTrainScenario", node=UNetTrainScenario)


def _save_sidecars(log_path: Path, experiment: str, p) -> None:
    """Persist arch JSON next to the checkpoints so
    ``UNetSegmenter.from_folder`` can rebuild at inference time."""
    params = {
        "model_name": experiment,
        "conv_dims": p.conv_dims,
        "in_channels": p.in_channels,
        "num_classes": p.num_classes,
        "unet_depth": p.unet_depth,
        "depth": p.unet_depth,
        "num_channels_init": p.num_channels_init,
        "use_batch_norm": p.use_batch_norm,
        "n_tiles": list(p.n_tiles),
        "tile_overlap": p.tile_overlap,
        "learning_rate": p.learning_rate,
        "batch_size": p.batch_size,
        "epochs": p.epochs,
        "optimizer": OmegaConf.select(p, "optimizer", default="adam"),
        "scheduler": OmegaConf.select(p, "scheduler", default=None),
    }
    (log_path / "training_config.json").write_text(
        json.dumps({"parameters": params}, indent=2)
    )
    (log_path / f"{experiment}.json").write_text(json.dumps(params, indent=2))


@hydra.main(config_path="conf", config_name="scenario_train_unet", version_base="1.3")
def main(config: UNetTrainScenario):
    base = config.train_data_paths.base_data_dir
    h5_path = os.path.join(base, config.train_data_paths.h5_file)
    log_path = Path(config.train_data_paths.log_path)
    experiment = config.train_data_paths.experiment_name
    log_path.mkdir(parents=True, exist_ok=True)

    p = config.parameters

    optimizer_name = OmegaConf.select(p, "optimizer", default="adam")
    optimizer_kwargs = (
        OmegaConf.to_container(
            OmegaConf.select(p, "optimizer_kwargs", default={}), resolve=True
        )
        or {}
    )
    scheduler_name = OmegaConf.select(p, "scheduler", default=None)
    scheduler_kwargs = (
        OmegaConf.to_container(
            OmegaConf.select(p, "scheduler_kwargs", default={}), resolve=True
        )
        or {}
    )

    pipe = TrainingPipeline(
        experiment_name=experiment,
        log_path=log_path,
        epochs=p.epochs,
        learning_rate=p.learning_rate,
        accelerator=p.accelerator,
        devices=p.devices,
        precision=p.train_precision,
        strategy=p.strategy,
    )
    pipe.setup_unet_transforms(
        pmin=p.pmin,
        pmax=p.pmax,
        augment=p.augment,
        gaussian_noise_std=p.gaussian_noise_std,
    )
    pipe.setup_unet_h5_datasets(
        h5_file=h5_path,
        batch_size=p.batch_size,
        num_workers=p.num_workers,
    )
    pipe.setup_unet_model(
        conv_dims=p.conv_dims,
        in_channels=p.in_channels,
        num_classes=p.num_classes,
        unet_depth=p.unet_depth,
        num_channels_init=p.num_channels_init,
        use_batch_norm=p.use_batch_norm,
    )
    pipe.setup_optimizer(optimizer_name, **optimizer_kwargs)
    pipe.setup_scheduler(scheduler_name, **scheduler_kwargs)
    pipe.setup_care_module(
        n_tiles=list(p.n_tiles),
        tile_overlap=p.tile_overlap,
    )
    pipe.setup_checkpointing()
    pipe.setup_csv_logger()

    _save_sidecars(log_path, experiment, p)
    pipe.train()
    print(f"Done. Checkpoints in {log_path}/")


if __name__ == "__main__":
    main()
