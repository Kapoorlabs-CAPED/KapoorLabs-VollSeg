"""PyTorch CARE backbone — wraps the careamics UNet inside a CareModule.

This is the new first-class CARE backbone. It owns:

- the underlying ``careamics.models.unet.UNet`` (architecture)
- the :class:`kapoorlabs_vollseg._lightning.CareModule` (Lightning module that
  shapes inputs as ``(B, C, Z, Y, X)`` and exposes ``predict_step`` for
  tiled inference)

Loading a checkpoint that was trained via ``kapoorlabs-lightning`` works
out of the box because we mirror its ``CareModule`` shape (network is
held as ``self.network``, hyperparameters ignored on
``load_from_checkpoint``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from .._lightning.care_module import CareModule


def _build_unet(
    *,
    conv_dims: int = 3,
    in_channels: int = 1,
    num_classes: int = 1,
    depth: int = 3,
    num_channels_init: int = 64,
    use_batch_norm: bool = True,
):
    """Local import — careamics is heavy and we only need the UNet."""
    from careamics.models.unet import UNet

    return UNet(
        conv_dims=conv_dims,
        in_channels=in_channels,
        num_classes=num_classes,
        depth=depth,
        num_channels_init=num_channels_init,
        use_batch_norm=use_batch_norm,
    )


class CAREBackbone:
    """Hold a trained CareModule, plus the architecture knobs needed to rebuild it.

    Parameters
    ----------
    care_module
        A :class:`CareModule` instance with weights loaded.
    """

    def __init__(self, care_module: CareModule):
        self.module = care_module
        self.module.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Union[str, Path],
        *,
        conv_dims: int = 3,
        in_channels: int = 1,
        num_classes: int = 1,
        depth: int = 3,
        num_channels_init: int = 64,
        use_batch_norm: bool = True,
        map_location: Optional[str] = None,
    ) -> CAREBackbone:
        """Build a CAREBackbone from a Lightning ``.ckpt`` file.

        Architecture knobs must match the ones used at training time —
        these are stored alongside the checkpoint as
        ``{experiment_name}.json`` by ``CareInception``, but we keep the
        constructor explicit so the caller is in control.
        """
        unet = _build_unet(
            conv_dims=conv_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            depth=depth,
            num_channels_init=num_channels_init,
            use_batch_norm=use_batch_norm,
        )
        module = CareModule.load_from_checkpoint(
            checkpoint_path=str(checkpoint),
            network=unet,
            loss_func=torch.nn.MSELoss(),
            optim_func=None,
            map_location=map_location,
        )
        return cls(module)
