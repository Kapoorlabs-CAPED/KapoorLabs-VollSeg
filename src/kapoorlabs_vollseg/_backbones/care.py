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

import sys
import types
from pathlib import Path
from typing import Optional, Union

import torch

from ..care_lightning.module import CareModule


def _install_pickle_stubs(*module_paths: str) -> None:
    """Install stub modules that satisfy ``torch.load``'s class lookups.

    Lightning checkpoints pickle class references for the scheduler /
    optimizer / preset modules used at training time
    (e.g. ``kapoorlabs_lightning.care_presets``). When the prediction
    environment doesn't have those packages installed, ``torch.load``
    raises ``ModuleNotFoundError`` before it ever gets to the
    state_dict — even though for architecture inference we only need
    tensor shapes.

    For each missing dotted path, install a ``types.ModuleType`` whose
    ``__getattr__`` returns a freshly-minted dummy class. Pickle's
    ``find_class`` is satisfied, the class is never actually
    instantiated with real state (Lightning re-builds the optimizer
    fresh on ``load_from_checkpoint``), and the state_dict loads
    normally. No-op when the real module is importable.

    Stubs are tagged with a ``_kapoorlabs_stub`` attribute so a later
    call doesn't overwrite a real module that happened to get imported
    in between.
    """
    for path in module_paths:
        try:
            __import__(path)
            continue  # real module is importable; nothing to do
        except ImportError:
            pass
        parts = path.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            existing = sys.modules.get(sub)
            if existing is not None and not getattr(
                existing, "_kapoorlabs_stub", False
            ):
                continue  # real package already loaded
            stub = types.ModuleType(sub)
            # ``__path__ = []`` marks this as a namespace package, so
            # ``__import__("foo.bar")`` can resolve ``bar`` against this
            # parent stub without hitting filesystem lookup machinery.
            stub.__path__ = []
            stub._kapoorlabs_stub = True
            stub.__getattr__ = lambda name, _s=stub: type(
                name, (object,), {"__module__": _s.__name__}
            )
            sys.modules[sub] = stub


# Stubs are installed once at import time — keeps the prediction env
# lean (no need to install kapoorlabs-lightning just to deserialize a
# checkpoint that referenced one of its preset classes).
_install_pickle_stubs(
    "kapoorlabs_lightning.care_presets",
    "kapoorlabs_lightning.optimizers",
    "kapoorlabs_lightning.schedulers",
)


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


def infer_arch_from_checkpoint(
    checkpoint: Union[str, Path],
    *,
    weights_only: bool = False,
) -> dict:
    """Read a Lightning ``.ckpt`` and infer the trained architecture.

    Returns ``{"conv_dims", "in_channels", "num_classes",
    "num_channels_init", "depth", "use_batch_norm"}`` from the
    state_dict's conv weight shapes — bulletproof against drift in
    the Hydra training config:

    - ``conv_dims`` from first encoder conv weight ``ndim`` (4 → 2D, 5 → 3D)
    - ``in_channels``, ``num_channels_init`` from its in/out channel sizes
    - ``num_classes`` from the final conv's out channels
    - ``depth`` from the count of ``encoder_blocks.{i}.conv1.weight`` entries
    - ``use_batch_norm`` from the presence of any ``batch_norm*`` key

    Architecture knobs explicitly passed to ``from_checkpoint`` still
    win — this is only consulted to fill in ``None``.
    """
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=weights_only)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # Prefix differs between training entry-points; tolerate any.
    first_key = next(
        (k for k in state if k.endswith("encoder.encoder_blocks.0.conv1.weight")),
        None,
    )
    if first_key is None:
        raise ValueError(
            f"Could not locate first encoder conv weight in {checkpoint}; "
            f"cannot infer architecture."
        )
    prefix = first_key[: -len("encoder.encoder_blocks.0.conv1.weight")]

    first_w = state[first_key]
    conv_dims = first_w.ndim - 2  # (out, in, *spatial)
    num_channels_init = int(first_w.shape[0])
    in_channels = int(first_w.shape[1])

    final_key = next(
        (k for k in state if k.endswith("final_conv.weight") and k.startswith(prefix)),
        None,
    )
    num_classes = int(state[final_key].shape[0]) if final_key is not None else 1

    # Each downsampling level adds two conv-block weight tensors named
    # ``encoder_blocks.{2*level}.conv1.weight`` (the odd indices are
    # MaxPool with no weight). So depth = count of those.
    depth = sum(
        1
        for k in state
        if k.startswith(f"{prefix}encoder.encoder_blocks.")
        and k.endswith(".conv1.weight")
    )

    use_batch_norm = any(".batch_norm" in k for k in state if k.startswith(prefix))

    return {
        "conv_dims": conv_dims,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "num_channels_init": num_channels_init,
        "depth": depth,
        "use_batch_norm": use_batch_norm,
    }


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
        conv_dims: Optional[int] = None,
        in_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        depth: Optional[int] = None,
        num_channels_init: Optional[int] = None,
        use_batch_norm: Optional[bool] = None,
        map_location: Optional[str] = None,
        weights_only: bool = False,
    ) -> CAREBackbone:
        """Build a CAREBackbone from a Lightning ``.ckpt`` file.

        All architecture knobs default to ``None`` — when not supplied,
        they're inferred directly from the checkpoint's state_dict by
        :func:`infer_arch_from_checkpoint`. So a 2D-trained ROI
        Mask-UNet (``conv_dims=2`` per
        ``KapoorLabs-Lightning/scripts/conf/parameters/roi.yaml``) and
        a 3D-trained CARE U-Net both load with the same call.

        ``weights_only`` defaults to ``False`` because Hydra-trained
        checkpoints pickle ``omegaconf.ListConfig`` (and friends) into
        ``hyper_parameters``, which trips PyTorch 2.6's safe-load
        default. Flip to ``True`` if you want strict torch.load.
        """
        arch = infer_arch_from_checkpoint(checkpoint, weights_only=weights_only)
        # Caller-supplied values win; otherwise fall back to detection.
        conv_dims = conv_dims if conv_dims is not None else arch["conv_dims"]
        in_channels = in_channels if in_channels is not None else arch["in_channels"]
        num_classes = num_classes if num_classes is not None else arch["num_classes"]
        depth = depth if depth is not None else arch["depth"]
        num_channels_init = (
            num_channels_init
            if num_channels_init is not None
            else arch["num_channels_init"]
        )
        use_batch_norm = (
            use_batch_norm if use_batch_norm is not None else arch["use_batch_norm"]
        )
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
            weights_only=weights_only,
        )
        return cls(module)
