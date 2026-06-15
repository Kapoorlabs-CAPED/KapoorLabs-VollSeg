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

import importlib.abc
import importlib.machinery
import sys
import types
from pathlib import Path
from typing import Optional, Union

import torch

from ..care_lightning.module import CareModule


def _make_stub_getattr(stub_module):
    """Build a ``__getattr__`` that returns a dummy class for any name.

    Dunder attributes (``__file__``, ``__name__``, ``__loader__``, …)
    must NOT be intercepted — Python's ``inspect`` / ``traceback`` /
    ``importlib`` machinery probes them via ``getattr`` and expects
    strings or ``None``, never a class. Returning a class for
    ``__file__`` produces ``AttributeError("type object '__file__' has
    no attribute 'endswith'")`` when Hydra tries to format a traceback.
    """

    def __getattr__(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return type(name, (object,), {"__module__": stub_module.__name__})

    return __getattr__


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        stub = types.ModuleType(spec.name)
        stub.__path__ = []  # mark as namespace package so submodules resolve
        stub.__file__ = "<kapoorlabs_vollseg pickle stub>"
        stub._kapoorlabs_stub = True
        stub.__getattr__ = _make_stub_getattr(stub)
        return stub

    def exec_module(self, module):
        # Body is intentionally empty — the stub is fully populated by
        # ``create_module`` and resolves any further attribute access
        # lazily via ``__getattr__``.
        return None


class _StubFinder(importlib.abc.MetaPathFinder):
    """Catch-all finder for any submodule under a missing top-level package.

    Installed once at import time when the real package isn't available,
    so ``torch.load`` can deserialize Lightning checkpoints that pickled
    class references from ANY submodule of that package (e.g.
    ``kapoorlabs_lightning.care_presets``,
    ``kapoorlabs_lightning.care_transforms``,
    ``kapoorlabs_lightning.optimizers``…) without us having to enumerate
    every one. The class references end up as dummy classes; Lightning
    rebuilds the optimizer / scheduler / transforms fresh on
    ``load_from_checkpoint`` so the dummies are never instantiated.
    """

    def __init__(self, prefix: str):
        self.prefix = prefix

    def find_spec(self, fullname, path, target=None):
        if fullname == self.prefix or fullname.startswith(self.prefix + "."):
            return importlib.machinery.ModuleSpec(fullname, _StubLoader())
        return None


def _install_pickle_stub_finder(prefix: str) -> None:
    """Install a :class:`_StubFinder` for ``prefix`` unless the real
    top-level package is already importable."""
    try:
        __import__(prefix)
        return  # real package present; no stubbing needed
    except ImportError:
        pass
    if any(isinstance(f, _StubFinder) and f.prefix == prefix for f in sys.meta_path):
        return  # already installed
    sys.meta_path.append(_StubFinder(prefix))


# Catch every ``kapoorlabs_lightning.*`` reference in pickled
# checkpoints written by training runs that used the standalone
# kapoorlabs-lightning package — care_presets, care_transforms,
# optimizers, schedulers, anything else. The prediction env doesn't
# need that package installed.
_install_pickle_stub_finder("kapoorlabs_lightning")


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
