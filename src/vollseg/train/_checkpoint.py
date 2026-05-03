"""Tiny helper: load any present ``weights_*.h5`` checkpoint into a model.

The original VollSeg trainers had four copies of this if/else block per
trainer; consolidating it here keeps each ``Trainer.fit`` short.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


_DEFAULT_CHECKPOINTS: tuple = ("weights_now.h5", "weights_last.h5", "weights_best.h5")


def load_latest_checkpoint(
    model,
    model_dir: Path,
    model_name: str,
    candidates: Iterable[str] = _DEFAULT_CHECKPOINTS,
) -> str | None:
    """Load the last existing checkpoint from ``candidates``. Returns its name.

    Loading happens in order, so a later entry in ``candidates`` overrides
    earlier ones — matching the original VollSeg behavior where ``best`` is
    preferred over ``last`` over ``now``.
    """
    loaded = None
    for cp in candidates:
        path = model_dir / model_name / cp
        if os.path.exists(path):
            print(f"Loading checkpoint: {cp}")
            model.load_weights(os.fspath(path))
            loaded = cp
    return loaded
