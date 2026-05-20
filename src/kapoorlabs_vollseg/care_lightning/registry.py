"""String-keyed registry that maps yaml ``optimizer: adam`` /
``scheduler: cosine`` entries to the right :class:`_Optimizer` /
:class:`_Schedulers` subclass.

Mirrors the kietzmann ``TrainingPipeline.setup_optimizer(name=...)`` /
``setup_scheduler(name=...)`` interface so yaml can stay declarative
and the trainer doesn't have to import every variant by hand.

Both registries are lowercase-keyed and tolerant of common aliases
(``"cosine_warm_restart"`` ↔ ``"cosine"``-with-restart). Adding a new
backend is a one-liner — add the import + a dict entry, the trainer
picks it up automatically.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from . import optimizers, schedulers


# --------------------------------------------------------------- optimizers

OPTIMIZER_REGISTRY: dict[str, type] = {
    "adam": optimizers.Adam,
    "adamw": optimizers.AdamW,
    "adamw_clip": optimizers.AdamWClipStyle,
    "sgd": optimizers.SGD,
    "rmsprop": optimizers.RMSprop,
    "rprop": optimizers.Rprop,
    "lars": optimizers.LARS,
}


def get_optimizer_factory(name: Optional[str], **kwargs) -> Callable[[Any], Any]:
    """Return a ``lambda params: optimizer(params)`` factory keyed by ``name``.

    The kietzmann ``_Optimizer`` classes wrap torch.optim and expose a
    ``forward(params)`` method that produces the actual optimizer. This
    helper composes that into the ``optim_factory`` shape every
    BaseModule expects (``params -> torch.optim.Optimizer``).

    ``name`` is case-insensitive. Pass any extra kwargs (``lr``,
    ``weight_decay``, ``momentum``, …) — they're forwarded to the
    optimizer's ``__init__``.
    """
    if name is None:
        # Sensible default — Adam with the trainer's learning_rate. Lets
        # yaml omit the optimizer: block entirely without breaking.
        opt_cls = optimizers.Adam
    else:
        key = name.lower()
        if key not in OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unknown optimizer {name!r}. "
                f"Available: {sorted(OPTIMIZER_REGISTRY)}"
            )
        opt_cls = OPTIMIZER_REGISTRY[key]
    opt = opt_cls(**kwargs)
    return lambda params: opt.forward(params)


# --------------------------------------------------------------- schedulers

SCHEDULER_REGISTRY: dict[str, type] = {
    "cosine": schedulers.CosineAnnealingScheduler,
    "cosine_annealing": schedulers.CosineAnnealingScheduler,
    "warm_cosine": schedulers.WarmCosineAnnealingLR,
    "cosine_warm": schedulers.WarmCosineAnnealingLR,
    "cosine_restart": schedulers.CosineScheduler,
    "exponential": schedulers.ExponentialLR,
    "multistep": schedulers.MultiStepLR,
    "plateau": schedulers.ReduceLROnPlateau,
    "reduce_on_plateau": schedulers.ReduceLROnPlateau,
    "linear": schedulers.LinearLR,
    "constant": schedulers.ConstantLR,
    "same": schedulers.SameLR,
    "none": schedulers.SameLR,
}


def get_scheduler_factory(
    name: Optional[str], **kwargs
) -> Optional[Callable[[Any], Any]]:
    """Return a ``lambda optimizer: lr_scheduler(optimizer)`` factory by name.

    Returns ``None`` when ``name`` is ``None`` / ``"none"`` so
    :meth:`BaseModule.configure_optimizers` can skip the
    ``lr_scheduler`` dict entirely (Lightning treats that as
    "no scheduler" and just returns the bare optimizer).

    ``kwargs`` is forwarded to the scheduler class — e.g.
    ``get_scheduler_factory("cosine", t_max=100, eta_min=1e-6)``.
    """
    if name is None or name.lower() == "none":
        return None
    key = name.lower()
    if key not in SCHEDULER_REGISTRY:
        raise ValueError(
            f"Unknown scheduler {name!r}. " f"Available: {sorted(SCHEDULER_REGISTRY)}"
        )
    sched = SCHEDULER_REGISTRY[key](**kwargs)
    return lambda optimizer: sched.forward(optimizer)


__all__ = [
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "get_optimizer_factory",
    "get_scheduler_factory",
]
