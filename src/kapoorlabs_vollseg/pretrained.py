"""Pretrained model registry — Zenodo-hosted weights keyed by class + name/alias.

Ported from the original VollSeg unchanged in behavior; this is a stable
contract with downstream notebooks.
"""

from collections import OrderedDict
from warnings import warn

from csbdeep.utils import _raise
from csbdeep.utils.six import Path
from csbdeep.utils.tf import keras_import

get_file = keras_import("utils", "get_file")


_MODELS: dict = {}
_ALIASES: dict = {}


def clear_models_and_aliases(*cls):
    if not cls:
        _MODELS.clear()
        _ALIASES.clear()
        return
    for c in cls:
        _MODELS.pop(c, None)
        _ALIASES.pop(c, None)


def register_model(cls, key, url, hash):
    models = _MODELS.setdefault(cls, OrderedDict())
    if key in models:
        warn(
            f"re-registering model '{key}' (was already registered for '{cls.__name__}')"
        )
    models[key] = dict(url=url, hash=hash)


def register_aliases(cls, key, *names):
    if not names:
        return
    models = _MODELS.get(cls, {})
    key in models or _raise(
        ValueError(f"model '{key}' is not registered for '{cls.__name__}'")
    )
    aliases = _ALIASES.setdefault(cls, OrderedDict())
    for name in names:
        if aliases.get(name, key) != key:
            warn(
                f"alias '{name}' was previously registered with model "
                f"'{aliases[name]}' for '{cls.__name__}'"
            )
        aliases[name] = key


def get_registered_models(cls, return_aliases=True, verbose=False):
    models = _MODELS.get(cls, {})
    aliases = _ALIASES.get(cls, {})
    model_keys = tuple(models.keys())
    model_aliases = {
        key: tuple(name for name in aliases if aliases[name] == key) for key in models
    }
    if verbose:
        n = len(models)
        print(
            f"There {'is' if n == 1 else 'are'} {n} registered "
            f"{'model' if n == 1 else 'models'} for '{cls.__name__}'"
            f"{':' if n > 0 else ''}"
        )
        if n > 0:
            maxkeylen = 2 + max(len(k) for k in models)
            header = f"Name{' ' * (maxkeylen - 4 + 3)}Alias(es)"
            print()
            print(header)
            print("-" * len(header))
            for key in models:
                aliases_str = "   "
                m = len(model_aliases[key])
                aliases_str += (
                    "'%s'" % "', '".join(model_aliases[key]) if m > 0 else "None"
                )
                print(("{s:%d}" % maxkeylen).format(s="'%s'" % key) + aliases_str)
    return (model_keys, model_aliases) if return_aliases else model_keys


def get_model_details(cls, key_or_alias, verbose=False):
    models = _MODELS.get(cls, {})
    if key_or_alias in models:
        key, alias = key_or_alias, None
    else:
        aliases = _ALIASES.get(cls, {})
        alias = key_or_alias
        alias in aliases or _raise(
            ValueError(f"'{alias}' is neither a key or alias for '{cls.__name__}'")
        )
        key = aliases[alias]
    if verbose:
        suffix = "" if alias is None else f" with alias '{alias}'"
        print(f"Found model '{key}'{suffix} for '{cls.__name__}'.")
    return key, alias, models[key]


def get_model_folder(cls, key_or_alias):
    key, _alias, m = get_model_details(cls, key_or_alias)
    target = str(Path("models") / cls.__name__ / key)
    path = Path(
        get_file(
            fname=key + ".zip",
            origin=m["url"],
            file_hash=m["hash"],
            cache_subdir=target,
            extract=True,
        )
    )
    assert path.exists() and path.parent.exists()
    return path.parent


def get_model_instance(cls, key_or_alias):
    path = get_model_folder(cls, key_or_alias)
    model = cls(config=None, name=path.stem, basedir=path.parent)
    model.basedir = None  # read-only
    return model
