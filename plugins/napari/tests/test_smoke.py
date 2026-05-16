"""Import-level + catalog + RoleChoice smoke tests."""

from __future__ import annotations


def test_imports():
    from kapoorlabs_vollseg_napari import MODEL_CATALOG, VollSegWidget  # noqa: F401


def test_model_catalog_has_expected_roles():
    from kapoorlabs_vollseg_napari import MODEL_CATALOG

    for role in ("care", "unet", "maskunet", "stardist", "cellpose"):
        assert role in MODEL_CATALOG, role


def test_every_registry_model_classified():
    from kapoorlabs_vollseg.hub import XENOPUS_MODELS

    from kapoorlabs_vollseg_napari import MODEL_CATALOG

    classified = {name for bucket in MODEL_CATALOG.values() for name in bucket}
    assert set(XENOPUS_MODELS.keys()) == classified


def test_role_choice_defaults():
    from kapoorlabs_vollseg_napari._runner import RoleChoice

    rc = RoleChoice()
    assert rc.mode == "none"
    assert not rc.active

    rc = RoleChoice(mode="pretrained", pretrained_name="nuclei_xenopus_mari")
    assert rc.active and rc.is_pretrained and not rc.is_custom

    rc = RoleChoice(mode="custom", custom_path="/tmp/whatever")
    assert rc.active and rc.is_custom and not rc.is_pretrained


def test_runspec_default_construction():
    import numpy as np

    from kapoorlabs_vollseg_napari._runner import RunSpec

    spec = RunSpec(image=np.zeros((4, 8, 8), dtype=np.float32))
    assert spec.care.mode == "none"
    assert spec.stardist.mode == "none"
    assert spec.membrane_mode is False
    assert spec.model_dir.name == "kapoorlabs-vollseg"
