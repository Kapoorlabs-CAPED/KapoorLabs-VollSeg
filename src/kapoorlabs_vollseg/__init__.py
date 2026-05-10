"""VollSeg — hierarchical, composable biological-image segmentation.

Two backends coexist:

- **PyTorch + Lightning + careamics** is the first-class backend. Bare
  names (``CAREDenoiser``, ``UNetSegmenter``, ``MaskUNetSegmenter``,
  ``StarDistSegmenter``, ``CellPoseSegmenter``) point here and are
  always available with the default install
  (``pip install kapoorlabs-vollseg``).
- **csbdeep / stardist (keras)** is **optional**. Names carry a
  ``Keras`` suffix (``CAREDenoiserKeras``, ``UNetSegmenterKeras``,
  ``MaskUNetSegmenterKeras``, ``StarDistSegmenterKeras``). Install with
  ``pip install kapoorlabs-vollseg[keras]``. If the optional extras
  aren't installed, those names are simply not exposed — the package
  still imports cleanly without tensorflow on disk.

Both kinds of singletons satisfy the :class:`Pipeline` protocol, so
Layer 2 composites (``DenoisedPipeline`` etc.) and Layer 3 factories
(``VollSeg.from_models``, ``VollCellSeg.from_models``) compose either
or both seamlessly.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

# --- PyTorch first-class — always available ------------------------------
from ._backbones import (
    CAREBackbone,
    CellPoseBackbone,
    MaskUNetBackbone,
    StarDistBackbone,
    UNetBackbone,
)
from .fusion import cellpose_watershed_fuse, watershed_fuse
from .hub import XENOPUS_MODELS, ensure_cellpose_checkpoint, ensure_model, hf_repo_for
from .models import (
    CAREDenoiser,
    CellPoseSegmenter,
    MaskUNetSegmenter,
    StarDistSegmenter,
    UNetSegmenter,
)
from .pipelines import (
    Chunked,
    DenoisedPipeline,
    NucleiSeededCellPosePipeline,
    Pipeline,
    ROIPipeline,
    Result,
    UNetStarDistPipeline,
    VollCellSeg,
    VollSeg,
)
from .pretrained import (
    clear_models_and_aliases,
    get_registered_models,
    register_aliases,
    register_model,
)
from .seedpool import SeedPool, UnetStarMask

__all__ = [
    # Layer 1 — PyTorch first-class
    "CAREDenoiser",
    "UNetSegmenter",
    "MaskUNetSegmenter",
    "StarDistSegmenter",
    "CellPoseSegmenter",
    # Layer 2
    "UNetStarDistPipeline",
    "NucleiSeededCellPosePipeline",
    "DenoisedPipeline",
    "ROIPipeline",
    "Chunked",
    # Layer 3
    "VollSeg",
    "VollCellSeg",
    # shared
    "Pipeline",
    "Result",
    "watershed_fuse",
    "cellpose_watershed_fuse",
    "SeedPool",
    "UnetStarMask",
    # backbones — PyTorch
    "CAREBackbone",
    "UNetBackbone",
    "MaskUNetBackbone",
    "StarDistBackbone",
    "CellPoseBackbone",
    # registry
    "register_model",
    "register_aliases",
    "get_registered_models",
    "clear_models_and_aliases",
    # HuggingFace hub
    "XENOPUS_MODELS",
    "ensure_model",
    "ensure_cellpose_checkpoint",
    "hf_repo_for",
]


# --- Optional keras backend -----------------------------------------------
# Loaded only when `pip install kapoorlabs-vollseg[keras]` has provided
# csbdeep and stardist. On a PyTorch-only install these classes are simply
# absent from the module — `from kapoorlabs_vollseg import CAREDenoiserKeras`
# will fail with a clean AttributeError instead of dragging in tensorflow.

HAS_KERAS = False
try:
    from ._backbones import (
        CAREBackboneKeras,
        MaskUNetBackboneKeras,
        StarDist2DBackboneKeras,
        StarDist3DBackboneKeras,
        UNetBackboneKeras,
    )

    HAS_KERAS = True
    __all__.extend(
        [
            "CAREDenoiserKeras",
            "UNetSegmenterKeras",
            "MaskUNetSegmenterKeras",
            "StarDistSegmenterKeras",
            "CAREBackboneKeras",
            "UNetBackboneKeras",
            "MaskUNetBackboneKeras",
            "StarDist2DBackboneKeras",
            "StarDist3DBackboneKeras",
        ]
    )
except ImportError:
    pass


# --- Pretrained Zenodo registry ------------------------------------------
# Existing pretrained weights are csbdeep / stardist (keras) checkpoints.
# Skip registration entirely on PyTorch-only installs — the registry is
# only useful if the loader (`*Keras` classes) is also available.

if HAS_KERAS:
    clear_models_and_aliases(
        StarDist2DBackboneKeras,
        StarDist3DBackboneKeras,
        UNetBackboneKeras,
        MaskUNetBackboneKeras,
        CAREBackboneKeras,
    )

    register_model(
        CAREBackboneKeras,
        "Denoise_3D_cells",
        "https://zenodo.org/record/6671170/files/GenericDenoising3D.zip",
        "a0eb25ffd794e2b3b31a4de5b72a392f",
    )
    register_model(
        CAREBackboneKeras,
        "Denoise_carcinoma",
        "https://zenodo.org/record/5910645/files/denoise_carcinoma.zip",
        "fd33199738f0b17761272118cbffdf04",
    )
    register_model(
        UNetBackboneKeras,
        "Embryo Cell Model (3D)",
        "https://zenodo.org/record/6337699/files/embryo_cell_model.zip",
        "c84fdec38a5b3cc6c1869c94ff23f3ba",
    )
    register_model(
        UNetBackboneKeras,
        "Xenopus Tissue (2D)",
        "https://zenodo.org/record/6060378/files/Xenopus_tissue_model.zip",
        "2694d8b05fa828aceb055eef8cd5ca1f",
    )
    register_model(
        StarDist2DBackboneKeras,
        "White_Blood_Cells",
        "https://zenodo.org/record/5815521/files/WBCSeg.zip",
        "7889f5902d8562766a4dee2726c90d49",
    )
    register_model(
        StarDist3DBackboneKeras,
        "Carcinoma_cells",
        "https://zenodo.org/record/6354077/files/carcinoma_stardist.zip",
        "b92b9d5347862e52279629be575fe0b7",
    )
    register_model(
        UNetBackboneKeras,
        "Microtubule Kymograph Segmentation",
        "https://zenodo.org/record/6355705/files/microtubule_kymograph_segmentation.zip",
        "a42fcd4ba732734d36eda3dbbb3d5673",
    )
    register_model(
        UNetBackboneKeras,
        "Unet_White_Blood_Cells",
        "https://zenodo.org/record/5815588/files/UNETWBC.zip",
        "9645f004db478f661811d6da615ccc0b",
    )
    register_model(
        UNetBackboneKeras,
        "Unet_Arabidopsis",
        "https://zenodo.org/record/6670747/files/Unet_Arabidopsis.zip",
        "ed7bdead6ebb11c3e13c22a156288f60",
    )
    register_model(
        UNetBackboneKeras,
        "Unet_Cyto_White_Blood_Cells",
        "https://zenodo.org/record/5815603/files/UNETcytoWBC.zip",
        "dd3bf8b8e2a04536144954e882445a5e",
    )
    register_model(
        UNetBackboneKeras,
        "Unet_Lung_Segmentation",
        "https://zenodo.org/record/6060177/files/Montgomery_county.zip",
        "be41937a00693e28961358440d242417",
    )
    register_model(
        MaskUNetBackboneKeras,
        "Xenopus_Cell_Tissue_Segmentation",
        "https://zenodo.org/record/6060378/files/Xenopus_tissue_model.zip",
        "2694d8b05fa828aceb055eef8cd5ca1f",
    )
    register_model(
        MaskUNetBackboneKeras,
        "Unet_Arabidopsis_Mask",
        "https://zenodo.org/record/6670732/files/Unet_Arabidopsis_Mask.zip",
        "114df78e0153b39d80d0253a4dcc236f",
    )

    register_aliases(
        UNetBackboneKeras, "Embryo Cell Model (3D)", "Embryo Cell Model (3D)"
    )
    register_aliases(StarDist2DBackboneKeras, "White_Blood_Cells", "White_Blood_Cells")
    register_aliases(StarDist3DBackboneKeras, "Carcinoma_cells", "Carcinoma_cells")
    register_aliases(
        UNetBackboneKeras, "Unet_White_Blood_Cells", "Unet_White_Blood_Cells"
    )
    register_aliases(
        UNetBackboneKeras, "Unet_Cyto_White_Blood_Cells", "Unet_Cyto_White_Blood_Cells"
    )
    register_aliases(
        UNetBackboneKeras,
        "Microtubule Kymograph Segmentation",
        "Microtubule Kymograph Segmentation",
    )
    register_aliases(UNetBackboneKeras, "Xenopus Tissue (2D)", "Xenopus Tissue (2D)")
    register_aliases(
        UNetBackboneKeras, "Unet_Lung_Segmentation", "Unet_Lung_Segmentation"
    )
    register_aliases(UNetBackboneKeras, "Unet_Arabidopsis", "Unet_Arabidopsis")
    register_aliases(
        MaskUNetBackboneKeras,
        "Xenopus_Cell_Tissue_Segmentation",
        "Xenopus_Cell_Tissue_Segmentation",
    )
    register_aliases(
        MaskUNetBackboneKeras, "Unet_Arabidopsis_Mask", "Unet_Arabidopsis_Mask"
    )
    register_aliases(CAREBackboneKeras, "Denoise_3D_cells", "Denoise_3D_cells")
