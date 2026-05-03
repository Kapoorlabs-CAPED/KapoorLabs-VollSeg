"""VollSeg — hierarchical, composable biological-image segmentation.

Public API:

- Layer 1 singletons: :class:`CAREDenoiser`, :class:`UNetSegmenter`,
  :class:`StarDistSegmenter`.
- Layer 2 composites: :class:`UNetStarDistPipeline`,
  :class:`DenoisedPipeline`, :class:`ROIPipeline`, :class:`Chunked`.
- Layer 3 factory: :class:`VollSeg.from_models`.
- Shared types: :class:`Pipeline`, :class:`Result`.
- Backbones (rarely needed directly): :mod:`vollseg._backbones`.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from ._backbones import (
    CAREBackbone,
    MaskUNetBackbone,
    StarDist2DBackbone,
    StarDist3DBackbone,
    UNetBackbone,
)
from .fusion import watershed_fuse
from .models import (
    CAREDenoiser,
    MaskUNetSegmenter,
    StarDistSegmenter,
    UNetSegmenter,
)
from .pipelines import (
    Chunked,
    DenoisedPipeline,
    Pipeline,
    ROIPipeline,
    Result,
    UNetStarDistPipeline,
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
    # Layer 1
    "CAREDenoiser",
    "UNetSegmenter",
    "MaskUNetSegmenter",
    "StarDistSegmenter",
    # Layer 2
    "UNetStarDistPipeline",
    "DenoisedPipeline",
    "ROIPipeline",
    "Chunked",
    # Layer 3
    "VollSeg",
    # shared
    "Pipeline",
    "Result",
    "watershed_fuse",
    "SeedPool",
    "UnetStarMask",
    # backbones
    "CAREBackbone",
    "UNetBackbone",
    "MaskUNetBackbone",
    "StarDist2DBackbone",
    "StarDist3DBackbone",
    # registry
    "register_model",
    "register_aliases",
    "get_registered_models",
    "clear_models_and_aliases",
]


# --- Pretrained Zenodo registry -------------------------------------------
# Same URLs and hashes as the original VollSeg, but keyed by the new
# backbone classes so existing notebooks can ``from_pretrained(...)``.
clear_models_and_aliases(
    StarDist2DBackbone, StarDist3DBackbone, UNetBackbone, MaskUNetBackbone, CAREBackbone
)

register_model(
    CAREBackbone,
    "Denoise_3D_cells",
    "https://zenodo.org/record/6671170/files/GenericDenoising3D.zip",
    "a0eb25ffd794e2b3b31a4de5b72a392f",
)
register_model(
    CAREBackbone,
    "Denoise_carcinoma",
    "https://zenodo.org/record/5910645/files/denoise_carcinoma.zip",
    "fd33199738f0b17761272118cbffdf04",
)
register_model(
    UNetBackbone,
    "Embryo Cell Model (3D)",
    "https://zenodo.org/record/6337699/files/embryo_cell_model.zip",
    "c84fdec38a5b3cc6c1869c94ff23f3ba",
)
register_model(
    UNetBackbone,
    "Xenopus Tissue (2D)",
    "https://zenodo.org/record/6060378/files/Xenopus_tissue_model.zip",
    "2694d8b05fa828aceb055eef8cd5ca1f",
)
register_model(
    StarDist2DBackbone,
    "White_Blood_Cells",
    "https://zenodo.org/record/5815521/files/WBCSeg.zip",
    "7889f5902d8562766a4dee2726c90d49",
)
register_model(
    StarDist3DBackbone,
    "Carcinoma_cells",
    "https://zenodo.org/record/6354077/files/carcinoma_stardist.zip",
    "b92b9d5347862e52279629be575fe0b7",
)
register_model(
    UNetBackbone,
    "Microtubule Kymograph Segmentation",
    "https://zenodo.org/record/6355705/files/microtubule_kymograph_segmentation.zip",
    "a42fcd4ba732734d36eda3dbbb3d5673",
)
register_model(
    UNetBackbone,
    "Unet_White_Blood_Cells",
    "https://zenodo.org/record/5815588/files/UNETWBC.zip",
    "9645f004db478f661811d6da615ccc0b",
)
register_model(
    UNetBackbone,
    "Unet_Arabidopsis",
    "https://zenodo.org/record/6670747/files/Unet_Arabidopsis.zip",
    "ed7bdead6ebb11c3e13c22a156288f60",
)
register_model(
    UNetBackbone,
    "Unet_Cyto_White_Blood_Cells",
    "https://zenodo.org/record/5815603/files/UNETcytoWBC.zip",
    "dd3bf8b8e2a04536144954e882445a5e",
)
register_model(
    UNetBackbone,
    "Unet_Lung_Segmentation",
    "https://zenodo.org/record/6060177/files/Montgomery_county.zip",
    "be41937a00693e28961358440d242417",
)
register_model(
    MaskUNetBackbone,
    "Xenopus_Cell_Tissue_Segmentation",
    "https://zenodo.org/record/6060378/files/Xenopus_tissue_model.zip",
    "2694d8b05fa828aceb055eef8cd5ca1f",
)
register_model(
    MaskUNetBackbone,
    "Unet_Arabidopsis_Mask",
    "https://zenodo.org/record/6670732/files/Unet_Arabidopsis_Mask.zip",
    "114df78e0153b39d80d0253a4dcc236f",
)

register_aliases(UNetBackbone, "Embryo Cell Model (3D)", "Embryo Cell Model (3D)")
register_aliases(StarDist2DBackbone, "White_Blood_Cells", "White_Blood_Cells")
register_aliases(StarDist3DBackbone, "Carcinoma_cells", "Carcinoma_cells")
register_aliases(UNetBackbone, "Unet_White_Blood_Cells", "Unet_White_Blood_Cells")
register_aliases(UNetBackbone, "Unet_Cyto_White_Blood_Cells", "Unet_Cyto_White_Blood_Cells")
register_aliases(
    UNetBackbone, "Microtubule Kymograph Segmentation", "Microtubule Kymograph Segmentation"
)
register_aliases(UNetBackbone, "Xenopus Tissue (2D)", "Xenopus Tissue (2D)")
register_aliases(UNetBackbone, "Unet_Lung_Segmentation", "Unet_Lung_Segmentation")
register_aliases(UNetBackbone, "Unet_Arabidopsis", "Unet_Arabidopsis")
register_aliases(
    MaskUNetBackbone, "Xenopus_Cell_Tissue_Segmentation", "Xenopus_Cell_Tissue_Segmentation"
)
register_aliases(MaskUNetBackbone, "Unet_Arabidopsis_Mask", "Unet_Arabidopsis_Mask")
register_aliases(CAREBackbone, "Denoise_3D_cells", "Denoise_3D_cells")
