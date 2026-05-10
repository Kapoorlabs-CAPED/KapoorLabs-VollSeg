from csbdeep.models import CARE as _CSBDeepCARE


class CAREBackboneKeras(_CSBDeepCARE):
    """csbdeep CARE network — keras-backed weight container for denoising.

    Legacy backbone. New code should use :class:`kapoorlabs_vollseg.CAREBackbone`
    (PyTorch + careamics UNet via Lightning).
    """

    def __init__(self, config, name=None, basedir="."):
        super().__init__(config=config, name=name, basedir=basedir)
