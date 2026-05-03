from csbdeep.models import CARE as _CSBDeepCARE


class CAREBackbone(_CSBDeepCARE):
    """csbdeep CARE network — used as the weight container for denoising."""

    def __init__(self, config, name=None, basedir="."):
        super().__init__(config=config, name=name, basedir=basedir)
