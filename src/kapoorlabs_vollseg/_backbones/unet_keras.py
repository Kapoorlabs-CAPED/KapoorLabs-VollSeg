from csbdeep.models import CARE as _CSBDeepCARE


class UNetBackboneKeras(_CSBDeepCARE):
    """U-Net for semantic segmentation — keras (csbdeep CARE infrastructure).

    Legacy backbone. New code should use :class:`kapoorlabs_vollseg.UNetBackbone`
    (PyTorch + careamics UNet via Lightning).
    """

    def __init__(self, config, name=None, basedir="."):
        super(_CSBDeepCARE, self).__init__(config=config, name=name, basedir=basedir)
