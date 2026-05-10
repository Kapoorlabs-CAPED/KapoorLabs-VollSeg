from csbdeep.models import CARE as _CSBDeepCARE


class MaskUNetBackboneKeras(_CSBDeepCARE):
    """U-Net variant producing both segmentation + mask outputs — keras backend.

    Legacy backbone. New code should use :class:`kapoorlabs_vollseg.MaskUNetBackbone`.
    """

    def __init__(self, config, name=None, basedir="."):
        super(_CSBDeepCARE, self).__init__(config=config, name=name, basedir=basedir)
