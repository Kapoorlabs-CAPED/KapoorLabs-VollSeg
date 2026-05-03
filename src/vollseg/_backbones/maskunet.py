from csbdeep.models import CARE as _CSBDeepCARE


class MaskUNetBackbone(_CSBDeepCARE):
    """U-Net variant whose final layer produces both segmentation + mask outputs.

    Same csbdeep CARE base as :class:`UNetBackbone`; kept as a separate
    class so that pretrained weights register under their own key.
    """

    def __init__(self, config, name=None, basedir="."):
        super(_CSBDeepCARE, self).__init__(config=config, name=name, basedir=basedir)
