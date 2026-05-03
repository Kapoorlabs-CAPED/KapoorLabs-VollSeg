from csbdeep.models import CARE as _CSBDeepCARE


class UNetBackbone(_CSBDeepCARE):
    """U-Net for semantic segmentation, sharing the CARE training infrastructure."""

    def __init__(self, config, name=None, basedir="."):
        # Skip the CARE subclass __init__; bind directly to csbdeep CARE.
        super(_CSBDeepCARE, self).__init__(config=config, name=name, basedir=basedir)
