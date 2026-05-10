from stardist.models import StarDist2D as _StarDist2D
from stardist.models import StarDist3D as _StarDist3D


class StarDist2DBackboneKeras(_StarDist2D):
    """StarDist 2D — keras-backed. The bare-named PyTorch counterpart is planned."""

    def __init__(self, config, name=None, basedir="."):
        super().__init__(config=config, name=name, basedir=basedir)


class StarDist3DBackboneKeras(_StarDist3D):
    """StarDist 3D — keras-backed. The bare-named PyTorch counterpart is planned."""

    def __init__(self, config, name=None, basedir="."):
        super().__init__(config=config, name=name, basedir=basedir)
