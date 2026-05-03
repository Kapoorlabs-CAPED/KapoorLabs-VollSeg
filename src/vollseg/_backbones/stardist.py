from stardist.models import StarDist2D as _StarDist2D
from stardist.models import StarDist3D as _StarDist3D


class StarDist2DBackbone(_StarDist2D):
    def __init__(self, config, name=None, basedir="."):
        super().__init__(config=config, name=name, basedir=basedir)


class StarDist3DBackbone(_StarDist3D):
    def __init__(self, config, name=None, basedir="."):
        super().__init__(config=config, name=name, basedir=basedir)
