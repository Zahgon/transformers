
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...utils import auto_docstring


@auto_docstring
class PvtImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 224, "width": 224}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True


__all__ = ["PvtImageProcessor"]
