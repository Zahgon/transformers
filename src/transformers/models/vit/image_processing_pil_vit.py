
from ...image_processing_backends import PilBackend
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling


class ViTImageProcessorPil(PilBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    do_resize = True
    do_rescale = True
    do_normalize = True


__all__ = ["ViTImageProcessorPil"]
