
from ...image_processing_backends import PilBackend
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...utils import auto_docstring


@auto_docstring
class DeiTImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 256, "width": 256}
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True


__all__ = ["DeiTImageProcessorPil"]
