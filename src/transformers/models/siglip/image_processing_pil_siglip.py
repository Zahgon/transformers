
from ...image_processing_backends import PilBackend
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
)
from ...utils import auto_docstring


@auto_docstring(custom_intro="Constructs a SigLIP image processor.")
class SiglipImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 224, "width": 224}
    default_to_square = False
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True


__all__ = ["SiglipImageProcessorPil"]
