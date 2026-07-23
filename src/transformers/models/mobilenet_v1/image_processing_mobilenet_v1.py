
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
)
from ...utils import auto_docstring


@auto_docstring(custom_intro="Constructs a MobileNetV1 image processor.")
class MobileNetV1ImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 256}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = None


__all__ = ["MobileNetV1ImageProcessor"]
