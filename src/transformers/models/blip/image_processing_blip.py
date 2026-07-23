
from ...image_processing_backends import TorchvisionBackend
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...utils import auto_docstring


@auto_docstring
class BlipImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    default_to_square = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True


__all__ = ["BlipImageProcessor"]
