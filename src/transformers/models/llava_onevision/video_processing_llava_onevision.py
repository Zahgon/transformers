
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...video_processing_utils import BaseVideoProcessor


class LlavaOnevisionVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    rescale_factor = 1 / 255
    default_to_square = False
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models


__all__ = ["LlavaOnevisionVideoProcessor"]
