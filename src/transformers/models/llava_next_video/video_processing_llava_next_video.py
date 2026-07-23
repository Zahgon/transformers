
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...video_processing_utils import BaseVideoProcessor


class LlavaNextVideoVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models


__all__ = ["LlavaNextVideoVideoProcessor"]
