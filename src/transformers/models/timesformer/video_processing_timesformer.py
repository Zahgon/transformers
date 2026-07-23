
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...video_processing_utils import BaseVideoProcessor


class TimesformerVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 224}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False
    model_input_names = ["pixel_values"]

    def preprocess(self, videos, **kwargs):
        batch = super().preprocess(videos, **kwargs)
        batch["pixel_values"] = batch.pop("pixel_values_videos")
        return batch


__all__ = ["TimesformerVideoProcessor"]
