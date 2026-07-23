
from ...image_utils import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD, PILImageResampling
from ...video_processing_utils import BaseVideoProcessor


class VivitVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"shortest_edge": 256}
    default_to_square = False
    crop_size = {"height": 224, "width": 224}
    do_resize = True
    do_center_crop = True
    do_rescale = True
    rescale_factor = 1 / 127.5
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False
    model_input_names = ["pixel_values"]

    def rescale_and_normalize(self, images, do_rescale, rescale_factor, do_normalize, image_mean, image_std):
        if do_rescale:
            images = images.float() * rescale_factor - 1
        if do_normalize:
            images = self.normalize(images.float(), image_mean, image_std)
        return images

    def preprocess(self, videos, **kwargs):
        batch = super().preprocess(videos, **kwargs)
        batch["pixel_values"] = batch.pop("pixel_values_videos")
        return batch


__all__ = ["VivitVideoProcessor"]
