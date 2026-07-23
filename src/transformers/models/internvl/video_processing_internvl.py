
import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling, SizeDict
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


class InternVLVideoProcessorInitKwargs(VideosKwargs, total=False):
    initial_shift: bool | float | int


class InternVLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 384, "width": 384}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    initial_shift = True
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    valid_kwargs = InternVLVideoProcessorInitKwargs

    def __init__(self, **kwargs: Unpack[InternVLVideoProcessorInitKwargs]):
        super().__init__(**kwargs)

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        initial_shift: bool | float | int | None = None,
        **kwargs,
    ):
        pass

    def _preprocess(
        self,
        videos: list["torch.Tensor"],
        do_convert_rgb: bool,
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            if do_resize:
                stacked_videos = self.resize(stacked_videos, size=size, resample=resample)
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_center_crop:
                stacked_videos = self.center_crop(stacked_videos, crop_size)
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)

        return BatchFeature(data={"pixel_values_videos": processed_videos}, tensor_type=return_tensors)


__all__ = ["InternVLVideoProcessor"]
