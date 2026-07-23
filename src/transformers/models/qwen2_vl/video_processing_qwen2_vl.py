
import math

import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_utils import BatchFeature
from ...image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    PILImageResampling,
    SizeDict,
    get_image_size,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos
from .image_processing_qwen2_vl import smart_resize


class Qwen2VLVideoProcessorInitKwargs(VideosKwargs, total=False):
    min_pixels: int
    max_pixels: int
    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int


@add_start_docstrings(
    "Constructs a fast Qwen2-VL image processor that dynamically resizes videos based on the original videos.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 2):
            The temporal patch size of the vision encoder.
        merge_size (`int`, *optional*, defaults to 2):
            The merge size of the vision encoder to llm encoder.
        min_frames (`int`, *optional*, defaults to 4):
            The minimum number of frames that can be sampled.
        max_frames (`int`, *optional*, defaults to 768):
            The maximum number of frames that can be sampled.
    """,
)
class Qwen2VLVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 128 * 28 * 28, "longest_edge": 28 * 28 * 768}
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    patch_size = 14
    temporal_patch_size = 2
    merge_size = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    valid_kwargs = Qwen2VLVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Qwen2VLVideoProcessorInitKwargs]):
        size = kwargs.pop("size", None)
        min_pixels = kwargs.pop("min_pixels", None)
        max_pixels = kwargs.pop("max_pixels", None)
        size = self.size if size is None else size
        if min_pixels is not None:
            size["shortest_edge"] = min_pixels
            size.pop("min_pixels", None)
        if max_pixels is not None:
            size["longest_edge"] = max_pixels
            size.pop("max_pixels", None)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")

        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(
        self,
        size: SizeDict | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        **kwargs,
    ) -> dict:
        if min_pixels is not None and max_pixels is not None:
            size = SizeDict(shortest_edge=min_pixels, longest_edge=max_pixels)
        kwargs = super()._standardize_kwargs(size=size, **kwargs)
        size = kwargs.get("size", self.size)
        if not size.shortest_edge or not size.longest_edge:
            raise ValueError("size must contain 'shortest_edge' and 'longest_edge' keys.")
        return kwargs

    def sample_frames(
        self,
        metadata: VideoMetadata,
        temporal_patch_size: int | None = None,
        min_frames: int | None = None,
        max_frames: int | None = None,
        num_frames: int | None = None,
        fps: int | float | None = None,
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
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}
        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            height, width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            resized_height, resized_width = height, width
            if do_resize:
                resized_height, resized_width = smart_resize(
                    height,
                    width,
                    factor=patch_size * merge_size,
                    min_pixels=size["shortest_edge"],
                    max_pixels=size["longest_edge"],
                )
                stacked_videos = self.resize(
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)

            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            patches = stacked_videos

            T = patches.shape[1]
            if pad := -T % temporal_patch_size:
                repeats = patches[:, -1:].expand(-1, pad, -1, -1, -1)
                patches = torch.cat((patches, repeats), dim=1)

            batch_size, grid_t, channel = patches.shape[:3]
            grid_t = grid_t // temporal_patch_size
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size

            patches = patches.view(
                batch_size,
                grid_t,
                temporal_patch_size,
                channel,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                channel * temporal_patch_size * patch_size * patch_size,
            )

            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        pixel_values_videos = torch.cat(processed_videos, dim=0)
        video_grid_thw = torch.tensor(processed_grids)

        return BatchFeature(
            data={"pixel_values_videos": pixel_values_videos, "video_grid_thw": video_grid_thw},
            tensor_type=return_tensors,
        )

    def get_num_of_video_patches(self, num_frames: int, height: int, width: int, videos_kwargs=None):
        pass


__all__ = ["Qwen2VLVideoProcessor"]
