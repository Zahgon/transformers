

import numpy as np
import torch
from torch.nn import functional as F

from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    PILImageResampling,
    SizeDict,
)
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, is_torchvision_available, logging
from ...video_processing_utils import BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_MESSAGE = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
DEFAULT_VIDEO_INTRO = (
    "You are provided the following series of {frame_count} frames from a {video_duration} [H:MM:SS] video.\n"
)
DEFAULT_MEDIA_OUTTRO = "\n\n"
FRAME_TIMESTAMP_MESSAGE = "\nFrame from {timestamp}:"
MAX_IMAGE_SIZE = 4096  # 4k resolution as absolute maximum


def get_max_height_width(videos: list["torch.Tensor"]) -> list[int]:
    """
    Get the maximum height and width across all videos in a batch.
    """
    max_height = max_width = float("-inf")
    for video in videos:
        height, width = video.size()[-2:]
        max_height = max(height, max_height)
        max_width = max(width, max_width)
    return (max_height, max_width)


def get_resize_output_image_size(
    video,
    resolution_max_side: int,
) -> tuple[int, int]:
    """
    Get the output size of the video after resizing given a dictionary specifying the max and min sizes.
    Args:
        video (`np.ndarray`):
            Video to resize.
        resolution_max_side (`int`):
            The longest edge of the video will be resized to this value. The shortest edge will be resized to keep the
            input aspect ratio.
    Returns:
        The output size of the video after resizing.
    """
    height, width = video.size()[-2:]

    resolution_max_side = min(MAX_IMAGE_SIZE, resolution_max_side)
    resolution_max_side = max(height, width) if resolution_max_side is None else resolution_max_side
    aspect_ratio = width / height

    if width >= height:
        width = resolution_max_side
        height = int(width / aspect_ratio)
        if height % 2 != 0:
            height += 1
    elif height > width:
        height = resolution_max_side
        width = int(height * aspect_ratio)
        if width % 2 != 0:
            width += 1

    height = max(height, 1)
    width = max(width, 1)

    return height, width


class SmolVLMVideoProcessorInitKwargs(VideosKwargs, total=False):
    max_image_size: dict[str, int]


class SmolVLMVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.LANCZOS
    size = {"longest_edge": 4 * 364}
    max_image_size = {"longest_edge": 364}
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    do_pad = True
    do_sample_frames = False  # Set to False for BC, recommended to set `True` in new models
    valid_kwargs = SmolVLMVideoProcessorInitKwargs
    model_input_names = ["pixel_values", "pixel_attention_mask"]

    def __init__(self, **kwargs: Unpack[SmolVLMVideoProcessorInitKwargs]):
        super().__init__(**kwargs)
        if "size" in kwargs and "video_sampling" in kwargs:
            kwargs["video_sampling"]["video_size"] = kwargs["size"]

        if "video_sampling" in kwargs:
            self.num_frames = kwargs["video_sampling"]["max_frames"]
            self.fps = kwargs["video_sampling"]["fps"]
            self.size = get_size_dict(kwargs["video_sampling"]["video_size"], default_to_square=self.default_to_square)

    def resize(
        self,
        video: "torch.Tensor",
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        antialias: bool = True,
        **kwargs,
    ) -> "torch.Tensor":
        """
        Resize an video to `(size["height"], size["width"])`.
        Args:
            video (`torch.Tensor`):
                Video to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output video.
            resample (`PILImageResampling` or `InterpolationMode`, *optional*, defaults to `InterpolationMode.BILINEAR`):
                Resampling filter to use when resizing the video.
        Returns:
            `torch.Tensor`: The resized video.
        """
        if size.longest_edge:
            new_size = get_resize_output_image_size(
                video,
                resolution_max_side=size.longest_edge,
            )
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(f"Size must contain 'height' and 'width' keys, or 'longest_edge' key. Got {size}.")

        video = super().resize(
            video, SizeDict(height=new_size[0], width=new_size[1]), resample=resample, antialias=antialias
        )

        max_size = SizeDict(height=self.max_image_size["longest_edge"], width=self.max_image_size["longest_edge"])
        video = super().resize(video, max_size, resample=resample, antialias=antialias)
        return video

    def pad(
        self,
        video: "torch.Tensor",
        padded_size: tuple[int, int],
        max_num_frames: int,
        fill: int = 0,
        return_pixel_mask: bool = True,
    ):
        """Pads the sample with empty video to the padded_size
        Args:
            video (`torch.Tensor`):
                Batched video to pad.
            padded_size (`tuple[int, int]`):
                Height and width to pad.
            max_num_frames (`int`):
                The maximum number of frames to which video will be padded.
            fill (`int`, *optional*):
                The value to use for the padding.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return a pixel mask.
        """
        original_size = video.size()[-2:]
        num_frames = video.shape[1] if video.ndim == 5 else video.shape[0]
        padding_height = padded_size[0] - original_size[0]
        padding_width = padded_size[1] - original_size[1]
        padding_frame = max_num_frames - num_frames
        if padding_width < 0 or padding_height < 0 or padding_frame < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded max number of frames {max_num_frames} and padded size: {padded_size}, "
                f"original number of frames {num_frames} and size: {original_size}."
            )
        if original_size != padded_size or padding_frame > 0:
            padding = [0, padding_width, 0, padding_height, 0, 0, 0, padding_frame]
            video = F.pad(video, padding, value=fill)

        pixel_mask = None
        if return_pixel_mask:
            pixel_mask = torch.zeros_like(video[..., 0, :, :], dtype=torch.int64)
            pixel_mask[..., : original_size[0], : original_size[1]] = 1

        return video, pixel_mask

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        skip_secs: int | None = 1,
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
        do_pad: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
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
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_videos_grouped[shape] = stacked_videos

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)

        if do_pad:
            pad_size = get_max_height_width(processed_videos)
            max_num_frames = max(len(video) for video in processed_videos)
            grouped_videos, grouped_videos_index = group_videos_by_shape(processed_videos)
            processed_padded_mask_grouped = {}
            processed_videos_grouped = {}

            for shape, stacked_videos in grouped_videos.items():
                stacked_videos, padded_masks = self.pad(
                    stacked_videos, padded_size=pad_size, max_num_frames=max_num_frames
                )
                processed_videos_grouped[shape] = stacked_videos
                processed_padded_mask_grouped[shape] = padded_masks

            processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
            pixel_attention_mask = reorder_videos(processed_padded_mask_grouped, grouped_videos_index)

        data = {"pixel_values": processed_videos}

        if do_pad:
            data["pixel_attention_mask"] = (
                torch.stack(pixel_attention_mask, dim=0)
                if do_pad and return_tensors is not None
                else pixel_attention_mask
            )
        return BatchFeature(data, tensor_type=return_tensors)


__all__ = ["SmolVLMVideoProcessor"]
