
import math

import numpy as np
import torch

from ...image_processing_utils import BatchFeature
from ...image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from ...processing_utils import Unpack, VideosKwargs
from ...utils import TensorType, add_start_docstrings, is_torchvision_available, logging
from ...video_processing_utils import BASE_VIDEO_PROCESSOR_DOCSTRING, BaseVideoProcessor
from ...video_utils import VideoMetadata, group_videos_by_shape, reorder_videos


if is_torchvision_available():
    from torchvision.transforms.v2 import functional as tvF


logger = logging.get_logger(__name__)


class Cosmos3EdgeVideoProcessorInitKwargs(VideosKwargs, total=False):

    patch_size: int
    temporal_patch_size: int
    merge_size: int
    min_frames: int
    max_frames: int


def smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 1,
    factor: int = 32,
    min_pixels: int = 64 * 64,
    max_pixels: int = 24 * 1024 * 1024,
) -> tuple[int, int]:
    """Resize video frames while keeping the packed patch grid valid for Cosmos3 Edge.

    This follows Qwen3-VL's video resize function. Cosmos3 Edge changes the defaults because it processes every
    sampled frame independently (`temporal_factor=1`) and uses the checkpoint's pixel budget.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor

    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


@add_start_docstrings(
    "Constructs a video processor that dynamically resizes and packs Cosmos3 Edge video frames.",
    BASE_VIDEO_PROCESSOR_DOCSTRING,
    """
        patch_size (`int`, *optional*, defaults to 16):
            Spatial patch size of the vision encoder.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            Temporal patch size of the vision encoder. Cosmos3 Edge processes each sampled frame independently.
        merge_size (`int`, *optional*, defaults to 2):
            Spatial merge size applied by the vision projector.
    """,
)
class Cosmos3EdgeVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BICUBIC
    size = {"shortest_edge": 64 * 64, "longest_edge": 24 * 1024 * 1024}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    patch_size = 16
    temporal_patch_size = 1
    merge_size = 2
    fps = 2
    min_frames = 4
    max_frames = 768
    do_sample_frames = True
    valid_kwargs = Cosmos3EdgeVideoProcessorInitKwargs
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Unpack[Cosmos3EdgeVideoProcessorInitKwargs]):
        size = kwargs.pop("size", None)
        size = dict(self.size) if size is None else dict(size)
        if "shortest_edge" not in size or "longest_edge" not in size:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        if kwargs.get("temporal_patch_size", self.temporal_patch_size) != 1:
            raise ValueError("Cosmos3 Edge only supports `temporal_patch_size=1`.")
        super().__init__(size=size, **kwargs)

    def _standardize_kwargs(self, **kwargs) -> dict:
        kwargs = super()._standardize_kwargs(**kwargs)
        size = kwargs.get("size", self.size)
        if size.shortest_edge is None or size.longest_edge is None:
            raise ValueError("`size` must contain `shortest_edge` and `longest_edge` keys.")
        if kwargs.get("temporal_patch_size", self.temporal_patch_size) != 1:
            raise ValueError("Cosmos3 Edge only supports `temporal_patch_size=1`.")
        return kwargs

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ) -> np.ndarray:
        pass

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        temporal_patch_size: int | None = None,
        merge_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchFeature:
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            if do_convert_rgb:
                stacked_videos = self.convert_to_rgb(stacked_videos)
            batch_size, num_frames, channels, height, width = stacked_videos.shape
            if do_resize:
                resized_height, resized_width = smart_resize_video(
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_factor=temporal_patch_size,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.reshape(batch_size * num_frames, channels, height, width)
                stacked_videos = self.resize(
                    stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.reshape(
                    batch_size, num_frames, channels, resized_height, resized_width
                )
            resized_videos_grouped[shape] = stacked_videos

        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)
        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}

        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(stacked_videos[0], channel_dim=ChannelDimension.FIRST)
            patch_group_size = patch_size * merge_size
            if resized_height % patch_group_size or resized_width % patch_group_size:
                raise ValueError(
                    "Video frames must have dimensions divisible by `patch_size * merge_size`, got "
                    f"height={resized_height}, width={resized_width}, patch_size={patch_size}, "
                    f"merge_size={merge_size}."
                )
            stacked_videos = self.rescale_and_normalize(
                stacked_videos, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            batch_size, grid_t, channels = stacked_videos.shape[:3]
            grid_height, grid_width = resized_height // patch_size, resized_width // patch_size

            patches = stacked_videos.reshape(
                batch_size,
                grid_t,
                channels,
                grid_height // merge_size,
                merge_size,
                patch_size,
                grid_width // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 1, 3, 6, 4, 7, 5, 8, 2)
            processed_videos_grouped[shape] = patches.reshape(
                batch_size, grid_t * grid_height * grid_width, channels * patch_size * patch_size
            )
            processed_grids[shape] = [[grid_t, grid_height, grid_width]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        return BatchFeature(
            data={
                "pixel_values_videos": torch.cat(processed_videos, dim=0),
                "video_grid_thw": torch.tensor(processed_grids, dtype=torch.long),
            },
            tensor_type=return_tensors,
        )

    def get_number_of_video_patches(
        self, num_frames: int, height: int, width: int, videos_kwargs: dict | None = None
    ) -> int:
        pass


__all__ = ["Cosmos3EdgeVideoProcessor"]
