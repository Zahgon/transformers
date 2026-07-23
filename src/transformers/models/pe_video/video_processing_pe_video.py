
import torch

from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling
from ...processing_utils import Unpack, VideosKwargs
from ...video_processing_utils import BaseVideoProcessor, VideoMetadata
from ...video_utils import VideoInput


class PeVideoVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR

    def sample_frames(
        self,
        metadata: VideoMetadata,
        num_frames: int | None = None,
        fps: int | float | None = None,
        **kwargs,
    ):
        pass

    def _preprocess(
        self,
        videos: VideoInput,
        **kwargs: Unpack[VideosKwargs],
    ) -> BatchFeature:
        return_tensors = kwargs.pop("return_tensors", None)
        result = super()._preprocess(videos, **kwargs)
        pixels = result.pixel_values_videos
        data = {"pixel_values_videos": pixels}
        if return_tensors:
            lengths = torch.tensor([video.size(0) for video in pixels])
            pixels = torch.nn.utils.rnn.pad_sequence(pixels, batch_first=True, padding_value=0.0)
            data["pixel_values_videos"] = pixels
            if lengths.unique().size(0) > 1:
                mask = torch.arange(lengths.max())[None] < lengths[:, None]
                data["padding_mask_videos"] = mask
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["PeVideoVideoProcessor"]
