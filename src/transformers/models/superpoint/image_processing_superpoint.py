
from typing import TYPE_CHECKING

import torch

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring


if TYPE_CHECKING:
    from .modeling_superpoint import SuperPointKeypointDescriptionOutput

from torchvision.transforms.v2 import functional as tvF


class SuperPointImageProcessorKwargs(ImagesKwargs, total=False):

    do_grayscale: bool


def is_grayscale(image: "torch.Tensor") -> bool:
    """Checks if an image is grayscale (all RGB channels are identical)."""
    if image.ndim < 3 or image.shape[0 if image.ndim == 3 else 1] == 1:
        return True
    return torch.all(image[..., 0, :, :] == image[..., 1, :, :]) and torch.all(
        image[..., 1, :, :] == image[..., 2, :, :]
    )


def convert_to_grayscale(image: "torch.Tensor") -> "torch.Tensor":
    """
    Converts an image to grayscale format using the NTSC formula. Only support torch.Tensor.

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (torch.Tensor):
            The image to convert.
    """
    if is_grayscale(image):
        return image
    return tvF.rgb_to_grayscale(image, num_output_channels=3)


@auto_docstring
class SuperPointImageProcessor(TorchvisionBackend):
    valid_kwargs = SuperPointImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    size = {"height": 480, "width": 640}
    default_to_square = False
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = None
    do_grayscale = False

    def __init__(self, **kwargs: Unpack[SuperPointImageProcessorKwargs]):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        do_grayscale: bool = False,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_grayscale:
                stacked_images = convert_to_grayscale(stacked_images)
            if do_resize:
                stacked_images = self.resize(stacked_images, size=size, resample=resample)
            if do_rescale:
                stacked_images = self.rescale(stacked_images, rescale_factor)
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def post_process_keypoint_detection(
        self, outputs: "SuperPointKeypointDescriptionOutput", target_sizes: TensorType | list[tuple]
    ) -> list[dict[str, "torch.Tensor"]]:
        pass


__all__ = ["SuperPointImageProcessor"]
