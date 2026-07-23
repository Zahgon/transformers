
from typing import TYPE_CHECKING

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature
from ...image_utils import PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires


if TYPE_CHECKING:
    import torch

    from .modeling_superpoint import SuperPointKeypointDescriptionOutput


def is_grayscale(image: np.ndarray) -> bool:
    """Checks if an image is grayscale (all RGB channels are identical)."""
    if image.shape[0] == 1:
        return True
    return np.all(image[0, ...] == image[1, ...]) and np.all(image[1, ...] == image[2, ...])


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an image to grayscale format using the NTSC formula. Only support numpy arrays.

    This function is supposed to return a 1-channel image, but it returns a 3-channel image with the same value in each
    channel, because of an issue that is discussed in :
    https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

    Args:
        image (np.ndarray):
            The image to convert.
    """
    if is_grayscale(image):
        return image

    gray_image = image[0, ...] * 0.2989 + image[1, ...] * 0.5870 + image[2, ...] * 0.1140
    gray_image = np.stack([gray_image] * 3, axis=0)
    return gray_image


class SuperPointImageProcessorKwargs(ImagesKwargs, total=False):

    do_grayscale: bool


@auto_docstring
class SuperPointImageProcessorPil(PilBackend):
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
        images: list[np.ndarray],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        return_tensors: str | TensorType | None,
        do_grayscale: bool = False,
        **kwargs,
    ) -> BatchFeature:
        processed_images = []
        for image in images:
            if do_resize:
                image = self.resize(image, size, resample)
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_grayscale:
                image = convert_to_grayscale(image)
            processed_images.append(image)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    @requires(backends=("torch",))
    def post_process_keypoint_detection(
        self, outputs: "SuperPointKeypointDescriptionOutput", target_sizes: TensorType | list[tuple]
    ) -> list[dict[str, "torch.Tensor"]]:
        pass


__all__ = ["SuperPointImageProcessorPil"]
