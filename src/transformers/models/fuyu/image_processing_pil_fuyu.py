
import math
from typing import TYPE_CHECKING

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_utils import (
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    is_valid_image,
    make_list_of_images,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available, requires_backends
from ...utils.import_utils import requires


if TYPE_CHECKING:
    import torch

if is_torch_available():
    import torch


class FuyuImagesKwargs(ImagesKwargs, total=False):

    patch_size: SizeDict | None
    padding_value: float
    padding_mode: str


def make_list_of_list_of_images(
    images: list[list[ImageInput]] | list[ImageInput] | ImageInput,
) -> list[list[ImageInput]]:
    if is_valid_image(images):
        return [[images]]

    if isinstance(images, list) and all(isinstance(image, list) for image in images):
        return images

    if isinstance(images, list):
        return [make_list_of_images(image) for image in images]

    raise ValueError("images must be a list of list of images or a list of images or an image.")


@auto_docstring
@requires(backends=("torch",))
class FuyuImageProcessorPil(PilBackend):
    do_resize = True
    size = {"height": 1080, "width": 1920}
    patch_size = {"height": 30, "width": 30}
    resample = PILImageResampling.BILINEAR
    do_pad = True
    padding_value = 1.0
    padding_mode = "constant"
    do_normalize = True
    image_mean = 0.5
    image_std = 0.5
    do_rescale = True
    rescale_factor = 1 / 255
    model_input_names = [
        "images",
        "image_input_ids",
        "image_patches",
        "image_patch_indices_per_batch",
        "image_patch_indices_per_subsequence",
    ]
    valid_kwargs = FuyuImagesKwargs

    def __init__(self, **kwargs: Unpack[FuyuImagesKwargs]):
        super().__init__(**kwargs)

    def _prepare_images_structure(self, images: ImageInput, expected_ndims: int = 3) -> ImageInput:
        images = self.fetch_images(images)
        return make_list_of_list_of_images(images)

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        resample: PILImageResampling | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize an image to fit within `(size.height, size.width)` while maintaining aspect ratio.
        Only resizes if the image is larger than the target size.
        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the max size of the output image.
            resample (`PILImageResampling | tvF.InterpolationMode | int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
        """

        input_height, input_width = image.shape[-2:]
        target_height, target_width = size.height, size.width
        if input_width <= target_width and input_height <= target_height:
            return image
        height_scale_factor = target_height / input_height
        width_scale_factor = target_width / input_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(input_height * optimal_scale_factor)
        new_width = int(input_width * optimal_scale_factor)

        return super().resize(image, SizeDict(height=new_height, width=new_width), resample=resample)

    def _preprocess(
        self,
        images: list[list[np.ndarray]],
        do_resize: bool,
        size: SizeDict,
        resample: PILImageResampling | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        padding_value: float | None,
        padding_mode: str | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        original_image_sizes = []
        processed_images = []
        for batch_images in images:
            if batch_images:
                original_image_sizes.append(batch_images[0].shape[-2:])
                processed_batch = []
                for image in batch_images:
                    if do_resize:
                        image = self.resize(image=image, size=size, resample=resample)
                    processed_batch.append(image)
                processed_images.append(processed_batch)
            else:
                processed_images.append([])

        image_sizes = [batch_image[0].shape[-2:] for batch_image in processed_images if batch_image]
        image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
        image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]
        image_scale_factors = [
            [resized_size[0] / original_size[0]]
            for original_size, resized_size in zip(original_image_sizes, image_sizes)
        ]

        if do_pad:
            target_height, target_width = size.height, size.width
            for batch_idx, batch_images in enumerate(processed_images):
                for img_idx, image in enumerate(batch_images):
                    from ...image_utils import ChannelDimension

                    height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
                    padding_height = target_height - height
                    padding_width = target_width - width
                    if padding_height > 0 or padding_width > 0:
                        pad_width = ((0, 0), (0, padding_height), (0, padding_width))
                        if padding_mode == "constant":
                            image = np.pad(image, pad_width, mode="constant", constant_values=padding_value)
                        else:
                            image = np.pad(image, pad_width, mode=padding_mode)
                        processed_images[batch_idx][img_idx] = image

        for batch_idx, batch_images in enumerate(processed_images):
            for img_idx, image in enumerate(batch_images):
                if do_rescale:
                    image = self.rescale(image, rescale_factor)
                if do_normalize:
                    image = self.normalize(image, image_mean, image_std)
                processed_images[batch_idx][img_idx] = image

        return BatchFeature(
            data={
                "images": processed_images,
                "image_unpadded_heights": image_unpadded_heights,
                "image_unpadded_widths": image_unpadded_widths,
                "image_scale_factors": image_scale_factors,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["overflowing_values"],
        )

    def get_num_patches(self, image_height: int, image_width: int, patch_size: SizeDict | None = None) -> int:
        pass

    def patchify_image(
        self, image: "np.ndarray | torch.Tensor", patch_size: SizeDict | None = None
    ) -> "np.ndarray | torch.Tensor":
        pass

    def preprocess_with_tokenizer_info(
        self,
        image_input: "torch.Tensor",
        image_present: "torch.Tensor",
        image_unpadded_h: "torch.Tensor",
        image_unpadded_w: "torch.Tensor",
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
        patch_size: dict[str, int] | None = None,
    ) -> BatchFeature:
        pass

    def _standardize_kwargs(self, patch_size: dict[str, int] | SizeDict | None = None, **kwargs) -> dict:
        """
        Process Fuyu-specific kwargs before validation.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        if patch_size is not None and not isinstance(patch_size, SizeDict):
            patch_size = SizeDict(**get_size_dict(patch_size, param_name="patch_size"))
        kwargs["patch_size"] = patch_size
        return kwargs


__all__ = ["FuyuImageProcessorPil"]
