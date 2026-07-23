
import math

import torch
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    ImageInput,
    PILImageResampling,
    SizeDict,
    is_valid_image,
    make_list_of_images,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import (
    TensorType,
    auto_docstring,
    logging,
    requires_backends,
)


logger = logging.get_logger(__name__)


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


class FuyuImagesKwargs(ImagesKwargs, total=False):

    patch_size: SizeDict | None
    padding_value: float
    padding_mode: str


@auto_docstring
class FuyuImageProcessor(TorchvisionBackend):
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

    def _prepare_images_structure(
        self,
        images: ImageInput,
        expected_ndims: int = 3,
    ) -> ImageInput:
        images = self.fetch_images(images)
        return make_list_of_list_of_images(images)

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        antialias: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize an image to fit within `(size.height, size.width)` while maintaining aspect ratio.
        Only resizes if the image is larger than the target size.
        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Dictionary in the format `{"height": int, "width": int}` specifying the max size of the output image.
            resample (`PILImageResampling | tvF.InterpolationMode | int`, *optional*, defaults to `PILImageResampling.BILINEAR`):
                Resampling filter to use when resizing the image.
            antialias (`bool`, *optional*, defaults to `True`):
                Whether to apply antialiasing when resizing.
        """
        if resample is None:
            resample = PILImageResampling.BILINEAR
        image_height, image_width = image.shape[-2:]
        target_height, target_width = size.height, size.width
        if image_width <= target_width and image_height <= target_height:
            return image
        height_scale_factor = target_height / image_height
        width_scale_factor = target_width / image_width
        optimal_scale_factor = min(height_scale_factor, width_scale_factor)

        new_height = int(image_height * optimal_scale_factor)
        new_width = int(image_width * optimal_scale_factor)

        return super().resize(
            image, SizeDict(height=new_height, width=new_width), resample=resample, antialias=antialias
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        padding_value: float | None,
        padding_mode: str | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        original_image_sizes = [batch_image[0].shape[-2:] for batch_image in images if batch_image]
        grouped_images, grouped_images_index = group_images_by_shape(
            images, disable_grouping=disable_grouping, is_nested=True
        )
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index, is_nested=True)

        image_sizes = [batch_image[0].shape[-2:] for batch_image in resized_images if batch_image]
        image_unpadded_heights = [[image_size[0]] for image_size in image_sizes]
        image_unpadded_widths = [[image_size[1]] for image_size in image_sizes]
        image_scale_factors = [
            [resized_size[0] / original_size[0]]
            for original_size, resized_size in zip(original_image_sizes, image_sizes)
        ]
        if do_pad:
            resized_images = self.pad(
                resized_images,
                pad_size=size,
                fill_value=padding_value,
                padding_mode=padding_mode,
                disable_grouping=disable_grouping,
                is_nested=True,
            )
        grouped_images, grouped_images_index = group_images_by_shape(
            resized_images, disable_grouping=disable_grouping, is_nested=True
        )
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index, is_nested=True)

        images_tensor = torch.stack([torch.stack(batch) for batch in processed_images if batch])
        return BatchFeature(
            data={
                "images": images_tensor,
                "image_unpadded_heights": image_unpadded_heights,
                "image_unpadded_widths": image_unpadded_widths,
                "image_scale_factors": image_scale_factors,
            },
            tensor_type=return_tensors,
            skip_tensor_conversion=["overflowing_values"],
        )

    def get_num_patches(self, image_height: int, image_width: int, patch_size: SizeDict | None = None) -> int:
        pass

    def patchify_image(self, image: torch.Tensor, patch_size: SizeDict | None = None) -> torch.Tensor:
        pass

    def preprocess_with_tokenizer_info(
        self,
        image_input: torch.Tensor,
        image_present: torch.Tensor,
        image_unpadded_h: torch.Tensor,
        image_unpadded_w: torch.Tensor,
        image_placeholder_id: int,
        image_newline_id: int,
        variable_sized: bool,
        patch_size: dict[str, int] | None = None,
    ) -> BatchFeature:
        pass

    def _standardize_kwargs(
        self,
        patch_size: dict[str, int] | SizeDict | None = None,
        **kwargs,
    ) -> dict:
        """
        Process Fuyu-specific kwargs before validation.
        """
        kwargs = super()._standardize_kwargs(**kwargs)
        if patch_size is not None and not isinstance(patch_size, SizeDict):
            patch_size = SizeDict(**get_size_dict(patch_size, param_name="patch_size"))
        kwargs["patch_size"] = patch_size
        return kwargs


__all__ = ["FuyuImageProcessor"]
