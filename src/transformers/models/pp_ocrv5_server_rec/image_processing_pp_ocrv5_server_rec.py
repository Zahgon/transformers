
import math

import torch
import torchvision.transforms.v2.functional as tvF

from ...feature_extraction_utils import BatchFeature
from ...image_processing_backends import TorchvisionBackend
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...processing_utils import ImagesKwargs
from ...utils import auto_docstring, requires_backends
from ...utils.constants import IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from ...utils.generic import TensorType


class PPOCRV5ServerRecImageProcessorKwargs(ImagesKwargs, total=False):

    max_image_width: int
    character_list: str


@auto_docstring
class PPOCRV5ServerRecImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 48, "width": 320}
    pad_size = {"height": 48, "width": 320}
    do_resize = True
    do_rescale = True
    do_convert_rgb = True
    do_normalize = True
    do_pad = True
    max_image_width = 3200
    character_list = []
    valid_kwargs = PPOCRV5ServerRecImageProcessorKwargs

    def _preprocess(
        self,
        images: list["torch.Tensor"],
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
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}

        shape_list = list(grouped_images.keys())
        target_size = self.get_target_size(shape_list)

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images.float(), size=target_size, resample=resample)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        if do_pad and target_size.width < pad_size.width:
            processed_images = self.pad(processed_images, pad_size=pad_size, disable_grouping=disable_grouping)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def get_target_size(self, shape_list: list[torch.Size]):
        """
        Calculate the width and height from the widest image in the batch.
        """
        max_width = -1
        max_height = -1
        for height, width in shape_list:
            if width > max_width:
                max_width = width
                max_height = height

        default_height, default_width = self.size["height"], self.size["width"]
        ratio = max(max_width / max_height, default_width / default_height)

        target_width = int(default_height * ratio)
        target_height = default_height

        if target_width > self.max_image_width:
            target_width = self.max_image_width
        else:
            ratio = max_width / float(max_height)
            if target_width >= math.ceil(default_height * ratio):
                target_width = int(math.ceil(default_height * ratio))

        return SizeDict(height=target_height, width=target_width)

    def post_process_text_recognition(
        self,
        predictions,
    ) -> list[dict[str, str | float]]:
        pass


__all__ = ["PPOCRV5ServerRecImageProcessor"]
