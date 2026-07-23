

import torch
import torch.nn.functional as F

from ...feature_extraction_utils import BatchFeature
from ...image_processing_backends import TorchvisionBackend
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import PILImageResampling, SizeDict
from ...utils import auto_docstring
from ...utils.generic import TensorType
from ...utils.import_utils import requires


@auto_docstring
@requires(backends=("torch",))
class UVDocImageProcessor(TorchvisionBackend):
    do_rescale = True
    do_resize = True
    size = {"height": 712, "width": 488}
    resample = PILImageResampling.BILINEAR

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            stacked_images = stacked_images[:, [2, 1, 0], :, :]
            processed_images_grouped[shape] = stacked_images

        rescale_and_normalize_images = reorder_images(processed_images_grouped, grouped_images_index)

        original_images = rescale_and_normalize_images.copy()

        grouped_images, grouped_images_index = group_images_by_shape(
            rescale_and_normalize_images, disable_grouping=disable_grouping
        )
        interpolated_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = F.interpolate(
                    stacked_images, size=(size.height, size.width), mode="bilinear", align_corners=True
                )
            interpolated_images_grouped[shape] = stacked_images

        pixel_values = reorder_images(interpolated_images_grouped, grouped_images_index)

        return BatchFeature(
            data={"pixel_values": pixel_values, "original_images": original_images},
            tensor_type=return_tensors,
            skip_tensor_conversion=["original_images"],
        )

    def post_process_document_rectification(
        self,
        prediction: torch.Tensor,
        original_images: list[torch.Tensor],
        scale: float = 255.0,
    ) -> list[dict[str, torch.Tensor]]:
        pass


__all__ = ["UVDocImageProcessor"]
