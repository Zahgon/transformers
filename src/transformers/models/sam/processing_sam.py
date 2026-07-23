
from copy import deepcopy
from typing import Union

import numpy as np

from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available


if is_torch_available():
    import torch

NestedList = list[Union[float | int | None, "NestedList"]]


class SamImagesKwargs(ImagesKwargs, total=False):

    segmentation_maps: ImageInput | None
    input_points: "NestedList | torch.Tensor | None"
    input_labels: "NestedList | int | torch.Tensor | None"
    input_boxes: "NestedList | torch.Tensor | None"
    point_pad_value: int | None
    mask_size: dict[str, int]
    mask_pad_size: dict[str, int]


class SamProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: SamImagesKwargs
    _defaults = {
        "images_kwargs": {
            "point_pad_value": -10,
        }
    }


@auto_docstring
class SamProcessor(ProcessorMixin):
    def __init__(self, image_processor):
        super().__init__(image_processor)
        self.target_size = self.image_processor.size["longest_edge"]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs,
    ) -> BatchEncoding:
        output_kwargs = self._merge_kwargs(
            SamProcessorKwargs,
            tokenizer_init_kwargs={},
            **kwargs,
        )
        input_points = output_kwargs["images_kwargs"].pop("input_points", None)
        input_labels = output_kwargs["images_kwargs"].pop("input_labels", None)
        input_boxes = output_kwargs["images_kwargs"].pop("input_boxes", None)
        point_pad_value = output_kwargs["images_kwargs"].pop("point_pad_value", None)

        encoding_image_processor = self.image_processor(
            images,
            **output_kwargs["images_kwargs"],
        )

        original_sizes = encoding_image_processor["original_sizes"]

        if hasattr(original_sizes, "numpy"):
            original_sizes = original_sizes.numpy()

        input_points, input_labels, input_boxes = self._check_and_preprocess_points(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
        )

        encoding_image_processor = self._normalize_and_convert(
            encoding_image_processor,
            original_sizes,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors=output_kwargs["images_kwargs"].get("return_tensors"),
            point_pad_value=point_pad_value,
        )

        return encoding_image_processor

    def _normalize_and_convert(
        self,
        encoding_image_processor,
        original_sizes,
        input_points=None,
        input_labels=None,
        input_boxes=None,
        return_tensors="pt",
        point_pad_value=-10,
    ):
        pass

    def _pad_points_and_labels(self, input_points, input_labels, point_pad_value):
        pass

    def _normalize_coordinates(
        self, target_size: int, coords: np.ndarray, original_size, is_bounding_box=False
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.image_processor._get_preprocess_shape(original_size, longest_edge=target_size)
        coords = deepcopy(coords).astype(float)

        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)

        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        return coords

    def _check_and_preprocess_points(
        self,
        input_points=None,
        input_labels=None,
        input_boxes=None,
    ):
        pass

    @property
    def model_input_names(self):
        pass

    def post_process_masks(self, *args, **kwargs):
        return self.image_processor.post_process_masks(*args, **kwargs)


__all__ = ["SamProcessor"]
