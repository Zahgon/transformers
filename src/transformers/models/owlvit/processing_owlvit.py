
from typing import TYPE_CHECKING

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import TensorType, auto_docstring, is_torch_available


if TYPE_CHECKING:
    from .modeling_owlvit import OwlViTImageGuidedObjectDetectionOutput, OwlViTObjectDetectionOutput


class OwlViTImagesKwargs(ImagesKwargs, total=False):

    query_images: ImageInput | None


class OwlViTProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: OwlViTImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class OwlViTProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[OwlViTProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
                `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
                `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **query_pixel_values** -- Pixel values of the query images to be fed to a model. Returned when `query_images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            OwlViTProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        query_images = output_kwargs["images_kwargs"].pop("query_images", None)
        return_tensors = output_kwargs["text_kwargs"]["return_tensors"]

        if text is None and query_images is None and images is None:
            raise ValueError(
                "You have to specify at least one text or query image or image. All three cannot be none."
            )

        data = {}
        if text is not None:
            if isinstance(text, str) or (isinstance(text, list) and not isinstance(text[0], list)):
                encodings = [self.tokenizer(text, **output_kwargs["text_kwargs"])]

            elif isinstance(text, list) and isinstance(text[0], list):
                encodings = []

                max_num_queries = max(len(text_single) for text_single in text)

                for text_single in text:
                    if len(text_single) != max_num_queries:
                        text_single = text_single + [" "] * (max_num_queries - len(text_single))

                    encoding = self.tokenizer(text_single, **output_kwargs["text_kwargs"])
                    encodings.append(encoding)
            else:
                raise TypeError("Input text should be a string, a list of strings or a nested list of strings")

            if return_tensors == "np":
                input_ids = np.concatenate([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = np.concatenate([encoding["attention_mask"] for encoding in encodings], axis=0)
            elif return_tensors == "pt" and is_torch_available():
                import torch

                input_ids = torch.cat([encoding["input_ids"] for encoding in encodings], dim=0)
                attention_mask = torch.cat([encoding["attention_mask"] for encoding in encodings], dim=0)
            else:
                raise ValueError("Target return tensor type could not be returned")

            data["input_ids"] = input_ids
            data["attention_mask"] = attention_mask

        if query_images is not None:
            query_pixel_values = self.image_processor(query_images, **output_kwargs["images_kwargs"]).pixel_values
            data = {"query_pixel_values": query_pixel_values}

        if images is not None:
            image_features = self.image_processor(images, **output_kwargs["images_kwargs"])
            data["pixel_values"] = image_features.pixel_values

        return BatchFeature(data=data, tensor_type=return_tensors)

    def post_process(self, *args, **kwargs):
        pass

    def post_process_grounded_object_detection(
        self,
        outputs: "OwlViTObjectDetectionOutput",
        threshold: float = 0.1,
        target_sizes: TensorType | list[tuple] | None = None,
        text_labels: list[list[str]] | None = None,
    ):
        pass

    def post_process_image_guided_detection(
        self,
        outputs: "OwlViTImageGuidedObjectDetectionOutput",
        threshold: float = 0.0,
        nms_threshold: float = 0.3,
        target_sizes: TensorType | list[tuple] | None = None,
    ):
        pass


__all__ = ["OwlViTProcessor"]
