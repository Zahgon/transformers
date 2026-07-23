
from copy import deepcopy

import numpy as np

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import TensorType, auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_cxcywh_to_xywh(x):
    pass


def box_xywh_to_xyxy(x):
    x, y, w, h = x.unbind(-1)
    b = [(x), (y), (x + w), (y + h)]
    return torch.stack(b, dim=-1)


def box_xywh_to_cxcywh(x):
    x, y, w, h = x.unbind(-1)
    b = [(x + 0.5 * w), (y + 0.5 * h), (w), (h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_xywh(x):
    pass


def box_xyxy_to_cxcywh(x):
    pass


def box_area(boxes):
    """
    Batched version of box area. Boxes should be in [x0, y0, x1, y1] format.

    Inputs:
    - boxes: Tensor of shape (..., 4)

    Returns:
    - areas: Tensor of shape (...,)
    """
    x0, y0, x1, y1 = boxes.unbind(-1)
    return (x1 - x0) * (y1 - y0)


@requires(backends=("torch",))
@auto_docstring
class Sam3Processor(ProcessorMixin):
    def __init__(
        self, image_processor, tokenizer, target_size: int | None = None, point_pad_value: int = -10, **kwargs
    ):
        r"""
        target_size (`int`, *optional*):
            The target size (target_size, target_size) to which the image will be resized.
        point_pad_value (`int`, *optional*, defaults to -10):
            The value used for padding input boxes.
        """
        super().__init__(image_processor, tokenizer, **kwargs)
        self.point_pad_value = point_pad_value
        self.target_size = target_size if target_size is not None else self.image_processor.size["height"]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        segmentation_maps: ImageInput | None = None,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = None,
        input_boxes_labels: list[list[list[int]]] | torch.Tensor | None = None,
        original_sizes: list[list[float]] | torch.Tensor | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        images (`ImageInput`, *optional*):
            The image(s) to process.
        text (`str`, `list[str]`, `list[list[str]]`, *optional*):
            The text to process.
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to process.
        input_boxes (`list[list[list[float]]]`, `torch.Tensor`, *optional*):
            The bounding boxes to process.
        input_boxes_labels (`list[list[int]]`, `torch.Tensor`, *optional*):
            The labels for the bounding boxes.
        original_sizes (`list[list[float]]`, `torch.Tensor`, *optional*):
            The original sizes of the images.

        Returns:
            A [`BatchEncoding`] with the following fields:
            - `pixel_values` (`torch.Tensor`): The processed image(s).
            - `original_sizes` (`list[list[float]]`): The original sizes of the images.
            - `labels` (`torch.Tensor`): The processed segmentation maps (if provided).
            - `input_boxes_labels` (`torch.Tensor`): The processed labels for the bounding boxes.
            - `input_boxes` (`torch.Tensor`): The processed bounding boxes.
        """
        encoding = None
        if images is not None:
            encoding = self.image_processor(
                images,
                segmentation_maps=segmentation_maps,
                return_tensors=return_tensors,
                **kwargs,
            )
        elif original_sizes is not None:
            if isinstance(original_sizes, torch.Tensor):
                original_sizes = original_sizes.cpu().tolist()
            encoding = BatchEncoding({"original_sizes": original_sizes}, tensor_type=return_tensors)
        elif input_boxes is not None:
            raise ValueError("Either images or original_sizes must be provided if input_boxes is not None")

        text = self._resolve_text_prompts(text, input_boxes)
        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors=return_tensors, padding="max_length", max_length=32)
            if encoding is not None:
                encoding.update(text_inputs)
            else:
                encoding = text_inputs

        if input_boxes is not None:
            original_sizes = encoding["original_sizes"]
            processed_boxes = self._validate_single_input(
                input_boxes,
                expected_depth=3,
                input_name="boxes",
                expected_format="[image level, box level, box coordinates]",
                expected_coord_size=4,
            )
            processed_boxes_labels = self._validate_single_input(
                input_boxes_labels,
                expected_depth=2,
                input_name="labels",
                expected_format="[image level, box level]",
            )

            if processed_boxes is not None and processed_boxes_labels is None:
                processed_boxes_labels = self._generate_default_box_labels(processed_boxes)

            if processed_boxes is not None:
                boxes_max_dims = self._get_nested_dimensions(processed_boxes)[:2]
            if processed_boxes_labels is not None:
                boxes_labels_max_dims = self._get_nested_dimensions(processed_boxes_labels)[:2]

            if processed_boxes is not None and processed_boxes_labels is not None:
                if boxes_max_dims != boxes_labels_max_dims:
                    raise ValueError(
                        "Input boxes and labels have inconsistent dimensions. Please ensure they have the same dimensions."
                    )

            if processed_boxes is not None:
                padded_boxes = self._pad_nested_list(processed_boxes, boxes_max_dims + [4])
                final_boxes = torch.tensor(padded_boxes, dtype=torch.float32)
                self._normalize_tensor_coordinates(
                    final_boxes, original_sizes, is_bounding_box=True, preserve_padding=True
                )
                final_boxes = box_xyxy_to_cxcywh(final_boxes)
                encoding.update({"input_boxes": final_boxes})

            if processed_boxes_labels is not None:
                padded_boxes_labels = self._pad_nested_list(processed_boxes_labels, boxes_labels_max_dims)
                final_boxes_labels = torch.tensor(padded_boxes_labels, dtype=torch.int64)
                encoding.update({"input_boxes_labels": final_boxes_labels})

        return encoding

    def _normalize_coordinates(self, coords: "torch.Tensor", original_size, is_bounding_box=False) -> "torch.Tensor":
        """
        Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format.

        Args:
            target_size (`int`):
                The target size of the image.
            coords (`torch.Tensor`):
                The coordinates to be normalized.
            original_size (`tuple`):
                The original size of the image.
            is_bounding_box (`bool`, *optional*, defaults to `False`):
                Whether the coordinates are bounding boxes.
        """
        old_h, old_w = original_size
        coords = deepcopy(coords).float()

        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)
        coords[..., 0] = coords[..., 0] / old_w
        coords[..., 1] = coords[..., 1] / old_h

        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        return coords

    def _generate_default_box_labels(self, processed_boxes):
        pass

    def _convert_to_nested_list(self, data, expected_depth, current_depth=0):
        pass

    def _resolve_text_prompts(self, text, input_boxes):
        pass

    def _get_nested_dimensions(self, nested_list, max_dims=None):
        pass

    def _pad_nested_list(self, nested_list, target_dims, current_level=0, pad_value=None):
        pass

    def _create_empty_nested_structure(self, dims, pad_value):
        pass

    def _get_nesting_level(self, input_list):
        pass

    def _validate_single_input(
        self,
        data: torch.Tensor | np.ndarray | list,
        expected_depth: int,
        input_name: str,
        expected_format: str,
        expected_coord_size: int | None = None,
    ) -> list:
        pass

    def _normalize_tensor_coordinates(self, tensor, original_sizes, is_bounding_box=False, preserve_padding=False):
        pass

    def post_process_semantic_segmentation(
        self, outputs, target_sizes=None, threshold=0.5, return_segmentation_scores=False
    ):
        """
        Converts the output of [`Sam3Model`] into semantic segmentation maps.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing semantic_seg.
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the semantic segmentation masks.
            return_segmentation_scores (`bool`, *optional*, defaults to `False`):
                Whether to return segmentation scores alongside the segmentation map.

        Returns:
            semantic_segmentation: `list[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry is a binary mask (0 or 1).
        """
        return self.image_processor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
            return_segmentation_scores=return_segmentation_scores,
        )

    def post_process_object_detection(self, outputs, threshold=0.3, target_sizes=None):
        """
        Converts the raw output of [`Sam3Model`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format. This is a convenience wrapper around the image processor method.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing pred_boxes, pred_logits, and optionally presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep object detection predictions.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted box on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.

        Example:

        ```python
        >>> from transformers import AutoModel, AutoProcessor
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> model = AutoModel.from_pretrained("facebook/sam3-base")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam3-base")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))
        >>> inputs = processor(images=image, text="cat", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> # Post-process to get bounding boxes
        >>> results = processor.post_process_object_detection(outputs, threshold=0.3, target_sizes=[image.size[::-1]])
        >>> boxes = results[0]["boxes"]
        >>> scores = results[0]["scores"]
        ```
        """
        return self.image_processor.post_process_object_detection(outputs, threshold, target_sizes)

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold=0.3,
        mask_threshold=0.5,
        target_sizes=None,
    ):
        pass


__all__ = ["Sam3Processor"]
