
from copy import deepcopy

import numpy as np

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

if is_torch_available():
    import torch


@requires(backends=("torch",))
@auto_docstring
class Sam2Processor(ProcessorMixin):
    def __init__(self, image_processor, target_size: int | None = None, point_pad_value: int = -10, **kwargs):
        r"""
        target_size (`int`, *optional*):
            The target size (in pixels) for normalizing input points and bounding boxes. If not provided, defaults
            to the image processor's size configuration. All input coordinates (points and boxes) are normalized
            to this size before being passed to the model. This ensures consistent coordinate representation
            regardless of the original image dimensions.
        point_pad_value (`int`, *optional*, defaults to -10):
            The value used for padding input points when batching sequences of different lengths. This value is
            used to mark padded positions and is preserved during coordinate normalization.
        """
        super().__init__(image_processor, **kwargs)
        self.point_pad_value = point_pad_value
        self.target_size = target_size if target_size is not None else self.image_processor.size["height"]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        segmentation_maps: ImageInput | None = None,
        input_points: list[list[list[list[float]]]] | torch.Tensor | None = None,
        input_labels: list[list[list[int]]] | torch.Tensor | None = None,
        input_boxes: list[list[list[float]]] | torch.Tensor | None = None,
        original_sizes: list[list[float]] | torch.Tensor | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to process.
        input_points (`list[list[list[list[float]]]]`, `torch.Tensor`, *optional*):
            The points to add to the frame.
        input_labels (`list[list[list[int]]]`, `torch.Tensor`, *optional*):
            The labels for the points.
        input_boxes (`list[list[list[float]]]`, `torch.Tensor`, *optional*):
            The bounding boxes to add to the frame.
        original_sizes (`list[list[float]]`, `torch.Tensor`, *optional*):
            The original sizes of the images.

        Returns:
            A [`BatchEncoding`] with the following fields:
            - `pixel_values` (`torch.Tensor`): The processed image(s).
            - `original_sizes` (`list[list[float]]`): The original sizes of the images.
            - `labels` (`torch.Tensor`): The processed segmentation maps (if provided).
            - `input_points` (`torch.Tensor`): The processed points.
            - `input_labels` (`torch.Tensor`): The processed labels.
            - `input_boxes` (`torch.Tensor`): The processed bounding boxes.
        """
        if images is not None:
            encoding_image_processor = self.image_processor(
                images,
                segmentation_maps=segmentation_maps,
                return_tensors=return_tensors,
                **kwargs,
            )
        elif original_sizes is not None:
            if isinstance(original_sizes, torch.Tensor):
                original_sizes = original_sizes.cpu().tolist()
            encoding_image_processor = BatchEncoding({"original_sizes": original_sizes}, tensor_type=return_tensors)
        else:
            raise ValueError("Either images or original_sizes must be provided")

        original_sizes = encoding_image_processor["original_sizes"]
        if images is not None and len(original_sizes) != 1 and len(original_sizes) != len(images):
            raise ValueError(
                "original_sizes must be of length 1 or len(images). If you are passing a single image, you must pass a single original_size."
            )

        if input_points is not None or input_labels is not None or input_boxes is not None:
            processed_points = self._validate_single_input(
                input_points,
                expected_depth=4,
                input_name="points",
                expected_format="[image level, object level, point level, point coordinates]",
                expected_coord_size=2,
            )
            processed_labels = self._validate_single_input(
                input_labels,
                expected_depth=3,
                input_name="labels",
                expected_format="[image level, object level, point level]",
            )
            processed_boxes = self._validate_single_input(
                input_boxes,
                expected_depth=3,
                input_name="boxes",
                expected_format="[image level, box level, box coordinates]",
                expected_coord_size=4,
            )

            if processed_points is not None:
                points_max_dims = self._get_nested_dimensions(processed_points)[:3]
            if processed_labels is not None:
                labels_max_dims = self._get_nested_dimensions(processed_labels)[:3]
            if processed_boxes is not None:
                boxes_max_dims = self._get_nested_dimensions(processed_boxes)[:2]

            if processed_points is not None and processed_labels is not None:
                if points_max_dims != labels_max_dims:
                    raise ValueError(
                        "Input points and labels have inconsistent dimensions. Please ensure they have the same dimensions."
                    )

            if processed_boxes is not None and len(processed_boxes) >= 2:
                if any(len(img_boxes) < boxes_max_dims[1] for img_boxes in processed_boxes):
                    raise ValueError(
                        "Input boxes have inconsistent dimensions that would require padding, "
                        "but boxes cannot be padded due to model limitations. "
                        "Please ensure all images have the same number of boxes."
                    )

            if processed_points is not None:
                padded_points = self._pad_nested_list(processed_points, points_max_dims + [2])
                final_points = torch.tensor(padded_points, dtype=torch.float32)
                self._normalize_tensor_coordinates(final_points, original_sizes, preserve_padding=True)
                encoding_image_processor.update({"input_points": final_points})

            if processed_labels is not None:
                padded_labels = self._pad_nested_list(processed_labels, labels_max_dims)
                final_labels = torch.tensor(padded_labels, dtype=torch.int64)
                encoding_image_processor.update({"input_labels": final_labels})

            if processed_boxes is not None:
                final_boxes = torch.tensor(processed_boxes, dtype=torch.float32)
                self._normalize_tensor_coordinates(final_boxes, original_sizes, is_bounding_box=True)
                encoding_image_processor.update({"input_boxes": final_boxes})

        return encoding_image_processor

    def _normalize_coordinates(
        self, target_size: int, coords: "torch.Tensor", original_size, is_bounding_box=False
    ) -> "torch.Tensor":
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
        new_h, new_w = target_size, target_size
        coords = deepcopy(coords).float()

        if is_bounding_box:
            coords = coords.reshape(-1, 2, 2)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)

        if is_bounding_box:
            coords = coords.reshape(-1, 4)

        return coords

    def _convert_to_nested_list(self, data, expected_depth, current_depth=0):
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

    def post_process_masks(
        self,
        masks,
        original_sizes,
        mask_threshold=0.0,
        binarize=True,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
        apply_non_overlapping_constraints=False,
        **kwargs,
    ):
        """
        Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[List[torch.Tensor], List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold for binarization and post-processing operations.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            max_hole_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a hole to fill.
            max_sprinkle_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a sprinkle to fill.
            apply_non_overlapping_constraints (`bool`, *optional*, defaults to `False`):
                Whether to apply non-overlapping constraints to the masks.

        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.
        """
        return self.image_processor.post_process_masks(
            masks,
            original_sizes,
            mask_threshold,
            binarize,
            max_hole_area,
            max_sprinkle_area,
            apply_non_overlapping_constraints,
            **kwargs,
        )

    @property
    def model_input_names(self):
        pass


__all__ = ["Sam2Processor"]
