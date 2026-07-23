
import math
from typing import Any, Optional, Union

import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_outputs import SemanticSegmentationPostProcessorOutput
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import get_size_with_aspect_ratio, group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size_for_max_height_width,
    get_max_height_width,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, logging


logger = logging.get_logger(__name__)


class Mask2FormerImageProcessorKwargs(ImagesKwargs, total=False):

    ignore_index: int | None
    do_reduce_labels: bool
    num_labels: int | None
    size_divisor: int
    pad_size: SizeDict | None


def binary_mask_to_rle(mask: "torch.Tensor | np.ndarray") -> list[int]:
    pass


def convert_segmentation_to_rle(segmentation):
    pass


def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    pass


def check_segment_validity(mask_labels, mask_probs, k, mask_threshold=0.5, overlap_mask_area_threshold=0.8):
    pass


def compute_segments(
    mask_probs,
    pred_scores,
    pred_labels,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    label_ids_to_fuse: set[int] | None = None,
    target_size: tuple[int, int] | None = None,
):
    pass


def convert_segmentation_map_to_binary_masks_fast(
    segmentation_map: "torch.Tensor",
    instance_id_to_semantic_id: dict[int, int] | None = None,
    ignore_index: int | None = None,
    do_reduce_labels: bool = False,
):
    if do_reduce_labels and ignore_index is None:
        raise ValueError("If `do_reduce_labels` is True, `ignore_index` must be provided.")

    if do_reduce_labels:
        segmentation_map = torch.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    all_labels = torch.unique(segmentation_map)

    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]  # drop background label if applicable

    binary_masks = [(segmentation_map == i) for i in all_labels]
    if binary_masks:
        binary_masks = torch.stack(binary_masks, dim=0)
    else:
        binary_masks = torch.zeros((0, *segmentation_map.shape), device=segmentation_map.device)

    if instance_id_to_semantic_id is not None:
        labels = torch.zeros(all_labels.shape[0], device=segmentation_map.device)

        for i, label in enumerate(all_labels):
            class_id = instance_id_to_semantic_id[(label.item() + 1 if do_reduce_labels else label.item())]
            labels[i] = class_id - 1 if do_reduce_labels else class_id
    else:
        labels = all_labels
    return binary_masks.float(), labels.long()


@auto_docstring
class Mask2FormerImageProcessor(TorchvisionBackend):
    valid_kwargs = Mask2FormerImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 800, "longest_edge": 1333}
    default_to_square = False
    do_resize = True
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    model_input_names = ["pixel_values", "pixel_mask"]
    size_divisor = 32
    do_reduce_labels = False

    def __init__(self, **kwargs: Unpack[Mask2FormerImageProcessorKwargs]) -> None:
        size = kwargs.pop("size", None)
        max_size = kwargs.pop("max_size", None)

        if size is None and max_size is not None:
            size = self.size.copy()
            size["longest_edge"] = max_size
        elif size is None:
            size = self.size

        kwargs["size"] = get_size_dict(size, max_size=max_size, default_to_square=False)
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. This method calls the superclass method and then removes the
        `_max_size` attribute from the dictionary.
        """
        image_processor_dict = super().to_dict()
        image_processor_dict.pop("_max_size", None)
        return image_processor_dict

    def reduce_label(self, labels: list["torch.Tensor"]):
        for idx in range(len(labels)):
            label = labels[idx]
            label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype), label)
            label = label - 1
            label = torch.where(label == 254, torch.tensor(255, dtype=label.dtype), label)
            labels[idx] = label

    def resize(
        self,
        image: torch.Tensor,
        size: SizeDict,
        size_divisor: int = 0,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None" = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Resize the image to the given size. Size can be `min_size` (scalar) or `(height, width)` tuple. If size is an
        int, smaller edge of the image will be matched to this number.

        Args:
            image (`torch.Tensor`):
                Image to resize.
            size (`SizeDict`):
                Size of the image's `(height, width)` dimensions after resizing.
            size_divisor (`int`, *optional*, defaults to 0):
                If `size_divisor` is given, the output image size will be divisible by the number.
            resample (`PILImageResampling | tvF.InterpolationMode | int | None`, *optional*):
                Resampling filter to use if resizing the image.
        """

        if size.shortest_edge and size.longest_edge:
            new_size = get_size_with_aspect_ratio(
                image.size()[-2:],
                size.shortest_edge,
                size.longest_edge,
            )
        elif size.max_height and size.max_width:
            new_size = get_image_size_for_max_height_width(image.size()[-2:], size.max_height, size.max_width)
        elif size.height and size.width:
            new_size = (size.height, size.width)
        else:
            raise ValueError(
                f"Size must contain 'height' and 'width' keys or 'shortest_edge' and 'longest_edge' keys. Got {size}."
            )
        if size_divisor > 0:
            height, width = new_size
            height = int(math.ceil(height / size_divisor) * size_divisor)
            width = int(math.ceil(width / size_divisor) * size_divisor)
            new_size = (height, width)

        image = super().resize(
            image, size=SizeDict(height=new_size[0], width=new_size[1]), resample=resample, **kwargs
        )
        return image

    def pad(
        self,
        images: torch.Tensor,
        padded_size: tuple[int, int],
        segmentation_maps: torch.Tensor | None = None,
        fill: int = 0,
        ignore_index: int = 255,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
        original_size = images.size()[-2:]
        padding_bottom = padded_size[0] - original_size[0]
        padding_right = padded_size[1] - original_size[1]
        if padding_bottom < 0 or padding_right < 0:
            raise ValueError(
                f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                f"original size. Got padded size: {padded_size}, original size: {original_size}."
            )
        if original_size != padded_size:
            padding = [0, 0, padding_right, padding_bottom]
            images = tvF.pad(images, padding, fill=fill)
            if segmentation_maps is not None:
                segmentation_maps = [tvF.pad(mask, padding, fill=ignore_index) for mask in segmentation_maps]

        pixel_mask = torch.zeros((images.shape[0], *padded_size), dtype=torch.int64, device=images.device)
        pixel_mask[:, : original_size[0], : original_size[1]] = 1

        return images, pixel_mask, segmentation_maps

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = None,
        **kwargs: Unpack[Mask2FormerImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps.
        instance_id_to_semantic_id (`Union[list[dict[int, int]], dict[int, int]]`, *optional*):
            A mapping from instance IDs to semantic IDs.
        """
        return super().preprocess(images, segmentation_maps, instance_id_to_semantic_id, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Union[str, "torch.device"] | None = None,
        **kwargs: Unpack[Mask2FormerImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        To be overridden by subclasses when image-like inputs other than images should be processed.
        It can be used for segmentation maps, depth maps, etc.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        if segmentation_maps is not None:
            segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )
        return self._preprocess(images, segmentation_maps, instance_id_to_semantic_id, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        segmentation_maps: Optional["torch.Tensor"],
        instance_id_to_semantic_id: dict[int, int] | None,
        do_resize: bool | None,
        size: SizeDict | None,
        pad_size: SizeDict | None,
        size_divisor: int | None,
        resample: Union["PILImageResampling", "tvF.InterpolationMode"] | None,
        do_rescale: bool | None,
        rescale_factor: float | None,
        do_normalize: bool | None,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        ignore_index: int | None,
        do_reduce_labels: bool | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if segmentation_maps is not None and len(images) != len(segmentation_maps):
            raise ValueError("Images and segmentation maps must have the same length.")

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        if segmentation_maps is not None:
            grouped_segmentation_maps, grouped_segmentation_maps_index = group_images_by_shape(
                segmentation_maps, disable_grouping=disable_grouping
            )
            resized_segmentation_maps_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(
                    image=stacked_images, size=size, size_divisor=size_divisor, resample=resample
                )
            if segmentation_maps is not None:
                stacked_segmentation_maps = grouped_segmentation_maps[shape]
                if do_resize:
                    stacked_segmentation_maps = self.resize(
                        image=stacked_segmentation_maps,
                        size=size,
                        size_divisor=size_divisor,
                        resample=tvF.InterpolationMode.NEAREST_EXACT,
                    )
            resized_images_grouped[shape] = stacked_images
            if segmentation_maps is not None:
                resized_segmentation_maps_grouped[shape] = stacked_segmentation_maps
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)
        if segmentation_maps is not None:
            resized_segmentation_maps = reorder_images(
                resized_segmentation_maps_grouped, grouped_segmentation_maps_index
            )
        if pad_size is not None:
            padded_size = (pad_size.height, pad_size.width)
        else:
            padded_size = get_max_height_width(resized_images)

        if segmentation_maps is not None:
            mask_labels = []
            class_labels = []
            for idx, segmentation_map in enumerate(resized_segmentation_maps):
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id
                masks, classes = convert_segmentation_map_to_binary_masks_fast(
                    segmentation_map.squeeze(0),
                    instance_id,
                    ignore_index=ignore_index,
                    do_reduce_labels=do_reduce_labels,
                )
                mask_labels.append(masks)
                class_labels.append(classes)

        if segmentation_maps is not None:
            grouped_images, grouped_segmentation_maps, grouped_images_index = group_images_by_shape(
                resized_images, mask_labels, disable_grouping=disable_grouping
            )
            processed_segmentation_maps_grouped = {}
        else:
            grouped_images, grouped_images_index = group_images_by_shape(
                resized_images, disable_grouping=disable_grouping
            )
        processed_images_grouped = {}
        processed_pixel_masks_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            padded_images, pixel_masks, padded_segmentation_maps = self.pad(
                images=stacked_images,
                segmentation_maps=grouped_segmentation_maps[shape] if segmentation_maps is not None else None,
                padded_size=padded_size,
                ignore_index=ignore_index,
            )
            processed_images_grouped[shape] = padded_images
            processed_pixel_masks_grouped[shape] = pixel_masks
            if segmentation_maps is not None:
                processed_segmentation_maps_grouped[shape] = padded_segmentation_maps

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)
        processed_pixel_masks = reorder_images(processed_pixel_masks_grouped, grouped_images_index)
        encoded_inputs = BatchFeature(
            data={"pixel_values": processed_images, "pixel_mask": processed_pixel_masks},
            tensor_type=return_tensors,
        )
        if segmentation_maps is not None:
            mask_labels = reorder_images(processed_segmentation_maps_grouped, grouped_images_index)
            encoded_inputs["mask_labels"] = mask_labels
            encoded_inputs["class_labels"] = class_labels

        return encoded_inputs

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
        return_segmentation_scores: bool = False,
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_segmentation_scores (`bool`, *optional*, defaults to `False`):
                Whether to return segmentation scores alongside the segmentation map. When `True`, each element of
                the returned list is a [`SemanticSegmentationPostProcessorOutput`] with fields `segmentation`
                (class IDs, shape `(height, width)`) and `segmentation_scores` (shape `(num_classes, height, width)`).

        Returns:
            `list[torch.Tensor]` or `list[SemanticSegmentationPostProcessorOutput]`: When
            `return_segmentation_scores=False` (default), a list of length `batch_size` where each item is a
            segmentation map of shape `(height, width)` with class IDs. When `return_segmentation_scores=True`,
            a list of [`SemanticSegmentationPostProcessorOutput`] with fields `segmentation` (class IDs, shape
            `(height, width)`) and `segmentation_scores` (shape `(num_classes, height, width)`). In both cases,
            `(height, width)` corresponds to the target size (if `target_sizes` is specified).
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.sigmoid()  # [batch_size, num_queries, height, width]

        segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(
                    SemanticSegmentationPostProcessorOutput(
                        data={"segmentation": semantic_map, "segmentation_scores": resized_logits[0]}
                    )
                )
        else:
            semantic_map = segmentation.argmax(dim=1)
            semantic_segmentation = [
                SemanticSegmentationPostProcessorOutput(
                    data={"segmentation": semantic_map[i], "segmentation_scores": segmentation[i]}
                )
                for i in range(batch_size)
            ]

        if not return_segmentation_scores:
            semantic_segmentation = [item.segmentation for item in semantic_segmentation]

        return semantic_segmentation

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: list[tuple[int, int]] | None = None,
        return_coco_annotation: bool | None = False,
        return_binary_maps: bool | None = False,
    ) -> list[dict]:
        pass

    def post_process_panoptic_segmentation(
        self,
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: set[int] | None = None,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> list[dict]:
        pass


__all__ = ["Mask2FormerImageProcessor"]
