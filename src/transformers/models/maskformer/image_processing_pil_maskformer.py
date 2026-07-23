
import math
from typing import Any

import numpy as np

from ...image_processing_backends import PilBackend
from ...image_processing_outputs import SemanticSegmentationPostProcessorOutput
from ...image_processing_utils import BatchFeature, get_size_dict
from ...image_transforms import PaddingMode, get_size_with_aspect_ratio
from ...image_transforms import pad as np_pad
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    get_image_size_for_max_height_width,
    get_max_height_width,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available, logging, requires_backends
from ...utils.import_utils import requires


if is_torch_available():
    import torch
    from torch import nn

logger = logging.get_logger(__name__)


def convert_segmentation_map_to_binary_masks(
    segmentation_map: np.ndarray,
    instance_id_to_semantic_id: dict[int, int] | None = None,
    ignore_index: int | None = None,
    do_reduce_labels: bool = False,
):
    """Convert segmentation map to binary masks using NumPy operations."""
    if do_reduce_labels and ignore_index is None:
        raise ValueError("If `do_reduce_labels` is True, `ignore_index` must be provided.")

    if do_reduce_labels:
        segmentation_map = np.where(segmentation_map == 0, ignore_index, segmentation_map - 1)

    all_labels = np.unique(segmentation_map)

    if ignore_index is not None:
        all_labels = all_labels[all_labels != ignore_index]

    binary_masks = [(segmentation_map == i) for i in all_labels]
    if binary_masks:
        binary_masks = np.stack(binary_masks, axis=0)
    else:
        binary_masks = np.zeros((0, *segmentation_map.shape), dtype=np.float32)

    if instance_id_to_semantic_id is not None:
        labels = np.zeros(all_labels.shape[0], dtype=np.int64)

        for i, label in enumerate(all_labels):
            class_id = instance_id_to_semantic_id[(int(label) + 1 if do_reduce_labels else int(label))]
            labels[i] = class_id - 1 if do_reduce_labels else class_id
    else:
        labels = all_labels.astype(np.int64)
    return binary_masks.astype(np.float32), labels


class MaskFormerImageProcessorKwargs(ImagesKwargs, total=False):

    ignore_index: int | None
    do_reduce_labels: bool
    num_labels: int | None
    size_divisor: int
    pad_size: SizeDict | None


def binary_mask_to_rle(mask):
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


def convert_segmentation_to_rle(segmentation):
    pass


def remove_low_and_no_objects(masks, scores, labels, object_mask_threshold, num_labels):
    pass


@auto_docstring
@requires(backends=("torch",))
class MaskFormerImageProcessorPil(PilBackend):
    valid_kwargs = MaskFormerImageProcessorKwargs
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

    def __init__(self, **kwargs: Unpack[MaskFormerImageProcessorKwargs]) -> None:
        size = kwargs.pop("size", None)
        max_size = kwargs.pop("max_size", None)

        self._max_size = max_size if max_size is not None else 1333

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

    def reduce_label(self, labels: list[np.ndarray]):
        """Reduce label values by 1, replacing 0 with 255."""
        for idx in range(len(labels)):
            label = labels[idx].copy()
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255
            labels[idx] = label

    def resize(
        self,
        image: np.ndarray,
        size: SizeDict,
        size_divisor: int = 0,
        resample: PILImageResampling | None = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Resize the image to the given size with optional size_divisor.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`SizeDict`):
                Size of the image's `(height, width)` dimensions after resizing.
            size_divisor (`int`, *optional*, defaults to 0):
                If `size_divisor` is given, the output image size will be divisible by the number.
            resample (`PILImageResampling | int | None`, *optional*):
                Resampling filter to use if resizing the image.
        """

        if size.shortest_edge and size.longest_edge:
            height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
            new_size = get_size_with_aspect_ratio((height, width), size.shortest_edge, size.longest_edge)
        elif size.max_height and size.max_width:
            height, width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
            new_size = get_image_size_for_max_height_width((height, width), size.max_height, size.max_width)
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

        return super().resize(image, size=SizeDict(height=new_size[0], width=new_size[1]), resample=resample, **kwargs)

    def pad(
        self,
        images: list[np.ndarray],
        padded_size: tuple[int, int],
        segmentation_maps: list[np.ndarray] | None = None,
        fill: int = 0,
        ignore_index: int = 255,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray] | None]:
        """
        Pad images and optionally segmentation maps to the given size.

        Args:
            images (`list[np.ndarray]`):
                Images to pad.
            padded_size (`tuple[int, int]`):
                Target size (height, width) to pad to.
            segmentation_maps (`list[np.ndarray]`, *optional*):
                Segmentation maps to pad.
            fill (`int`, *optional*, defaults to 0):
                Fill value for images.
            ignore_index (`int`, *optional*, defaults to 255):
                Fill value for segmentation maps.

        Returns:
            `tuple`: (padded_images, pixel_masks, padded_segmentation_maps)
        """
        padded_images = []
        pixel_masks = []

        for image in images:
            original_size = image.shape[-2:]
            padding_bottom = padded_size[0] - original_size[0]
            padding_right = padded_size[1] - original_size[1]
            if padding_bottom < 0 or padding_right < 0:
                raise ValueError(
                    f"Padding dimensions are negative. Please make sure that the padded size is larger than the "
                    f"original size. Got padded size: {padded_size}, original size: {original_size}."
                )
            if original_size != padded_size:
                padding = ((0, padding_bottom), (0, padding_right))
                image = np_pad(
                    image,
                    padding,
                    mode=PaddingMode.CONSTANT,
                    constant_values=fill,
                    data_format=ChannelDimension.FIRST,
                    input_data_format=ChannelDimension.FIRST,
                )
            padded_images.append(image)

            pixel_mask = np.zeros(padded_size, dtype=np.int64)
            pixel_mask[: original_size[0], : original_size[1]] = 1
            pixel_masks.append(pixel_mask)

        padded_segmentation_maps = None
        if segmentation_maps is not None:
            padded_segmentation_maps = []
            for mask in segmentation_maps:
                original_size = mask.shape[-2:]
                padding_bottom = padded_size[0] - original_size[0]
                padding_right = padded_size[1] - original_size[1]
                if original_size != padded_size:
                    padding = ((0, padding_bottom), (0, padding_right))
                    mask = np_pad(
                        mask,
                        padding,
                        mode=PaddingMode.CONSTANT,
                        constant_values=ignore_index,
                        data_format=ChannelDimension.FIRST,
                        input_data_format=ChannelDimension.FIRST,
                    )
                padded_segmentation_maps.append(mask)

        return padded_images, pixel_masks, padded_segmentation_maps

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = None,
        **kwargs: Unpack[MaskFormerImageProcessorKwargs],
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
        **kwargs: Unpack[MaskFormerImageProcessorKwargs],
    ) -> BatchFeature:
        """
        Preprocess image-like inputs.
        To be overridden by subclasses when image-like inputs other than images should be processed.
        It can be used for segmentation maps, depth maps, etc.
        """
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format
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
        images: list[np.ndarray],
        segmentation_maps: list[np.ndarray] | None,
        instance_id_to_semantic_id: dict[int, int] | None,
        do_resize: bool | None,
        size: SizeDict | None,
        pad_size: SizeDict | None,
        size_divisor: int | None,
        resample: PILImageResampling | None,
        do_rescale: bool | None,
        rescale_factor: float | None,
        do_normalize: bool | None,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        ignore_index: int | None,
        do_reduce_labels: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if segmentation_maps is not None and len(images) != len(segmentation_maps):
            raise ValueError("Images and segmentation maps must have the same length.")

        resized_images = []
        resized_segmentation_maps = None
        if segmentation_maps is not None:
            resized_segmentation_maps = []

        for idx, image in enumerate(images):
            if do_resize:
                image = self.resize(image=image, size=size, size_divisor=size_divisor, resample=resample)
            resized_images.append(image)

            if segmentation_maps is not None:
                seg_map = segmentation_maps[idx]
                if do_resize:
                    seg_map = self.resize(
                        image=seg_map, size=size, size_divisor=size_divisor, resample=PILImageResampling.NEAREST
                    )
                resized_segmentation_maps.append(seg_map)

        if pad_size is not None:
            padded_size = (pad_size.height, pad_size.width)
        else:
            padded_size = get_max_height_width(resized_images, input_data_format=ChannelDimension.FIRST)

        mask_labels = None
        class_labels = None
        if segmentation_maps is not None:
            mask_labels = []
            class_labels = []
            for idx, segmentation_map in enumerate(resized_segmentation_maps):
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id
                if segmentation_map.ndim == 3 and segmentation_map.shape[0] == 1:
                    segmentation_map = segmentation_map.squeeze(0)
                masks, classes = convert_segmentation_map_to_binary_masks(
                    segmentation_map, instance_id, ignore_index=ignore_index, do_reduce_labels=do_reduce_labels
                )
                mask_labels.append(masks)
                class_labels.append(classes)

        processed_images = []
        for image in resized_images:
            if do_rescale:
                image = self.rescale(image, rescale_factor)
            if do_normalize:
                image = self.normalize(image, image_mean, image_std)
            processed_images.append(image)

        padded_images, pixel_masks, padded_mask_labels = self.pad(
            images=processed_images,
            padded_size=padded_size,
            segmentation_maps=mask_labels,
            fill=0,
            ignore_index=ignore_index,  # Match Torchvision backend for cross-backend equivalence
        )

        encoded_inputs = BatchFeature(
            data={"pixel_values": padded_images, "pixel_mask": pixel_masks}, tensor_type=return_tensors
        )
        if segmentation_maps is not None:
            encoded_inputs["mask_labels"] = [
                torch.from_numpy(mask_label) if return_tensors == "pt" else mask_label
                for mask_label in padded_mask_labels
            ]
            encoded_inputs["class_labels"] = [
                torch.from_numpy(class_label) if return_tensors == "pt" else class_label
                for class_label in class_labels
            ]

        return encoded_inputs

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple[int, int]] | None = None, return_segmentation_scores: bool = False
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Converts the output of [`MaskFormerForInstanceSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`MaskFormerForInstanceSegmentation`]):
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
        requires_backends(self, "torch")

        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        masks_queries_logits = outputs.masks_queries_logits  # [batch_size, num_queries, height, width]

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
            seg_maps = segmentation.argmax(dim=1)
            semantic_segmentation = [
                SemanticSegmentationPostProcessorOutput(
                    data={"segmentation": seg_maps[i], "segmentation_scores": segmentation[i]}
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


__all__ = ["MaskFormerImageProcessorPil"]
