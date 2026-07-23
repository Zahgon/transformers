
import json
import os
from typing import Union

import torch
from huggingface_hub.utils import RepositoryNotFoundError
from torch import nn
from torchvision.transforms.v2 import functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_outputs import SemanticSegmentationPostProcessorOutput
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_max_height_width,
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, hf_api, logging


logger = logging.get_logger(__name__)


class OneFormerImageProcessorKwargs(ImagesKwargs, total=False):

    repo_path: str | None
    class_info_file: str | None
    num_text: int | None
    num_labels: int | None
    ignore_index: int | None
    do_reduce_labels: bool


def prepare_metadata(class_info):
    metadata = {}
    class_names = []
    thing_ids = []
    for key, info in class_info.items():
        metadata[key] = info["name"]
        class_names.append(info["name"])
        if info["isthing"]:
            thing_ids.append(int(key))
    metadata["thing_ids"] = thing_ids
    metadata["class_names"] = class_names
    return metadata


def load_metadata(repo_id, class_info_file):
    fname = os.path.join("" if repo_id is None else repo_id, class_info_file)

    if not os.path.exists(fname) or not os.path.isfile(fname):
        if repo_id is None:
            raise ValueError(f"Could not file {fname} locally. repo_id must be defined if loading from the hub")
        try:
            fname = hf_api().hf_hub_download(repo_id, class_info_file, repo_type="dataset")
        except RepositoryNotFoundError:
            fname = hf_api().hf_hub_download(repo_id, class_info_file)

    with open(fname, "r") as f:
        class_info = json.load(f)

    return class_info


def make_pixel_mask(image: "torch.Tensor", output_size: tuple[int, int]) -> "torch.Tensor":
    """
    Make a pixel mask for the image, where 1 indicates a valid pixel and 0 indicates padding.

    Args:
        image (`torch.Tensor`):
            Image to make the pixel mask for.
        output_size (`Tuple[int, int]`):
            Output size of the mask.
    """

    input_height, input_width = image.shape[-2], image.shape[-1]
    mask = torch.zeros(output_size, dtype=torch.int64)
    mask[:input_height, :input_width] = 1
    return mask


def binary_mask_to_rle(mask):
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


@auto_docstring
class OneFormerImageProcessor(TorchvisionBackend):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"shortest_edge": 800, "longest_edge": 1333}
    do_resize = True
    do_rescale = True
    do_normalize = True
    default_to_square = False
    do_convert_rgb = True
    rescale_factor = 1 / 255
    ignore_index = None
    do_reduce_labels = False
    repo_path = "shi-labs/oneformer_demo"
    class_info_file = None
    num_text = None
    num_labels = None
    valid_kwargs = OneFormerImageProcessorKwargs
    model_input_names = ["pixel_values", "pixel_mask", "task_inputs"]

    def __init__(self, **kwargs: Unpack[OneFormerImageProcessorKwargs]):
        super().__init__(**kwargs)
        if self.class_info_file:
            self.metadata = prepare_metadata(load_metadata(self.repo_path, self.class_info_file))

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        task_inputs: list[str] | None = None,
        segmentation_maps: ImageInput | None = None,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = None,
        **kwargs: Unpack[OneFormerImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        task_inputs (`list[str]`, *optional*):
            List of tasks (`"panoptic"`, `"instance"`, `"semantic"`) for each image in the batch.
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps.
        instance_id_to_semantic_id (`Union[list[dict[int, int]], dict[int, int]]`, *optional*):
            A mapping from instance IDs to semantic IDs.
        """
        return super().preprocess(images, task_inputs, segmentation_maps, instance_id_to_semantic_id, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        task_inputs: list[str] | None,
        segmentation_maps: ImageInput,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Union[str, "torch.device"] | None = None,
        **kwargs: Unpack[OneFormerImageProcessorKwargs],
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
        return self._preprocess(images, task_inputs, segmentation_maps, instance_id_to_semantic_id, **kwargs)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        task_inputs: list[str] | None,
        segmentation_maps: list["torch.Tensor"],
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None,
        do_resize: bool,
        size: SizeDict,
        resample: "PILImageResampling | tvF.InterpolationMode | int | None",
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        ignore_index: int | None,
        do_reduce_labels: bool | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)

        processed_images_grouped = {}

        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        processed_segmentation_maps = None
        if segmentation_maps is not None:
            grouped_segmentation_maps, grouped_segmentation_maps_index = group_images_by_shape(
                segmentation_maps, disable_grouping=disable_grouping
            )
            processed_segmentation_maps_grouped = {}
            for shape, stacked_segmentation_maps in grouped_segmentation_maps.items():
                if do_resize:
                    stacked_segmentation_maps = self.resize(
                        stacked_segmentation_maps, size=size, resample=tvF.InterpolationMode.NEAREST_EXACT
                    )
                processed_segmentation_maps_grouped[shape] = stacked_segmentation_maps
            processed_segmentation_maps = reorder_images(
                processed_segmentation_maps_grouped, grouped_segmentation_maps_index
            )

        encoded_inputs = self.encode_inputs(
            processed_images,
            task_inputs,
            segmentation_maps=processed_segmentation_maps,
            instance_id_to_semantic_id=instance_id_to_semantic_id,
            ignore_index=ignore_index,
            do_reduce_labels=do_reduce_labels,
            return_tensors=return_tensors,
        )

        return encoded_inputs

    def _pad_image(
        self,
        image: "torch.Tensor",
        output_size: tuple[int, int],
        constant_values: float = 0,
    ) -> "torch.Tensor":
        """
        Pad an image with zeros to the given size using torch operations.

        Args:
            image (`torch.Tensor`):
                Image tensor in channel-first format (C, H, W).
            output_size (`tuple[int, int]`):
                Target output size (height, width).
            constant_values (`float`, *optional*, defaults to 0):
                The value to use for padding.

        Returns:
            `torch.Tensor`: The padded image.
        """
        input_height, input_width = image.shape[1], image.shape[2]
        output_height, output_width = output_size

        pad_bottom = output_height - input_height
        pad_right = output_width - input_width

        padded_image = tvF.pad(image, padding=[0, 0, pad_right, pad_bottom], fill=constant_values)

        return padded_image

    def pad(
        self,
        images: list["torch.Tensor"],
        return_pixel_mask: bool = True,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        """
        Pad a batch of images to the same size using torch operations.

        Args:
            images (`List[torch.Tensor]`):
                List of image tensors in channel-first format.
            return_pixel_mask (`bool`, *optional*, defaults to `True`):
                Whether to return pixel masks.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return.

        Returns:
            `BatchFeature`: Padded images and optional pixel masks.
        """
        outputs = super().pad(images, return_mask=return_pixel_mask)
        padded_images = outputs[0] if return_pixel_mask else outputs
        pixel_masks = outputs[1] if return_pixel_mask else None

        if return_tensors:
            padded_images = torch.stack(padded_images, dim=0)
            if return_pixel_mask:
                pixel_masks = torch.stack(pixel_masks, dim=0)

        data = {"pixel_values": padded_images}
        if return_pixel_mask:
            data["pixel_mask"] = pixel_masks

        return BatchFeature(data=data, tensor_type=return_tensors)

    def convert_segmentation_map_to_binary_masks(
        self,
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
            all_labels = all_labels[all_labels != ignore_index]

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

        return (
            binary_masks.float(),
            labels.long(),
        )

    def get_semantic_annotations(self, label, num_class_obj):
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        texts = ["a semantic photo"] * self.num_text
        classes = []
        masks = []

        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]
            if not torch.all(mask == 0):
                if class_id not in classes:
                    cls_name = self.metadata[str(class_id.cpu().item())]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1
                else:
                    idx = classes.index(class_id)
                    masks[idx] += mask
                    masks[idx] = torch.clamp(masks[idx], 0, 1)

        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = torch.stack(classes)
        masks = torch.stack(masks)
        return classes, masks, texts

    def get_instance_annotations(self, label, num_class_obj):
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        texts = ["an instance photo"] * self.num_text
        classes = []
        masks = []

        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx]

            if class_id in self.metadata["thing_ids"]:
                if not torch.all(mask == 0):
                    cls_name = self.metadata[str(class_id.cpu().item())]
                    classes.append(class_id)
                    masks.append(mask)
                    num_class_obj[cls_name] += 1

        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = torch.stack(classes)
        masks = torch.stack(masks)
        return classes, masks, texts

    def get_panoptic_annotations(self, label, num_class_obj):
        annotation_classes = label["classes"]
        annotation_masks = label["masks"]

        texts = ["an panoptic photo"] * self.num_text
        classes = []
        masks = []
        for idx in range(len(annotation_classes)):
            class_id = annotation_classes[idx]
            mask = annotation_masks[idx] if hasattr(annotation_masks[idx], "data") else annotation_masks[idx]
            if not torch.all(mask == 0):
                cls_name = self.metadata[str(class_id.cpu().item())]
                classes.append(class_id)
                masks.append(mask)
                num_class_obj[cls_name] += 1

        num = 0
        for i, cls_name in enumerate(self.metadata["class_names"]):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        classes = torch.stack(classes)
        masks = torch.stack(masks)
        return classes, masks, texts

    def encode_inputs(
        self,
        pixel_values_list: list["ImageInput"],
        task_inputs: list[str] | None = None,
        segmentation_maps: list["ImageInput"] | None = None,
        instance_id_to_semantic_id: list[dict[int, int]] | dict[int, int] | None = None,
        ignore_index: int | None = None,
        do_reduce_labels: bool = False,
        return_tensors: str | TensorType | None = None,
    ) -> BatchFeature:
        ignore_index = self.ignore_index if ignore_index is None else ignore_index
        do_reduce_labels = self.do_reduce_labels if do_reduce_labels is None else do_reduce_labels
        if task_inputs is None:
            task_inputs = ["panoptic"]
        pixel_values_list = self._prepare_image_like_inputs(
            pixel_values_list, input_data_format=ChannelDimension.FIRST
        )
        if segmentation_maps is not None:
            segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )
        pad_size = get_max_height_width(pixel_values_list)
        encoded_inputs = self.pad(pixel_values_list, return_tensors=return_tensors)

        annotations = None
        if segmentation_maps is not None:
            annotations = []
            for idx, segmentation_map in enumerate(segmentation_maps):
                if isinstance(instance_id_to_semantic_id, list):
                    instance_id = instance_id_to_semantic_id[idx]
                else:
                    instance_id = instance_id_to_semantic_id

                masks, classes = self.convert_segmentation_map_to_binary_masks(
                    segmentation_map,
                    instance_id,
                    ignore_index=ignore_index,
                    do_reduce_labels=do_reduce_labels,
                )

                annotations.append({"masks": masks, "classes": classes})

        if annotations is not None:
            mask_labels = []
            class_labels = []
            text_inputs = []
            num_class_obj = dict.fromkeys(self.metadata["class_names"], 0)

            for i, label in enumerate(annotations):
                task = task_inputs[i]

                if task == "semantic":
                    classes, masks, texts = self.get_semantic_annotations(label, num_class_obj)
                elif task == "instance":
                    classes, masks, texts = self.get_instance_annotations(label, num_class_obj)
                elif task == "panoptic":
                    classes, masks, texts = self.get_panoptic_annotations(label, num_class_obj)
                else:
                    raise ValueError(f"{task} was not expected, expected `semantic`, `instance` or `panoptic`")
                padded_masks = [
                    self._pad_image(image=mask, output_size=pad_size, constant_values=ignore_index) for mask in masks
                ]
                padded_masks = torch.cat(padded_masks, dim=0)
                mask_labels.append(padded_masks)
                class_labels.append(classes)
                text_inputs.append(texts)

            encoded_inputs["mask_labels"] = mask_labels
            encoded_inputs["class_labels"] = class_labels
            encoded_inputs["text_inputs"] = text_inputs

        encoded_inputs["task_inputs"] = [f"the task is {task_input}" for task_input in task_inputs]
        return encoded_inputs

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
        return_segmentation_scores: bool = False,
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Converts the output of [`OneFormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`OneFormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
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
                resized_logits = tvF.resize(
                    segmentation[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    interpolation=tvF.InterpolationMode.BILINEAR,
                )
                semantic_segmentation.append(
                    SemanticSegmentationPostProcessorOutput(
                        data={
                            "segmentation": resized_logits[0].argmax(dim=0),
                            "segmentation_scores": resized_logits[0],
                        }
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
        task_type: str = "instance",
        is_demo: bool = True,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: list[tuple[int, int]] | None = None,
        return_coco_annotation: bool | None = False,
    ):
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


__all__ = ["OneFormerImageProcessor"]
