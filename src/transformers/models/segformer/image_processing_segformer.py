
from typing import Union

import torch
import torch.nn.functional as F
import torchvision.transforms.v2.functional as tvF

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
)
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, is_torch_available


class SegformerImageProcessorKwargs(ImagesKwargs, total=False):

    do_reduce_labels: bool


@auto_docstring
class SegformerImageProcessor(TorchvisionBackend):

    valid_kwargs = SegformerImageProcessorKwargs
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 512, "width": 512}
    default_to_square = True
    crop_size = None
    do_resize = True
    do_center_crop = None
    do_rescale = True
    do_normalize = True
    do_reduce_labels = False
    rescale_factor = 1 / 255

    def __init__(self, **kwargs: Unpack[SegformerImageProcessorKwargs]):
        super().__init__(**kwargs)

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        **kwargs: Unpack[SegformerImageProcessorKwargs],
    ) -> BatchFeature:
        r"""
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to preprocess.
        """
        return super().preprocess(images, segmentation_maps, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        return_tensors: str | TensorType | None,
        device: Union[str, "torch.device"] | None = None,
        **kwargs,
    ) -> BatchFeature:
        """Handle extra inputs beyond images."""
        images = self._prepare_image_like_inputs(
            images=images, do_convert_rgb=do_convert_rgb, input_data_format=input_data_format, device=device
        )
        images_kwargs = kwargs.copy()
        images_kwargs["do_reduce_labels"] = False
        data = {}
        data["pixel_values"] = self._preprocess(images, **images_kwargs)

        if segmentation_maps is not None:
            processed_segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            segmentation_maps_kwargs = kwargs.copy()
            segmentation_maps_kwargs.update(
                {
                    "do_normalize": False,
                    "do_rescale": False,
                    "resample": tvF.InterpolationMode.NEAREST_EXACT,
                }
            )
            processed_segmentation_maps = self._preprocess(
                images=processed_segmentation_maps, **segmentation_maps_kwargs
            )

            processed_segmentation_maps = [
                processed_segmentation_map.squeeze(0).to(torch.int64)
                for processed_segmentation_map in processed_segmentation_maps
            ]
            data["labels"] = processed_segmentation_maps

        return BatchFeature(data=data, tensor_type=return_tensors)

    def reduce_label(self, labels: list["torch.Tensor"]) -> list["torch.Tensor"]:
        """Reduce label values by 1, replacing 0 with 255."""
        for idx in range(len(labels)):
            label = labels[idx]
            label = torch.where(label == 0, torch.tensor(255, dtype=label.dtype, device=label.device), label)
            label = label - 1
            label = torch.where(label == 254, torch.tensor(255, dtype=label.dtype, device=label.device), label)
            labels[idx] = label
        return labels

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_reduce_labels: bool,
        resample: "PILImageResampling | None",
        do_resize: bool,
        do_rescale: bool,
        do_normalize: bool,
        size: SizeDict,
        rescale_factor: float,
        image_mean: float | list[float],
        image_std: float | list[float],
        disable_grouping: bool,
        **kwargs,
    ) -> BatchFeature:  # Return type can be list if return_tensors=None
        """Custom preprocessing for Segformer."""
        if do_reduce_labels:
            images = self.reduce_label(images)  # Apply reduction if needed

        resized_images = images
        if do_resize:
            grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
            resized_images_grouped = {}
            for shape, stacked_images in grouped_images.items():
                resized_stacked_images = self.resize(image=stacked_images, size=size, resample=resample)
                resized_images_grouped[shape] = resized_stacked_images
            resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images

        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        return processed_images

    def post_process_semantic_segmentation(
        self, outputs, target_sizes: list[tuple] | None = None, return_segmentation_scores: bool = False
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Converts the output of [`SegformerForSemanticSegmentation`] into semantic segmentation maps.

        Args:
            outputs ([`SegformerForSemanticSegmentation`]):
                Raw outputs of the model.
            target_sizes (`list[Tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
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
        if not is_torch_available():
            raise ImportError("PyTorch is required for post_process_semantic_segmentation")

        logits = outputs.logits

        if target_sizes is not None:
            if len(logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            if isinstance(target_sizes, torch.Tensor):
                target_sizes = target_sizes.numpy()

            semantic_segmentation = []

            for idx in range(len(logits)):
                resized_logits = F.interpolate(
                    logits[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(
                    SemanticSegmentationPostProcessorOutput(
                        data={"segmentation": semantic_map, "segmentation_scores": resized_logits[0]}
                    )
                )
        else:
            seg_maps = logits.argmax(dim=1)
            semantic_segmentation = [
                SemanticSegmentationPostProcessorOutput(
                    data={"segmentation": seg_maps[i], "segmentation_scores": logits[i]}
                )
                for i in range(logits.shape[0])
            ]

        if not return_segmentation_scores:
            semantic_segmentation = [item.segmentation for item in semantic_segmentation]

        return semantic_segmentation


__all__ = ["SegformerImageProcessor"]
