
from ...image_processing_outputs import SemanticSegmentationPostProcessorOutput
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from ...utils import is_torch_available, requires_backends
from ...video_processing_utils import BaseVideoProcessor


if is_torch_available():
    import torch
    import torch.nn.functional as F


def check_segment_validity(
    mask_labels: "torch.Tensor",
    mask_probs: "torch.Tensor",
    query_idx: int,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
) -> tuple[bool, "torch.Tensor"]:
    pass


def compute_segments(
    mask_probs: "torch.Tensor",
    pred_scores: "torch.Tensor",
    pred_labels: "torch.Tensor",
    label_ids_to_fuse: set[int] | None,
    mask_threshold: float = 0.5,
    overlap_mask_area_threshold: float = 0.8,
    target_size: tuple[int, int] | None = None,
) -> tuple["torch.Tensor", list[dict[str, int | float]]]:
    pass


class VideomtVideoProcessor(BaseVideoProcessor):
    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 640, "width": 640}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    rescale_factor = 1 / 255
    do_normalize = True
    do_convert_rgb = True
    do_sample_frames = False
    model_input_names = ["pixel_values_videos"]

    def _resize_mask_logits(
        self,
        masks_queries_logits: "torch.Tensor",
        target_sizes: list[tuple[int, int]],
    ) -> list["torch.Tensor"]:
        """Interpolates mask logits to each frame's original resolution."""
        resized = []
        for idx, original_size in enumerate(target_sizes):
            upsampled = F.interpolate(
                masks_queries_logits[idx][None, ...],
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )[0]
            resized.append(upsampled)
        return resized

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        return_segmentation_scores: bool = False,
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Converts the output of [`VideomtForUniversalSegmentation`] into semantic segmentation predictions.

        Args:
            outputs ([`VideomtForUniversalSegmentationOutput`]):
                Raw outputs of the model.
            target_sizes (`list[tuple[int, int]]`):
                List of `(height, width)` tuples corresponding to the requested final size of each prediction.
                Length should match the number of frames in the output.
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
            `(height, width)` corresponds to the target size.
        """
        requires_backends(self, ["torch"])

        masks_queries_logits = outputs.masks_queries_logits  # [num_frames, num_queries, height, width]
        class_queries_logits = outputs.class_queries_logits  # [num_frames, num_queries, num_classes+1]

        masks_classes = class_queries_logits.float().softmax(dim=-1)[..., :-1]
        masks_probs = masks_queries_logits.float().sigmoid()

        segmentation_logits = torch.matmul(masks_classes.transpose(1, 2), masks_probs.flatten(2))
        segmentation_logits = segmentation_logits.reshape(
            masks_probs.shape[0], masks_classes.shape[-1], masks_probs.shape[-2], masks_probs.shape[-1]
        )

        output_logits = self._resize_mask_logits(segmentation_logits, target_sizes)

        semantic_segmentation = [
            SemanticSegmentationPostProcessorOutput(
                data={"segmentation": logit.argmax(dim=0), "segmentation_scores": logit}
            )
            for logit in output_logits
        ]

        if not return_segmentation_scores:
            semantic_segmentation = [item.segmentation for item in semantic_segmentation]

        return semantic_segmentation

    def post_process_instance_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.5,
    ) -> list[dict]:
        pass

    def post_process_panoptic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]],
        threshold: float = 0.8,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        label_ids_to_fuse: set[int] | None = None,
    ) -> list[dict]:
        pass


__all__ = ["VideomtVideoProcessor"]
