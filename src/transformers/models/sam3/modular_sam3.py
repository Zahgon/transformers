

import torch

from ...image_processing_outputs import SemanticSegmentationPostProcessorOutput
from ...image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)
from ..sam2.image_processing_sam2 import Sam2ImageProcessor


def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


class Sam3ImageProcessor(Sam2ImageProcessor):
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size = {"height": 1008, "width": 1008}
    mask_size = {"height": 288, "width": 288}

    def post_process_semantic_segmentation(
        self,
        outputs,
        target_sizes: list[tuple[int, int]] | None = None,
        threshold: float = 0.5,
        return_segmentation_scores: bool = False,
    ) -> "list[torch.Tensor] | list[SemanticSegmentationPostProcessorOutput]":
        """
        Converts the output of [`Sam3Model`] into semantic segmentation maps.

        Args:
            outputs ([`Sam3ImageSegmentationOutput`]):
                Raw outputs of the model containing semantic_seg.
            target_sizes (`list[tuple[int, int]]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the semantic segmentation masks.
            return_segmentation_scores (`bool`, *optional*, defaults to `False`):
                Whether to return segmentation scores alongside the segmentation map. When `True`, each element of
                the returned list is a [`SemanticSegmentationPostProcessorOutput`] with fields `segmentation`
                (binary class IDs, shape `(height, width)`) and `segmentation_scores` (sigmoid probabilities,
                shape `(1, height, width)`).

        Returns:
            `list[torch.Tensor]` or `list[SemanticSegmentationPostProcessorOutput]`: When
            `return_segmentation_scores=False` (default), a list of length `batch_size` where each item is a
            segmentation map of shape `(height, width)` with class IDs. When `return_segmentation_scores=True`,
            a list of [`SemanticSegmentationPostProcessorOutput`] with fields `segmentation` (class IDs, shape
            `(height, width)`) and `segmentation_scores` (shape `(1, height, width)`). In both cases,
            `(height, width)` corresponds to the target size (if `target_sizes` is specified).
        """
        semantic_logits = outputs.semantic_seg

        if semantic_logits is None:
            raise ValueError(
                "Semantic segmentation output is not available in the model outputs. "
                "Make sure the model was run with semantic segmentation enabled."
            )

        semantic_probs = semantic_logits.sigmoid()
        batch_size = len(semantic_logits)

        if target_sizes is not None:
            if len(semantic_logits) != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_probs = torch.nn.functional.interpolate(
                    semantic_probs[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                semantic_segmentation.append(
                    SemanticSegmentationPostProcessorOutput(
                        data={
                            "segmentation": (resized_probs[0, 0] > threshold).to(torch.long),
                            "segmentation_scores": resized_probs[0],
                        }
                    )
                )
        else:
            semantic_segmentation = [
                SemanticSegmentationPostProcessorOutput(
                    data={
                        "segmentation": (semantic_probs[i, 0] > threshold).to(torch.long),
                        "segmentation_scores": semantic_probs[i],
                    }
                )
                for i in range(batch_size)
            ]

        if not return_segmentation_scores:
            semantic_segmentation = [item.segmentation for item in semantic_segmentation]

        return semantic_segmentation

    def post_process_object_detection(self, outputs, threshold: float = 0.3, target_sizes: list[tuple] | None = None):
        """
        Converts the raw output of [`Sam3Model`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

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
        """
        pred_logits = outputs.pred_logits  # (batch_size, num_queries)
        pred_boxes = outputs.pred_boxes  # (batch_size, num_queries, 4) in xyxy format
        presence_logits = outputs.presence_logits  # (batch_size, 1) or None

        batch_size = pred_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError("Make sure that you pass in as many target sizes as images")

        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  # (batch_size, 1)
            batch_scores = batch_scores * presence_scores  # Broadcast multiplication

        batch_boxes = pred_boxes

        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for scores, boxes in zip(batch_scores, batch_boxes):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            results.append({"scores": scores, "boxes": boxes})

        return results

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
        target_sizes: list[tuple] | None = None,
    ):
        pass


__all__ = ["Sam3ImageProcessor"]
