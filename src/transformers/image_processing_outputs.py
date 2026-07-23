from typing import TYPE_CHECKING

from .image_processing_base import BatchFeature


if TYPE_CHECKING:
    import torch


class SemanticSegmentationPostProcessorOutput(BatchFeature):

    segmentation: "torch.Tensor"
    segmentation_scores: "torch.Tensor | None"
