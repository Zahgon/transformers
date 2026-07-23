

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss

from .loss_d_fine import DFineForObjectDetectionLoss
from .loss_deformable_detr import DeformableDetrForObjectDetectionLoss, DeformableDetrForSegmentationLoss
from .loss_deimv2 import Deimv2ForObjectDetectionLoss
from .loss_for_object_detection import ForObjectDetectionLoss, ForSegmentationLoss
from .loss_grounding_dino import GroundingDinoForObjectDetectionLoss
from .loss_lw_detr import LwDetrForObjectDetectionLoss
from .loss_rf_detr import RfDetrForSegmentationLoss
from .loss_rnnt import ParakeetForRNNTLoss
from .loss_rt_detr import RTDetrForObjectDetectionLoss
from .loss_tdt import ParakeetForTDTLoss


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss


def ForMaskedLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    **kwargs,
):
    pass


def ForSequenceClassificationLoss(labels: torch.Tensor, pooled_logits: torch.Tensor, config, **kwargs) -> torch.Tensor:
    pass


def ForQuestionAnsweringLoss(start_logits, end_logits, start_positions, end_positions, **kwargs):
    pass


def ForTokenClassification(logits: torch.Tensor, labels, config, **kwargs):
    pass


def ForSemanticSegmentationLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = 255,
    num_items_in_batch: torch.Tensor | None = None,
    auxiliary_logits: torch.Tensor | None = None,
    auxiliary_loss_weight: float = 0.4,
    **kwargs,
) -> torch.Tensor:
    pass


LOSS_MAPPING = {
    "ForSemanticSegmentation": ForSemanticSegmentationLoss,
    "ForCausalLM": ForCausalLMLoss,
    "ForMaskedLM": ForMaskedLMLoss,
    "ForQuestionAnswering": ForQuestionAnsweringLoss,
    "ForSequenceClassification": ForSequenceClassificationLoss,
    "ForImageClassification": ForSequenceClassificationLoss,
    "ForVideoClassification": ForSequenceClassificationLoss,
    "ForAudioClassification": ForSequenceClassificationLoss,
    "ForTokenClassification": ForTokenClassification,
    "ForSegmentation": ForSegmentationLoss,
    "ForObjectDetection": ForObjectDetectionLoss,
    "ForConditionalGeneration": ForCausalLMLoss,
    "DeformableDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "ConditionalDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "DabDetrForObjectDetection": DeformableDetrForObjectDetectionLoss,
    "GroundingDinoForObjectDetection": GroundingDinoForObjectDetectionLoss,
    "MMGroundingDinoForObjectDetection": GroundingDinoForObjectDetectionLoss,
    "ConditionalDetrForSegmentation": DeformableDetrForSegmentationLoss,
    "RTDetrForObjectDetection": RTDetrForObjectDetectionLoss,
    "RTDetrV2ForObjectDetection": RTDetrForObjectDetectionLoss,
    "DFineForObjectDetection": DFineForObjectDetectionLoss,
    "Deimv2ForObjectDetection": Deimv2ForObjectDetectionLoss,
    "CsmForConditionalGeneration": ForCausalLMLoss,
    "LwDetrForObjectDetection": LwDetrForObjectDetectionLoss,
    "ParakeetForRNNT": ParakeetForRNNTLoss,
    "ParakeetForTDT": ParakeetForTDTLoss,
    "RfDetrForObjectDetection": LwDetrForObjectDetectionLoss,
    "RfDetrForInstanceSegmentation": RfDetrForSegmentationLoss,
}
