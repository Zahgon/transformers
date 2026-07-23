

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import is_vision_available
from .loss_for_object_detection import box_iou
from .loss_rt_detr import RTDetrHungarianMatcher, RTDetrLoss


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


def _set_aux_loss(outputs_class, outputs_coord):
    pass


def _set_aux_loss2(
    outputs_class, outputs_coord, outputs_corners, outputs_ref, teacher_corners=None, teacher_logits=None
):
    pass


def weighting_function(max_num_bins: int, up: torch.Tensor, reg_scale: int) -> torch.Tensor:
    """
    Generates the non-uniform Weighting Function W(n) for bounding box regression.

    Args:
        max_num_bins (int): Max number of the discrete bins.
        up (Tensor): Controls upper bounds of the sequence,
                     where maximum offset is ±up * H / W.
        reg_scale (float): Controls the curvature of the Weighting Function.
                           Larger values result in flatter weights near the central axis W(max_num_bins/2)=0
                           and steeper weights at both ends.
    Returns:
        Tensor: Sequence of Weighting Function.
    """
    upper_bound1 = abs(up[0]) * abs(reg_scale)
    upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
    step = (upper_bound1 + 1) ** (2 / (max_num_bins - 2))
    left_values = [-((step) ** i) + 1 for i in range(max_num_bins // 2 - 1, 0, -1)]
    right_values = [(step) ** i - 1 for i in range(1, max_num_bins // 2)]
    values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
    values = [v if v.dim() > 0 else v.unsqueeze(0) for v in values]
    values = torch.cat(values, 0)
    return values


def translate_gt(gt: torch.Tensor, max_num_bins: int, reg_scale: int, up: torch.Tensor):
    pass


def bbox2distance(points, bbox, max_num_bins, reg_scale, up, eps=0.1):
    pass


class DFineLoss(RTDetrLoss):

    def __init__(self, config):
        super().__init__(config)

        self.matcher = RTDetrHungarianMatcher(config)
        self.max_num_bins = config.max_num_bins
        self.weight_dict = {
            "loss_vfl": config.weight_loss_vfl,
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
            "loss_fgl": config.weight_loss_fgl,
            "loss_ddf": config.weight_loss_ddf,
        }
        self.losses = ["vfl", "boxes", "local"]
        self.reg_scale = config.reg_scale
        self.up = nn.Parameter(torch.tensor([config.up]), requires_grad=False)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        pass

    def loss_local(self, outputs, targets, indices, num_boxes, T=5):
        pass

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "cardinality": self.loss_cardinality,
            "local": self.loss_local,
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)


def DFineForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    outputs_class=None,
    outputs_coord=None,
    enc_topk_logits=None,
    enc_topk_bboxes=None,
    denoising_meta_values=None,
    predicted_corners=None,
    initial_reference_points=None,
    **kwargs,
):
    pass
