
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import is_scipy_available, is_vision_available, requires_backends
from .loss_for_object_detection import (
    box_iou,
    dice_loss,
    generalized_box_iou,
    nested_tensor_from_tensor_list,
    sigmoid_focal_loss,
)


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


def _set_aux_loss(outputs_class, outputs_coord):
    pass


class RTDetrHungarianMatcher(nn.Module):

    def __init__(self, config):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = config.matcher_class_cost
        self.bbox_cost = config.matcher_bbox_cost
        self.giou_cost = config.matcher_giou_cost

        self.use_focal_loss = config.use_focal_loss
        self.alpha = config.matcher_alpha
        self.gamma = config.matcher_gamma

        if self.class_cost == self.bbox_cost == self.giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])
        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["logits"].flatten(0, 1))
            out_prob = out_prob[:, target_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob**self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            class_cost = pos_cost_class - neg_cost_class
        else:
            out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            class_cost = -out_prob[:, target_ids]

        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class RTDetrLoss(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.matcher = RTDetrHungarianMatcher(config)
        self.num_classes = config.num_labels
        self.weight_dict = {
            "loss_vfl": config.weight_loss_vfl,
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
        }
        self.losses = ["vfl", "boxes"]
        self.eos_coef = config.eos_coefficient
        empty_weight = torch.ones(config.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)
        self.alpha = config.focal_loss_alpha
        self.gamma = config.focal_loss_gamma

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, log=True):
        pass

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "class_labels" containing a tensor of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")

        src_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_original = torch.cat([_target["class_labels"][i] for _target, (_, i) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_original

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.class_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pass

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        pass

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss. Targets dicts must contain the key
        "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def loss_labels_bce(self, outputs, targets, indices, num_boxes, log=True):
        pass

    def _get_source_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        pass

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "bce": self.loss_labels_bce,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["class_labels"]) for t in targets]
        device = targets[0]["class_labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def forward(self, outputs, targets):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`list[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if "auxiliary_outputs" not in k}

        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "dn_auxiliary_outputs" in outputs:
            if "denoising_meta_values" not in outputs:
                raise ValueError(
                    "The output must have the 'denoising_meta_values` key. Please, ensure that 'outputs' includes a 'denoising_meta_values' entry."
                )
            indices = self.get_cdn_matched_indices(outputs["denoising_meta_values"], targets)
            num_boxes = num_boxes * outputs["denoising_meta_values"]["dn_num_group"]

            for i, auxiliary_outputs in enumerate(outputs["dn_auxiliary_outputs"]):
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def RTDetrForObjectDetectionLoss(
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
    **kwargs,
):
    pass
