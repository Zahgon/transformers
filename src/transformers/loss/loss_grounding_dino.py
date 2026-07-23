import torch
import torch.nn as nn

from ..image_transforms import center_to_corners_format
from ..utils import is_scipy_available
from .loss_for_object_detection import HungarianMatcher, ImageLoss, _set_aux_loss, generalized_box_iou


if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """
    Loss used in RetinaNet for dense detection: https://huggingface.co/papers/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        num_boxes (`int`):
            The total number of boxes in the batch.
        alpha (`float`, *optional*, defaults to 0.25):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to 2):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.sum() / num_boxes


class GroundingDinoHungarianMatcher(HungarianMatcher):
    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
                * "label_maps": Tuple of tensors of dim [num_classes, hidden_dim].
            targets (`list[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `list[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, hidden_dim]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        label_maps = outputs["label_maps"]

        label_maps = torch.cat([label_map[target["class_labels"]] for label_map, target in zip(label_maps, targets)])
        label_maps = label_maps / label_maps.sum(dim=-1, keepdim=True)

        target_bbox = torch.cat([v["boxes"] for v in targets])

        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = (pos_cost_class - neg_cost_class) @ label_maps.t()

        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class GroundingDinoImageLoss(ImageLoss):

    def __init__(self, matcher, focal_alpha, losses):
        nn.Module.__init__(self)
        self.matcher = matcher
        self.focal_alpha = focal_alpha
        self.losses = losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pass

    def _get_target_classes_one_hot(self, outputs, targets, indices):
        """
        Create one_hot based on the matching indices
        """
        logits = outputs["logits"]
        class_labels = torch.cat(
            [
                target["class_labels"][J] + len(outputs["label_maps"][i]) if i > 0 else target["class_labels"][J]
                for i, (target, (_, J)) in enumerate(zip(targets, indices))
            ]
        )
        label_maps = torch.cat(outputs["label_maps"], dim=0)

        idx = self._get_source_permutation_idx(indices)
        target_classes_onehot = torch.zeros_like(logits, device=logits.device, dtype=torch.long)
        target_classes_onehot[idx] = label_maps[class_labels].to(torch.long)

        return target_classes_onehot

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        if "text_mask" not in outputs:
            raise KeyError("No text_mask were found in the outputs")

        target_classes_onehot = self._get_target_classes_one_hot(outputs, targets, indices)
        source_logits = outputs["logits"]
        text_mask = outputs["text_mask"]

        source_logits = torch.masked_select(source_logits, text_mask)
        target_classes_onehot = torch.masked_select(target_classes_onehot, text_mask)

        target_classes_onehot = target_classes_onehot.float()
        loss_ce = sigmoid_focal_loss(
            inputs=source_logits,
            targets=target_classes_onehot,
            num_boxes=num_boxes,
            alpha=self.focal_alpha,
            gamma=2,
        )

        losses = {"loss_ce": loss_ce}

        return losses


def GroundingDinoForObjectDetectionLoss(
    logits,
    labels,
    device,
    pred_boxes,
    config,
    label_maps,
    text_mask,
    outputs_class=None,
    outputs_coord=None,
    encoder_logits=None,
    encoder_pred_boxes=None,
):
    pass
