

import torch
import torch.nn.functional as F

from ..utils import is_vision_available
from .loss_d_fine import DFineLoss, _set_aux_loss, _set_aux_loss2
from .loss_for_object_detection import box_iou


if is_vision_available():
    from transformers.image_transforms import center_to_corners_format


class Deimv2Loss(DFineLoss):
    def __init__(self, config):
        super().__init__(config)
        self.weight_dict = {
            "loss_mal": config.weight_loss_mal,
            "loss_bbox": config.weight_loss_bbox,
            "loss_giou": config.weight_loss_giou,
            "loss_fgl": config.weight_loss_fgl,
            "loss_ddf": config.weight_loss_ddf,
        }
        self.losses = ["mal", "boxes", "local"]
        self.mal_alpha = config.mal_alpha
        self.use_dense_one_to_one = config.use_dense_one_to_one

    def loss_labels_mal(self, outputs, targets, indices, num_boxes):
        pass

    def _get_dense_o2o_indices(self, indices, indices_aux_list):
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for index in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(index, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx_pair in unique_sorted:
                row_idx, col_idx = idx_pair[0].item(), idx_pair[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=index.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=index.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "cardinality": self.loss_cardinality,
            "local": self.loss_local,
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "mal": self.loss_labels_mal,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

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
        if not self.use_dense_one_to_one:
            return super().forward(outputs, targets)

        outputs_without_aux = {k: v for k, v in outputs.items() if "auxiliary_outputs" not in k}
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        cached_indices = []
        indices_aux_list = []
        if "auxiliary_outputs" in outputs:
            for auxiliary_outputs in outputs["auxiliary_outputs"]:
                aux_indices = self.matcher(auxiliary_outputs, targets)
                cached_indices.append(aux_indices)
                indices_aux_list.append(aux_indices)

        indices_go = self._get_dense_o2o_indices(indices, indices_aux_list)
        num_boxes_go = sum(len(x[0]) for x in indices_go)
        num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes_go = torch.clamp(num_boxes_go, min=1).item()

        losses = {}
        for loss in self.losses:
            use_union = loss in ("boxes", "local")
            indices_in = indices_go if use_union else indices
            num_boxes_in = num_boxes_go if use_union else num_boxes
            l_dict = self.get_loss(loss, outputs, targets, indices_in, num_boxes_in)
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                for loss in self.losses:
                    use_union = loss in ("boxes", "local")
                    indices_in = indices_go if use_union else cached_indices[i]
                    num_boxes_in = num_boxes_go if use_union else num_boxes
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices_in, num_boxes_in)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if "dn_auxiliary_outputs" in outputs:
            if "denoising_meta_values" not in outputs:
                raise ValueError(
                    "The output must have the 'denoising_meta_values` key. "
                    "Please, ensure that 'outputs' includes a 'denoising_meta_values' entry."
                )
            dn_indices = self.get_cdn_matched_indices(outputs["denoising_meta_values"], targets)
            dn_num_boxes = num_boxes * outputs["denoising_meta_values"]["dn_num_group"]
            for i, auxiliary_outputs in enumerate(outputs["dn_auxiliary_outputs"]):
                for loss in self.losses:
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, dn_indices, dn_num_boxes)
                    l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


def Deimv2ForObjectDetectionLoss(
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
