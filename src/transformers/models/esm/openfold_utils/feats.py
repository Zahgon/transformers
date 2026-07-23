
from typing import overload

import torch
import torch.types
from torch import nn

from . import residue_constants as rc
from .rigid_utils import Rigid, Rotation
from .tensor_utils import batched_gather


@overload
def pseudo_beta_fn(aatype: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_masks: None) -> torch.Tensor: ...


@overload
def pseudo_beta_fn(
    aatype: torch.Tensor, all_atom_positions: torch.Tensor, all_atom_masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_masks):
    pass


def atom14_to_atom37(atom14: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pass


def build_template_angle_feat(template_feats: dict[str, torch.Tensor]) -> torch.Tensor:
    pass


def build_template_pair_feat(
    batch: dict[str, torch.Tensor],
    min_bin: torch.types.Number,
    max_bin: torch.types.Number,
    no_bins: int,
    use_unit_vector: bool = False,
    eps: float = 1e-20,
    inf: float = 1e8,
) -> torch.Tensor:
    pass


def build_extra_msa_feat(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pass


def torsion_angles_to_frames(
    r: Rigid,
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
) -> Rigid:
    default_4x4 = rrgdf[aatype, ...]

    default_r = r.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    alpha = torch.cat([bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2)


    all_rots = alpha.new_zeros(default_r.get_rots().get_rot_mats().shape)
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:] = alpha

    all_frames = default_r.compose(Rigid(Rotation(rot_mats=all_rots), None))

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = Rigid.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Rigid,
    aatype: torch.Tensor,
    default_frames: torch.Tensor,
    group_idx: torch.Tensor,
    atom_mask: torch.Tensor,
    lit_positions: torch.Tensor,
) -> torch.Tensor:
    group_mask = group_idx[aatype, ...]

    group_mask_one_hot: torch.LongTensor = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    t_atoms_to_global = r[..., None, :] * group_mask_one_hot

    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions
