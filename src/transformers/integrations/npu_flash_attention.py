
import math
import os

import torch

from ..utils.import_utils import is_torch_npu_available


if is_torch_npu_available():
    from torch_npu import npu_fusion_attention


TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE = 2
DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE = 3

SPARSE_MODE = int(os.getenv("NPU_FA2_SPARSE_MODE", default=DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE))
if SPARSE_MODE not in [TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE, DOWN_RIGHT_ALIGNED_CAUSAL_MASK_MODE]:
    raise ValueError(
        "Environment variable `NPU_FA2_SPARSE_MODE` can only be set as 2 (top-left aligned causal mask) "
        "or 3 (down-right aligned causal mask)."
    )

ATTN_MASK_NPU_CACHE = {}


def get_attn_mask_npu(device):
    pass


def is_npu_fa2_top_left_aligned_causal_mask():
    return SPARSE_MODE == TOP_LEFT_ALIGNED_CAUSAL_MASK_MODE if is_torch_npu_available() else False


def npu_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    pass


def npu_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q=None,  # defined for aligning params order with corresponding function in `flash-attn`
    max_seqlen_k=None,  # defined for aligning params order with corresponding function in `flash-attn`
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    pass


def npu_flash_attn_with_kvcache():
    raise NotImplementedError("npu_flash_attn_with_kvcache is not implemented")
