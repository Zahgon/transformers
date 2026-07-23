import torch

from ..utils import logging
from .sdpa_attention import sdpa_attention_forward


logger = logging.get_logger(__name__)

MSA_SUPPORTED_TOPK = (4, 8, 16, 32)
MSA_SUPPORTED_BLOCK_SIZE = 128
MSA_SUPPORTED_HEAD_DIM = 128

_MSA_KERNEL = None


def load_and_register_msa_kernel(attn_implementation: str):
    pass


@torch.library.custom_op("transformers_msa::sparse_atten", mutates_args=())
def _msa_sparse_atten_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    topk: int,
    block_size: int,
    total_k: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    qheads_per_kv: int,
    scaling: float,
    impl: str,
) -> torch.Tensor:
    pass


@_msa_sparse_atten_op.register_fake
def _msa_sparse_atten_fake(
    q,
    k,
    v,
    q2k,
    cu_seqlens_q,
    cu_seqlens_k,
    topk,
    block_size,
    total_k,
    max_seqlen_q,
    max_seqlen_k,
    qheads_per_kv,
    scaling,
    impl,
):
    pass


def _validate_msa_init(module, query: torch.Tensor, dropout: float) -> None:
    pass


def _sparse_attention(module, query, key, value, scaling, block_indices, block_size, cache_position):
    pass


def msa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    block_indices: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    pass
