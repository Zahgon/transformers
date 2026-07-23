import torch

from ..generation.continuous_batching import PagedAttentionCache
from ..modeling_flash_attention_utils import lazy_import_paged_flash_attention


def paged_attention_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,  # Unused in flash
    cache: PagedAttentionCache,
    cu_seq_lens_q: torch.Tensor,
    cu_seq_lens_k: torch.Tensor | dict[str, torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int | dict[str, int],
    block_table: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    pass


@torch.compiler.disable
def _paged_decode_forward(
    module: torch.nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cache: PagedAttentionCache,
    cu_seq_lens_k: torch.Tensor,
    sliding_window: tuple[int, int],
    flash_attn_with_kvcache,
    block_table: torch.Tensor,
    **flash_kwargs,
) -> torch.Tensor:
    pass
