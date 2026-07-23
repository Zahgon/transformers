# coding=utf-8

from typing import Optional, Union

import torch
from packaging import version

from ..utils import is_torch_flex_attn_available, logging
from ..utils.import_utils import (
    get_torch_version,
    is_torch_greater_or_equal,
    is_torch_less_or_equal,
    is_torchdynamo_compiling,
)


_TORCH_FLEX_USE_AUX = is_torch_greater_or_equal("2.9.0")


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

    if _TORCH_FLEX_USE_AUX:
        from torch.nn.attention.flex_attention import AuxRequest
    else:
        AuxRequest = None


logger = logging.get_logger(__name__)


class WrappedFlexAttention:

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self, training):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled or training != self.training:
            self.training = training
            if is_torch_less_or_equal("2.5.1"):
                self._compiled_flex_attention = torch.compile(flex_attention, dynamic=False)
            elif version.parse(get_torch_version()).base_version == "2.6.0" and training:
                self._compiled_flex_attention = torch.compile(
                    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs"
                )
            else:
                self._compiled_flex_attention = torch.compile(flex_attention)

            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def get_flex_attention_lse_kwargs(return_lse: bool) -> dict[str, bool | Optional["AuxRequest"]]:
    pass


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    training=False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    pass


Offset = torch.Tensor | int


def make_flex_block_causal_mask(
    attention_mask_2d: torch.Tensor,
    attention_chunk_size: int | None = None,
    query_length=None,
    key_length=None,
    offsets: tuple[Offset, Offset] | None = None,
    is_causal: bool | None = True,
) -> "BlockMask":
    pass


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def flex_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Union[torch.Tensor, "BlockMask"],
    scaling: float | None = None,
    softcap: float | None = None,
    s_aux: torch.Tensor | None = None,
    position_bias: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    pass
