
import warnings
from dataclasses import dataclass
from typing import Union

import torch

from .utils.import_utils import is_torchdynamo_compiling, is_tracing


DEPRECATION_MESSAGE = (
    "The attention mask API under `transformers.modeling_attn_mask_utils` (`AttentionMaskConverter`) "
    "is deprecated and will be removed in Transformers v5.10. Please use the new API in `transformers.masking_utils`."
)


@dataclass
class AttentionMaskConverter:

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: int | None = None):
        warnings.warn(DEPRECATION_MESSAGE, FutureWarning)

        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: torch.dtype,
        device: Union[torch.device, "str"] = "cpu",
    ) -> torch.Tensor | None:
        pass

    def to_4d(
        self,
        attention_mask_2d: torch.Tensor,
        query_length: int,
        dtype: torch.dtype,
        key_value_length: int | None = None,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: int | None = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        warnings.warn(DEPRECATION_MESSAGE, FutureWarning)

        bsz, tgt_len = input_ids_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        mask = mask.to(dtype)

        if past_key_values_length > 0:
            mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

        if sliding_window is not None:
            diagonal = past_key_values_length - sliding_window - 1

            context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=diagonal)
            if is_torchdynamo_compiling():
                mask = mask.clone()
            mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)

    @staticmethod
    def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        warnings.warn(DEPRECATION_MESSAGE, FutureWarning)

        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

        inverted_mask = torch.tensor(1.0, dtype=dtype) - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    @staticmethod
    def _unmask_unattended(
        expanded_mask: torch.FloatTensor,
        min_dtype: float,
    ):
        pass

    @staticmethod
    def _ignore_causal_mask_sdpa(
        attention_mask: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
        sliding_window: int | None = None,
        is_training: bool = False,
    ) -> bool:
        pass


def _prepare_4d_causal_attention_mask(
    attention_mask: torch.Tensor | None,
    input_shape: torch.Size | tuple | list,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: int | None = None,
):
    pass


def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: torch.Tensor | None,
    input_shape: torch.Size | tuple | list,
    inputs_embeds: torch.Tensor,
    past_key_values_length: int,
    sliding_window: int | None = None,
):
    pass


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return AttentionMaskConverter._expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _prepare_4d_attention_mask_for_sdpa(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None):
    pass


def _create_4d_causal_attention_mask(
    input_shape: torch.Size | tuple | list,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    pass
