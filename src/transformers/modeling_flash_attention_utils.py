import importlib
import inspect
import os
from collections.abc import Callable
from functools import partial
from typing import TypedDict

import torch
import torch.nn.functional as F

from .utils import (
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_flash_attn_4_available,
    is_rocm_platform,
    is_torch_cuda_available,
    is_torch_mlu_available,
    is_torch_npu_available,
    is_torch_xpu_available,
    logging,
)
from .utils.generic import split_attention_implementation
from .utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING, is_tracing


logger = logging.get_logger(__name__)


def flash_attn_supports_top_left_mask():
    if is_flash_attn_2_available() or is_flash_attn_3_available() or is_flash_attn_4_available():
        return False

    from .integrations.npu_flash_attention import is_npu_fa2_top_left_aligned_causal_mask

    return is_npu_fa2_top_left_aligned_causal_mask()


def is_flash_attn_available():
    return (
        is_flash_attn_4_available()
        or is_flash_attn_3_available()
        or is_flash_attn_2_available()
        or is_torch_npu_available()
        or is_torch_xpu_available()
    )



FLASH_ATTN_KERNEL_FALLBACK = {
    "flash_attention_2": "kernels-community/flash-attn2",
    "flash_attention_3": (
        "kernels-community/aiter-flash-attn" if is_rocm_platform() else "kernels-community/vllm-flash-attn3"
    ),
    "flash_attention_4": "kernels-community/flash-attn4",
}


FLASH_ATTENTION_COMPATIBILITY_MATRIX = {
    2: {
        "flash_attn_version": 2,
        "general_availability_check": is_flash_attn_2_available,
        "pkg_availability_check": lambda *args, **kwargs: (
            importlib.util.find_spec("flash_attn") is not None
            and "flash-attn" in [pkg.replace("_", "-") for pkg in PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn", [])]
        ),
        "supported_devices": (
            (is_torch_cuda_available, "cuda"),
            (is_torch_mlu_available, "mlu"),
            (is_torch_npu_available, "npu"),
            (is_torch_xpu_available, "xpu"),
        ),
        "custom_supported_devices": (
            (is_torch_npu_available, "Detect using FlashAttention2 on Ascend NPU."),
            (
                is_torch_xpu_available,
                f"Detect using FlashAttention2 (via kernel `{FLASH_ATTN_KERNEL_FALLBACK['flash_attention_2']}`) on XPU.",
            ),
        ),
    },
    3: {
        "flash_attn_version": 3,
        "general_availability_check": is_flash_attn_3_available,
        "pkg_availability_check": lambda *args, **kwargs: (
            importlib.util.find_spec("flash_attn_interface") is not None
            and "flash-attn-3"
            in [pkg.replace("_", "-") for pkg in PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn_interface", [])]
        ),
        "supported_devices": ((is_torch_cuda_available, "cuda"),),
        "cuda_min_major_version": 8,  # Ampere
    },
    4: {
        "flash_attn_version": 4,
        "general_availability_check": is_flash_attn_4_available,
        "pkg_availability_check": lambda *args, **kwargs: (
            importlib.util.find_spec("flash_attn") is not None
            and "flash-attn-4" in [pkg.replace("_", "-") for pkg in PACKAGE_DISTRIBUTION_MAPPING.get("flash_attn", [])]
        ),
        "supported_devices": ((is_torch_cuda_available, "cuda"),),
        "cuda_min_major_version": 9,  # Hopper
    },
}


_loaded_implementation = None
_flash_fn = None
_flash_varlen_fn = None
_flash_with_kvcache_fn = None
_pad_fn = None
_unpad_fn = None

_process_flash_kwargs_fn = None
_hf_api_to_flash_mapping = {
    "dropout": "dropout_p",
    "sliding_window": "window_size",
}
_flash_api_alternative_names = {"s_aux": "learnable_sink"}


def _lazy_imports(
    implementation: str | None, attention_wrapper: Callable | None = None, allow_all_kernels: bool = False
):
    """
    Lazy loads the respective flash attention implementations.

    Return:
        flash_attn_func: The base flash attention function.
        flash_attn_varlen_func: The flash attention function supporting variable sequence lengths,
                                e.g. for padding-free training.
        pad_input: The function to pad inputs into one sequence and returning the respective kwargs.
        unpad_input: The function to unpad outputs based on the kwargs (from pad_input).
    """
    is_fa2 = is_flash_attn_2_available()
    is_fa3 = is_flash_attn_3_available()
    is_fa4 = is_flash_attn_4_available()

    pad_input, unpad_input = _pad_input, _unpad_input

    is_paged, implementation = split_attention_implementation(implementation)

    if (implementation == "flash_attention_2" and is_fa2) or (
        implementation is None and is_fa2 and not is_fa3 and not is_fa4
    ):
        from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
        from flash_attn.bert_padding import pad_input, unpad_input
    elif is_torch_npu_available():
        from .integrations.npu_flash_attention import npu_flash_attn_func as flash_attn_func
        from .integrations.npu_flash_attention import npu_flash_attn_varlen_func as flash_attn_varlen_func
        from .integrations.npu_flash_attention import npu_flash_attn_with_kvcache as flash_attn_with_kvcache
    else:
        if implementation == "flash_attention_3" or (implementation is None and is_fa3 and not is_fa4):
            from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
        elif implementation == "flash_attention_4" or (implementation is None and is_fa4):
            from flash_attn.cute import flash_attn_func, flash_attn_varlen_func

            flash_attn_with_kvcache = None  # not supported yet
        else:
            from .integrations.hub_kernels import load_and_register_attn_kernel

            kernel_repo = FLASH_ATTN_KERNEL_FALLBACK.get(implementation, implementation)
            kernel_implementation = f"paged|{implementation}" if is_paged else kernel_repo
            kernel = load_and_register_attn_kernel(
                kernel_implementation, attention_wrapper, allow_all_kernels=allow_all_kernels
            )

            flash_attn_func = getattr(kernel, "flash_attn_func", None)
            flash_attn_varlen_func = getattr(kernel, "flash_attn_varlen_func", None)
            flash_attn_with_kvcache = getattr(kernel, "flash_attn_with_kvcache", None)
            if flash_attn_varlen_func is None and hasattr(kernel, "sparse_atten_func"):
                return flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache, pad_input, unpad_input
            if flash_attn_varlen_func is None:
                raise ValueError(
                    f"Could not find the currently requested flash attention implementation at `{implementation}`."
                    "Make sure that you request a valid kernel from the hub, e.g. `kernels-community/flash-attn2`."
                )
            if flash_attn_func is None:
                logger.warning(
                    f"The loaded flash attention implementation at `{implementation}` only supports varlen, i.e. "
                    "it can only be used with continuous batching and does not support the full functionality for "
                    "the base transformers generation methods."
                )
            if flash_attn_with_kvcache is None:
                logger.warning(
                    f"The loaded flash attention implementation at `{implementation}` does not support block tables, so"
                    " the full performances of continuous batching will not be achieved, only the varlen path will be "
                    "used."
                )

    return flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache, pad_input, unpad_input


def _lazy_define_process_function(flash_function):
    """
    Depending on the version and kernel some features are not supported. Due to limitations in
    `torch.compile`, we opt to statically type which (optional) kwarg parameters are supported
    within `_process_flash_attention_kwargs`.

    NOTE: While all supported kwargs are marked as `True`, everything else is marked as `False`.
          This might be confusing for kwargs that we use in any case, e.g. `is_causal`.
    """

    flash_parameters = inspect.signature(flash_function).parameters
    process_parameters = inspect.signature(_process_flash_attention_kwargs).parameters

    supports_mapping = {}
    for param in process_parameters:
        fa_param = _hf_api_to_flash_mapping.get(param, param)
        supports_mapping[fa_param] = fa_param in flash_parameters

        if (fa_alternative_name := _flash_api_alternative_names.get(param, param)) != fa_param:
            supports_mapping[fa_alternative_name] = fa_alternative_name in flash_parameters

    return partial(_process_flash_attention_kwargs, supports_mapping=supports_mapping)


def lazy_import_flash_attention(
    implementation: str | None, attention_wrapper: Callable | None = None, allow_all_kernels: bool = False
):
    """
    Lazily import flash attention and return the respective functions + flags.

    NOTE: For fullgraph, this needs to be called before compile, while no fullgraph can
    work without preloading. See `load_and_register_attn_kernel` in `integrations.hub_kernels`.
    """
    global _loaded_implementation
    if implementation is None and _loaded_implementation is None:
        raise ValueError("Could not find any flash attn implementation based on your environment.")

    global _flash_fn, _flash_varlen_fn, _flash_with_kvcache_fn, _pad_fn, _unpad_fn, _process_flash_kwargs_fn
    if implementation is not None and _loaded_implementation != implementation:
        _loaded_implementation = implementation

        _flash_fn, _flash_varlen_fn, _flash_with_kvcache_fn, _pad_fn, _unpad_fn = _lazy_imports(
            implementation, attention_wrapper, allow_all_kernels=allow_all_kernels
        )
        _process_flash_kwargs_fn = _lazy_define_process_function(_flash_varlen_fn) if _flash_varlen_fn else None

    return (_flash_fn, _flash_varlen_fn, _flash_with_kvcache_fn, _pad_fn, _unpad_fn), _process_flash_kwargs_fn


def lazy_import_paged_flash_attention(implementation: str | None, allow_all_kernels: bool = False):
    """
    Same as `lazy_import_flash_attention` but explicitly wrapping it with the paged implementation.
    """
    from .integrations.flash_paged import paged_attention_forward

    (_, flash_attn_varlen_func, flash_attn_with_kvcache_fn, _, _), _ = lazy_import_flash_attention(
        implementation, attention_wrapper=paged_attention_forward, allow_all_kernels=allow_all_kernels
    )
    return flash_attn_varlen_func, flash_attn_with_kvcache_fn


def _index_first_axis(tensor, indices):
    """
    A local implementation of the PyTorch indexing operation `tensor[indices]` on the first axis,
    after flattening the first two dimensions of the tensor. This is functionally equivalent to
    FA2's `index_first_axis` and replaces the need to import it.
    """
    reshaped_tensor = tensor.reshape(-1, *tensor.shape[2:])
    return reshaped_tensor[indices]


def _unpad_input(hidden_states, attention_mask, unused_mask=None):
    pass


def _pad_input(hidden_states, indices, batch, seqlen):
    pass


def _get_unpad_data(attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    unpad_input_func,
):
    """
    Unpads query, key, and values tensors, using a single dimension for all tokens even though they belong to different batches.
    This function is used instead of `flash_attn.bert_padding.unpad_input` in order to avoid the recomputation of the same intermediary
    tensors for query, key, value tensors.

    Arguments:
        query_layer (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.
        query_length (`int`):
            Target length.
        unpad_input_func:
            The function to use for unpadding the input tensors.

    Return:
        query_layer (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key_layer (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value_layer (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        indices_q (`torch.Tensor`):
            The indices of non-masked tokens from the flattened input target sequence.
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

    if key_layer.shape[1] > (seq_len := attention_mask.shape[-1]):
        key_layer, value_layer = key_layer[:, :seq_len, :, :], value_layer[:, :seq_len, :, :]

    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = _index_first_axis(key_layer, indices_k)
    value_layer = _index_first_axis(value_layer, indices_k)
    if query_length == kv_seq_len:
        query_layer = _index_first_axis(query_layer, indices_k)
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q, *_ = unpad_input_func(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def prepare_fa_kwargs_from_position_ids(position_ids):
    """
    This function returns all the necessary kwargs to call `flash_attn_varlen_func` extracted from position_ids.

    Arguments:
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into
            ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query,
            `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    tensor_kwargs = {"dtype": torch.int32, "device": position_ids.device}

    position_ids = position_ids.reshape(-1)
    indices_q = (position_ids == 0).nonzero().view(-1)

    cu_seq_lens_q = torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(position_ids.size(), **tensor_kwargs),
        )
    )
    cu_seq_lens_k = cu_seq_lens_q

    max_length_q = cu_seq_lens_q.diff().max()
    max_length_k = max_length_q

    return (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k)


def _prepare_from_posids(query, key, value, position_ids):
    """
    This function returns necessary arguments to call `flash_attn_varlen_func`.
    All three query, key, value states will be flattened.
    Cumulative lengths of each examples in the batch will be extracted from position_ids.
    NOTE: ideally cumulative lengths should be prepared at the data collator stage

    Arguments:
        query (`torch.Tensor`):
            Query state with padding. Shape: (batch_size, query_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (batch_size, kv_seq_len, num_key_value_heads, head_dim).
        position_ids (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        query (`torch.Tensor`):
            Query state without padding. Shape: (total_target_length, num_heads, head_dim).
        key (`torch.Tensor`):
            Key state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        value (`torch.Tensor`):
            Value state with padding. Shape: (total_source_length, num_key_value_heads, head_dim).
        (cu_seqlens_q, cu_seqlens_k) (`tuple[int]`):
            The cumulative sequence lengths for the target (query) and source (key, value), used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k) (`tuple[int]`):
            Maximum sequence length in batch (`max_seqlen_in_batch_q` for the target sequence i.e. query, `max_seqlen_in_batch_k` for the source sequence i.e. key/value).
    """
    query = query.contiguous().view(-1, query.size(-2), query.size(-1))
    key = key.contiguous().view(-1, key.size(-2), key.size(-1))
    value = value.contiguous().view(-1, value.size(-2), value.size(-1))

    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(position_ids)

    return (query, key, value, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k))


def _is_packed_sequence(position_ids, batch_size):
    """
    Check the position ids whether packed sequences are indicated or not
        1. Position ids exist
        2. Flattened sequences only are supported
        3. Compile-friendly `not (torch.diff(position_ids, dim=-1) >= 0).all()`, i.e. we have multiple increasing sequences
    """
    if position_ids is None:
        return False

    increasing_position_sequences = (
        torch.arange(position_ids.shape[1], device=position_ids.device) + position_ids.min()
    )
    return batch_size == 1 and (increasing_position_sequences - position_ids).abs().sum().bool()


def fa_peft_integration_check(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    target_dtype: torch.dtype | None = None,
):
    """
    PEFT usually casts the layer norms in float32 for training stability reasons
    therefore the input hidden states gets silently casted in float32. Hence, we need
    cast them back in float16 / bfloat16 just to be sure everything works as expected.
    This might slowdown training & inference so it is recommended to not cast the LayerNorms!
    """
    if target_dtype and q.dtype == torch.float32:
        logger.warning_once(f"Casting fp32 inputs back to {target_dtype} for flash-attn compatibility.")
        q, k, v = q.to(target_dtype), k.to(target_dtype), v.to(target_dtype)
    return q, k, v


class FlashAttentionKwargs(TypedDict, total=False):

    cu_seq_lens_q: torch.LongTensor | None
    cu_seq_lens_k: torch.LongTensor | None
    max_length_q: int | None
    max_length_k: int | None


def _process_flash_attention_kwargs(
    query_length: int,
    key_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    use_top_left_mask: bool = False,
    softcap: float | None = None,
    deterministic: bool | None = None,
    s_aux: torch.Tensor | None = None,
    max_seqlen_q: int | torch.IntTensor | None = None,
    max_seqlen_k: int | torch.IntTensor | None = None,
    supports_mapping: dict[str, bool] | None = None,
    **kwargs,
):
    pass


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    use_top_left_mask: bool = False,
    softcap: float | None = None,
    deterministic: bool | None = None,
    cu_seq_lens_q: torch.LongTensor | None = None,
    cu_seq_lens_k: torch.LongTensor | None = None,
    max_length_q: int | None = None,
    max_length_k: int | None = None,
    target_dtype: torch.dtype | None = None,
    attn_implementation: str | None = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    (Optional) kwargs are described further in `_process_flash_attention_kwargs` and `FlashAttentionKwargs`.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`, *optional*):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        attn_implementation (`str`, *optional*):
            The attention implementation to use. If None, will default to the one based on the environment.
    """
    (flash_fn, flash_varlen_fn, _, pad_fn, unpad_fn), process_flash_kwargs_fn = lazy_import_flash_attention(
        attn_implementation
    )

    query_states, key_states, value_states = fa_peft_integration_check(
        query_states, key_states, value_states, target_dtype
    )

    flash_kwargs = partial(
        process_flash_kwargs_fn,
        query_length=query_length,
        key_length=key_states.size(1),
        is_causal=is_causal,
        dropout=dropout,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        use_top_left_mask=use_top_left_mask,
        softcap=softcap,
        deterministic=deterministic,
        **kwargs,
    )

    is_fa_with_position_ids = _is_packed_sequence(position_ids, batch_size=query_states.size(0))
    is_fa_with_varlen_kwargs = all(
        kwarg is not None for kwarg in (cu_seq_lens_q, cu_seq_lens_k, max_length_q, max_length_k)
    )

    if attention_mask is not None:
        q, k, v, indices_q, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _upad_input(
            query_states, key_states, value_states, attention_mask, query_length, unpad_fn
        )

        if "mps" in str(q.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        out_unpad = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            **flash_kwargs(max_seqlen_q=max_length_q, max_seqlen_k=max_length_k),
        )
        if isinstance(out_unpad, tuple):
            out_unpad = out_unpad[0]

        out = pad_fn(out_unpad, indices_q, query_states.size(0), query_length)

    elif is_fa_with_varlen_kwargs or is_fa_with_position_ids:
        if cu_seq_lens_q is None or cu_seq_lens_k is None:
            q, k, v, (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = _prepare_from_posids(
                query_states, key_states, value_states, position_ids
            )
        else:
            q = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
            k = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
            v = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

        if "mps" in str(q.device):
            cu_seq_lens_k = cu_seq_lens_k.clone()

        out = flash_varlen_fn(
            q,
            k,
            v,
            cu_seqlens_q=cu_seq_lens_q,
            cu_seqlens_k=cu_seq_lens_k,
            **flash_kwargs(max_seqlen_q=max_length_q, max_seqlen_k=max_length_k),
        )
        if isinstance(out, tuple):
            out = out[0]

        out = out.view(query_states.size(0), -1, out.size(-2), out.size(-1))

    else:
        out = flash_fn(query_states, key_states, value_states, **flash_kwargs())
        if isinstance(out, tuple):
            out = out[0]

    return out
