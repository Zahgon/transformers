from collections.abc import Callable

import torch
import torch.nn.functional as F

from .cache_utils import Cache
from .configuration_utils import PreTrainedConfig
from .utils import is_torch_xpu_available, logging
from .utils.generic import GeneralInterface, is_flash_attention_requested
from .utils.import_utils import (
    is_torch_flex_attn_available,
    is_torch_greater_or_equal,
    is_tracing,
)


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE as flex_default_block_size
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask
else:
    BlockMask = torch.Tensor

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)
_is_torch_xpu_available = is_torch_xpu_available()

if _is_torch_greater_or_equal_than_2_6:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex


logger = logging.get_logger(__name__)


def and_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the intersection of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def and_mask(batch_idx, head_idx, q_idx, kv_idx):
        pass

    return and_mask


def or_masks(*mask_functions: Callable) -> Callable:
    """Returns a mask function that is the union of provided mask functions"""
    if not all(callable(arg) for arg in mask_functions):
        raise RuntimeError(f"All inputs should be callable mask_functions: {mask_functions}")

    def or_mask(batch_idx, head_idx, q_idx, kv_idx):
        pass

    return or_mask


def causal_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    pass


def bidirectional_mask_function(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
    pass


def sliding_window_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a sliding window pattern. Add it on top of a causal mask for a proper sliding
    window mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        pass

    return inner_mask


def chunked_overlay(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a chunked attention pattern. Add it on top of a causal mask for a proper chunked
    attention mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        pass

    return inner_mask


def blockwise_overlay(block_sequence_ids: torch.Tensor) -> Callable:
    """
    This is an overlay depicting a blockwise masking pattern. Instead of a single
    token, each block consists of arbitrary length tokens. In causal setup, each block
    can attend to prev block causally and can't attend to future blocks. Within one block
    the attention is always bidirectional.
    Mostly used in MLLMs when non-text data attends bidirectionally to itself.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        pass

    return inner_mask


def sliding_window_causal_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a sliding window mask.
    """
    return and_masks(sliding_window_overlay(sliding_window), causal_mask_function)


def sliding_window_bidirectional_overlay(sliding_window: int) -> Callable:
    """
    This is an overlay depicting a bidirectional sliding window pattern.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        pass

    return inner_mask


def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable:
    """
    This return the mask_function function to create a bidirectional sliding window mask.
    """
    return and_masks(sliding_window_bidirectional_overlay(sliding_window), bidirectional_mask_function)


def chunked_causal_mask_function(chunk_size: int, left_padding: torch.Tensor) -> Callable:
    """
    This return the mask_function function to create a chunked attention mask.
    """
    return and_masks(chunked_overlay(chunk_size, left_padding), causal_mask_function)


def padding_mask_function(padding_mask: torch.Tensor) -> Callable:
    pass


def packed_sequence_mask_function(packed_sequence_mask: torch.Tensor) -> Callable:
    """
    This return the mask_function function corresponding to a 2D packed sequence mask.
    """

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        pass

    return inner_mask


def add_offsets_to_mask_function(mask_function: Callable, q_offset: int, kv_offset: int) -> Callable:
    pass


def prepare_padding_mask(attention_mask: torch.Tensor | None, kv_length: int, kv_offset: int) -> torch.Tensor | None:
    pass


def maybe_pad_block_sequence_ids(
    block_sequence_ids: torch.Tensor, attention_mask: torch.Tensor | None, kv_length: int, kv_offset: int
) -> torch.Tensor:
    """
    Pads the `block_sequence_ids` in case the total length is less than `kv_length`.
    Usually that happens with `StaticCache` generation or generating without cache.
    Pads to the right with `-1`.
    """
    if (padding_length := kv_length + kv_offset - block_sequence_ids.shape[-1]) > 0:
        block_sequence_ids = F.pad(block_sequence_ids, pad=(0, padding_length), value=-1)
    return block_sequence_ids


def fast_all(tensor: torch.BoolTensor) -> torch.BoolTensor:
    pass


def _ignore_causal_mask_sdpa(
    padding_mask: torch.Tensor | None,
    q_length: int,
    kv_length: int,
    q_offset: int,
    kv_offset: int,
    local_attention_size: int | None = None,
) -> bool:
    pass


def _can_skip_bidirectional_mask_xpu(
    padding_mask: torch.Tensor | None,
    kv_length: int,
    local_attention_size: int | None,
) -> bool:
    pass


def _ignore_bidirectional_mask_sdpa(
    padding_mask: torch.Tensor | None,
    kv_length: int,
    local_attention_size: int | None = None,
) -> bool:
    pass


def _vmap_expansion_sdpa(mask_function: Callable) -> Callable:
    pass


def _non_vmap_expansion_sdpa(
    batch_indices: torch.Tensor, head_indices: torch.Tensor, q_indices: torch.Tensor, kv_indices: torch.Tensor
):
    """
    Used to broadcast our mask_functions over the all 4 dimensions (b_idx, h_idx, q_idx, kv_idx) of the inputs.
    Allows the usage of any index-based mask function without relying on vmap.

    NOTE: This is limited to index based functions only and is not guaranteed to work otherwise.

    Reference:
        - https://github.com/huggingface/optimum-onnx/blob/c123e8f4fab61b54a8e0e31ce74462bcacca576e/optimum/exporters/onnx/model_patcher.py#L362-L365
    """
    batch_indices = batch_indices[:, None, None, None]
    head_indices = head_indices[None, :, None, None]
    q_indices = q_indices[None, None, :, None]
    kv_indices = kv_indices[None, None, None, :]
    return batch_indices, head_indices, q_indices, kv_indices


def sdpa_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    local_size: int | None = None,
    allow_is_causal_skip: bool = True,
    allow_is_bidirectional_skip: bool = False,
    allow_torch_fix: bool = True,
    use_vmap: bool = False,
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.Tensor | None:
    pass


def eager_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    allow_is_bidirectional_skip: bool = False,
    use_vmap: bool = False,
    device: torch.device | str = "cpu",
    **kwargs,
) -> torch.Tensor:
    pass


def flash_attention_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    pass


def flex_attention_mask(
    batch_size: int,
    q_length: int,
    kv_length: int,
    q_offset: int = 0,
    kv_offset: int = 0,
    mask_function: Callable = causal_mask_function,
    attention_mask: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    **kwargs,
) -> BlockMask:
    pass


class AttentionMaskInterface(GeneralInterface):
    _global_mapping = {
        "sdpa": sdpa_mask,
        "eager": eager_mask,
        "flash_attention_2": flash_attention_mask,
        "flash_attention_3": flash_attention_mask,
        "flash_attention_4": flash_attention_mask,
        "flex_attention": flex_attention_mask,
    }


ALL_MASK_ATTENTION_FUNCTIONS: AttentionMaskInterface = AttentionMaskInterface()


def find_packed_sequence_indices(position_ids: torch.Tensor) -> torch.Tensor | None:
    """
    Find the indices of the sequence to which each new query token in the sequence belongs when using packed
    tensor format (i.e. several sequences packed in the same batch dimension).

    Args:
        position_ids (`torch.Tensor`)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.

    Returns:
        A 2D tensor where each similar integer indicates that the tokens belong to the same sequence. For example, if we
        pack 3 sequences of 2, 3 and 1 tokens respectively along a single batch dim, this will return [[0, 0, 1, 1, 1, 2]].

        If the there is only one sequence in each batch item (and we don't compile), then we return `None` indicating
        no packed sequences. This is the same as [[0, 0, 0, 0, 0, 0]] for the example above.
    """
    first_dummy_value = position_ids[:, :1] - 1  # We just need the diff on this first value to be 1
    position_diff = torch.diff(position_ids, prepend=first_dummy_value, dim=-1)
    packed_sequence_mask = (position_diff != 1).cumsum(-1)

    if not is_tracing(packed_sequence_mask) and (packed_sequence_mask[:, -1] == 0).all():
        return None

    return packed_sequence_mask


def _preprocess_mask_arguments(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    layer_idx: int | None,
    encoder_hidden_states: torch.Tensor | None = None,
) -> tuple[bool, torch.Tensor | BlockMask | None, int, int]:
    """
    Perform some common pre-processing of the mask arguments we get from the modeling code. Mostly determine the
    key-value length and offsets, and if we should early exit or not.

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        layer_idx (`int`, optional):
            If `past_key_values` is not None, this is the layer index of the cache from which to get the key-value
            length and offset. Indeed, for hybrid caches, different layers may return different lengths.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `inputs_embeds` to infer the kv length.

    Returns:
        early_exit (`bool`):
            Whether we should early exit mask creation, and return the mask as-is.
        attention_mask (`torch.Tensor` or `BlockMask` or `None`):
            The attention mask to either return immediately, or to use in downstream mask creation.
        packed_sequence_mask (`torch.Tensor`, optional):
            In case we detected packed sequence format, this is a tensor where each similar integer indicates that
            the tokens belong to the same sequence.
        q_length (`int`):
            The size that the query states will have during the attention computation.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        q_offset (`int`, optional):
            An optional offset to indicate at which first position the query states will refer to.
        kv_offset (`int`):
            An offset to indicate at which first position the key and values states will refer to.
    """
    if isinstance(attention_mask, (torch.Tensor, BlockMask)) and len(attention_mask.shape) == 4:
        return True, attention_mask, None, None, None, None, None

    if config._attn_implementation not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        return True, None, None, None, None, None, None

    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)

    q_length = inputs_embeds.shape[1]
    if past_key_values is not None:
        q_offset = past_key_values.get_query_offset(layer_idx)
        q_offset = q_offset.to(inputs_embeds.device) if isinstance(q_offset, torch.Tensor) else q_offset
        kv_length, kv_offset = past_key_values.get_mask_sizes(q_length, layer_idx)
    else:
        q_offset = 0
        if attention_mask is None:
            kv_length = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else q_length
            kv_offset = 0
        else:
            kv_length, kv_offset = attention_mask.shape[-1], 0

    packed_sequence_mask = None
    if position_ids is not None and attention_mask is None and past_key_values is None:
        batch_size = inputs_embeds.shape[0]
        if batch_size != position_ids.shape[0]:
            position_ids = position_ids.expand(batch_size, -1)
        packed_sequence_mask = find_packed_sequence_indices(position_ids)

    return False, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset


def create_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
    layer_idx: int | None = None,
) -> torch.Tensor | BlockMask | None:
    """
    Create a standard causal mask based on the attention implementation used (stored in the config). If `past_key_values`
    has an hybrid cache structure, this function will return the mask corresponding to one of the "full_attention" layers (to align
    to what is needed in the `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            Deprecated and unused.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        block_sequence_ids (`torch.Tensor`, *optional*):
            A tensor of same shape as input IDs indicating to which block or group each token belongs to. Tokens from
            the same block will keep a bidirectional mask within the block, attending causally to the past. Index `-1`
            can be used for blocks that have to keep complete causality within itself.
        layer_idx (`int`, *optional*):
            The cache layer to size the mask against. By default, the first "full_attention" layer is used, which is
            correct whenever all layers of a given mask type have seen the same tokens. Pass it explicitly for caches
            where layers of the same type hold different lengths (e.g. per-depth MTP streams).
    """
    if not getattr(config, "is_causal", True):
        return create_bidirectional_mask(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values=past_key_values,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )

    if layer_idx is None:
        if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
            layer_idx = past_key_values.is_sliding.index(False)
        else:
            layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    mask_factory_function = causal_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    use_vmap = False

    allow_is_causal_skip = not (getattr(past_key_values, "is_compileable", False) and q_length == 1)

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False
    if block_sequence_ids is not None:
        block_sequence_ids = maybe_pad_block_sequence_ids(block_sequence_ids, attention_mask, kv_length, kv_offset)
        mask_factory_function = or_masks(mask_factory_function, blockwise_overlay(block_sequence_ids))
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
        device=device,
    )
    return causal_mask


def create_bidirectional_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    **kwargs,
) -> torch.Tensor | BlockMask | None:
    """
    Create a standard bidirectional mask based on the attention implementation used (stored in the config).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is only used to infer metadata
            such as the batch size, query length, dtype, and device.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, kv_length).
            It can also be an already prepared 4D mask of shape (batch_size, 1, query_length, kv_length),
            in which case it is returned as-is.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `inputs_embeds` to infer the batch size, kv length and dtype.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the union of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
    """
    if hasattr(past_key_values, "is_sliding") and False in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(False)
    else:
        layer_idx = 0

    early_exit, attention_mask, _, q_length, kv_length, q_offset, kv_offset = _preprocess_mask_arguments(
        config, inputs_embeds, attention_mask, past_key_values, None, layer_idx, encoder_hidden_states
    )
    if early_exit:
        return attention_mask

    embeds = encoder_hidden_states if encoder_hidden_states is not None else inputs_embeds
    batch_size, dtype = embeds.shape[0], embeds.dtype
    device = inputs_embeds.device
    mask_factory_function = bidirectional_mask_function
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    allow_is_bidirectional_skip = True
    use_vmap = False

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True

    attention_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
        device=device,
    )
    return attention_mask


def create_sliding_window_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
    layer_idx: int | None = None,
) -> torch.Tensor | BlockMask | None:
    """
    Create a sliding window causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Mistral. If `past_key_values` has an hybrid cache structure, this
    function will return the mask corresponding to one of the "sliding_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            Deprecated and unused.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the sliding causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the sliding causal one, for example for image tokens handling.
        block_sequence_ids (`torch.Tensor`, *optional*):
            A tensor of same shape as input IDs indicating to which block or group each token belongs to. Tokens from
            the same block will keep a bidirectional mask within the block, attending causally to the past. Index `-1`
            can be used for blocks that have to keep complete causality within itself.
        layer_idx (`int`, *optional*):
            The cache layer to size the mask against. By default, the first "full_attention" layer is used, which is
            correct whenever all layers of a given mask type have seen the same tokens. Pass it explicitly for caches
            where layers of the same type hold different lengths (e.g. per-depth MTP streams).
    """
    if not getattr(config, "is_causal", True):
        return create_bidirectional_sliding_window_mask(
            config,
            inputs_embeds,
            attention_mask,
            past_key_values=past_key_values,
            or_mask_function=or_mask_function,
            and_mask_function=and_mask_function,
        )

    if layer_idx is None:
        if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
            layer_idx = past_key_values.is_sliding.index(True)
        else:
            layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    mask_factory_function = sliding_window_causal_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    use_vmap = False
    allow_is_causal_skip = not (getattr(past_key_values, "is_compileable", False) and q_length == 1)

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False
    if block_sequence_ids is not None:
        block_sequence_ids = maybe_pad_block_sequence_ids(block_sequence_ids, attention_mask, kv_length, kv_offset)
        mask_factory_function = or_masks(mask_factory_function, blockwise_overlay(block_sequence_ids))
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=sliding_window,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
        device=device,
    )
    return causal_mask


def create_bidirectional_sliding_window_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    encoder_hidden_states: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    **kwargs,
) -> torch.Tensor | BlockMask | None:
    """
    Create a standard bidirectional sliding window mask based on the attention implementation used (stored in the config).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is only used to infer metadata
            such as the batch size, query length, dtype, and device.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, kv_length).
            It can also be an already prepared 4D mask of shape (batch_size, 1, query_length, kv_length),
            in which case it is returned as-is.
        encoder_hidden_states (`torch.Tensor`, optional):
            The input embeddings of shape (batch_size, kv_length, hidden_dim). If provided, it is used instead of
            `inputs_embeds` to infer the batch size, kv length and dtype.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the union of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the base mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top, for example for image tokens handling.
    """
    if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
        layer_idx = past_key_values.is_sliding.index(True)
    else:
        layer_idx = 0

    early_exit, attention_mask, _, q_length, kv_length, q_offset, kv_offset = _preprocess_mask_arguments(
        config, inputs_embeds, attention_mask, past_key_values, None, layer_idx, encoder_hidden_states
    )
    if early_exit:
        return attention_mask

    sliding_window = getattr(config, "sliding_window", None)
    if sliding_window is None:
        raise ValueError("Could not find a `sliding_window` argument in the config, or it is not set")

    embeds = encoder_hidden_states if encoder_hidden_states is not None else inputs_embeds
    batch_size, dtype, device = embeds.shape[0], embeds.dtype, embeds.device
    mask_factory_function = sliding_window_bidirectional_mask_function(sliding_window)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    use_vmap = False
    allow_is_bidirectional_skip = True

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_bidirectional_skip = False
        use_vmap = True

    attention_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=False,
        allow_is_bidirectional_skip=allow_is_bidirectional_skip,
        local_size=sliding_window,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
        device=device,
    )
    return attention_mask


def create_chunked_causal_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    layer_idx: int | None = None,
) -> torch.Tensor | BlockMask | None:
    """
    Create a chunked attention causal mask based on the attention implementation used (stored in the config). This type
    of attention pattern was mostly democratized by Llama4. If `past_key_values` has an hybrid cache structure, this
    function will return the mask corresponding to one of the "chunked_attention" layers (to align to what is needed in the
    `modeling_xxx.py` files).

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            Deprecated and unused.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the chunked causal mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the chunked causal one, for example for image tokens handling.
        layer_idx (`int`, *optional*):
            The cache layer to size the mask against. By default, the first "full_attention" layer is used, which is
            correct whenever all layers of a given mask type have seen the same tokens. Pass it explicitly for caches
            where layers of the same type hold different lengths (e.g. per-depth MTP streams).
    """
    if layer_idx is None:
        if hasattr(past_key_values, "is_sliding") and True in past_key_values.is_sliding:
            layer_idx = past_key_values.is_sliding.index(True)
        else:
            layer_idx = 0

    early_exit, attention_mask, packed_sequence_mask, q_length, kv_length, q_offset, kv_offset = (
        _preprocess_mask_arguments(config, inputs_embeds, attention_mask, past_key_values, position_ids, layer_idx)
    )
    if early_exit:
        return attention_mask

    chunk_size = getattr(config, "attention_chunk_size", None)
    if chunk_size is None:
        raise ValueError("Could not find an `attention_chunk_size` argument in the config, or it is not set")

    if is_flash_attention_requested(config) and kv_length + kv_offset > chunk_size:
        raise ValueError(
            "Flash attention cannot handle chunked attention, and the key-value length is larger than the chunk size so the "
            "chunked pattern cannot be respected. You should use another `attn_implementation` when instantiating the model"
        )

    batch_size, dtype, device = inputs_embeds.shape[0], inputs_embeds.dtype, inputs_embeds.device
    if attention_mask is not None:
        left_padding_tokens = (attention_mask.cumsum(dim=-1) == torch.zeros_like(attention_mask)).sum(dim=-1)
    else:
        left_padding_tokens = torch.zeros(batch_size, device=device, dtype=int)
    mask_factory_function = chunked_causal_mask_function(chunk_size, left_padding_tokens)
    mask_interface = ALL_MASK_ATTENTION_FUNCTIONS[config._attn_implementation]

    use_vmap = False
    allow_is_causal_skip = not (getattr(past_key_values, "is_compileable", False) and q_length == 1)

    if or_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = or_masks(mask_factory_function, or_mask_function)
        allow_is_causal_skip = False
        use_vmap = True
    if and_mask_function is not None:
        if not _is_torch_greater_or_equal_than_2_6:
            raise ValueError("Using `or_mask_function` or `and_mask_function` arguments require torch>=2.6")
        mask_factory_function = and_masks(mask_factory_function, and_mask_function)
        allow_is_causal_skip = False
        use_vmap = True

    if packed_sequence_mask is not None:
        mask_factory_function = and_masks(mask_factory_function, packed_sequence_mask_function(packed_sequence_mask))
        allow_is_causal_skip = False

    causal_mask = mask_interface(
        batch_size=batch_size,
        q_length=q_length,
        kv_length=kv_length,
        q_offset=q_offset,
        kv_offset=kv_offset,
        mask_function=mask_factory_function,
        attention_mask=attention_mask,
        allow_is_causal_skip=allow_is_causal_skip,  # additional kwarg for sdpa
        local_size=chunk_size,  # Additional kwarg for sdpa
        dtype=dtype,  # Additional kwarg for eager
        config=config,  # Pass the config as well, in case someone wants to easily have their own mask_interface
        use_vmap=use_vmap,  # Short-circuit to non-vmap expansions for the mask
        device=device,
    )
    return causal_mask


def create_recurrent_attention_mask(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs,
) -> torch.Tensor | None:
    """Return the 2D padding mask for mamba / linear-attention layers, sized to the local sequence.

    Returns ``None`` (so the consumer skips masking entirely) when any of:
    - the input mask is missing or is already a custom 4D attention mask (no 2D padding signal);
    - the current forward is a single-token decode step (a generated token is never padding; this
      also keeps the growing 2D mask out of the compiled decode graph);
    - the mask is all-ones (un-padded batch — the masking multiply would be a no-op), skipped
      only outside trace/compile so the graph specialisation stays stable.

    Otherwise we trim the mask to the trailing ``inputs_embeds.shape[1]`` positions so it aligns
    with the current forward's local sequence and the consumer can multiply directly without
    further slicing. Note that this includes multi-token forwards continuing from a cache (chunked
    prefill, cache continuation): padding inside the new segment must still be zeroed out, otherwise
    it leaks into the recurrent state.
    """
    if attention_mask is None or attention_mask.ndim != 2:
        return None
    if inputs_embeds.shape[1] == 1:
        return None
    if not is_tracing(attention_mask) and torch.all(attention_mask == 1):
        return None
    return attention_mask[:, -inputs_embeds.shape[1] :].contiguous()


LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING = {
    "full_attention": create_causal_mask,
    "sliding_attention": create_sliding_window_causal_mask,
    "chunked_attention": create_chunked_causal_mask,
    "compressed_sparse_attention": create_sliding_window_causal_mask,
    "heavily_compressed_attention": create_sliding_window_causal_mask,
    "minimax_m3_sparse": create_causal_mask,
    "deepseek_sparse_attention": create_causal_mask,
    "linear_attention": create_recurrent_attention_mask,
    "conv": create_recurrent_attention_mask,
    "hybrid": {"full_attention": create_causal_mask, "linear_attention": create_recurrent_attention_mask},
    "hybrid_sliding": {"sliding_attention": create_causal_mask, "linear_attention": create_recurrent_attention_mask},
}


def create_masks_for_generate(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
    block_sequence_ids: torch.Tensor | None = None,
    **kwargs,
):
    """
    This function mimics how we create the masks in the `modeling_xxx.py` files, and is used in places like `generate`
    in order to easily create the masks in advance, when we compile the forwards with Static caches.

    Args:
        config (`PreTrainedConfig`):
            The model config.
        inputs_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        position_ids (`torch.Tensor`, optional)
            A 2D tensor of shape (batch_size, query_length) indicating the positions of each token in the sequences.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        block_sequence_ids (`torch.Tensor`, *optional*):
            A tensor of same shape as input IDs indicating to which block or group each token belongs to. Tokens from
            the same block will keep a bidirectional mask within the block, attending causally to the past. Index `-1`
            can be used for blocks that have to keep complete causality within itself.
    """
    effective_config = config.get_text_config()
    mask_kwargs = {
        "config": effective_config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "or_mask_function": or_mask_function,
        "and_mask_function": and_mask_function,
        "block_sequence_ids": block_sequence_ids,
    }

    if hasattr(effective_config, "layer_types"):
        layer_patterns = set(effective_config.layer_types)
        if any(layer_type not in LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING for layer_type in layer_patterns):
            return attention_mask
        causal_masks = {}
        for layer_pattern in layer_patterns:
            mask_function = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern]
            if isinstance(mask_function, dict):
                for actual_pattern, actual_function in mask_function.items():
                    if actual_pattern not in causal_masks:
                        causal_masks[actual_pattern] = actual_function(**mask_kwargs)
            else:
                causal_masks[layer_pattern] = mask_function(**mask_kwargs)
        return causal_masks
    elif getattr(effective_config, "sliding_window", None) is not None:
        return create_sliding_window_causal_mask(**mask_kwargs)
    elif getattr(effective_config, "attention_chunk_size", None) is not None:
        return create_chunked_causal_mask(**mask_kwargs)
    return create_causal_mask(**mask_kwargs)
