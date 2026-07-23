import inspect
from math import floor, gcd, sqrt
from typing import Any

import torch

from ...configuration_utils import PreTrainedConfig
from ...generation.configuration_utils import ContinuousBatchingConfig
from ...utils.generic import is_flash_attention_requested
from .cache_manager import BlockManager, CacheAllocator, FullAttentionCacheAllocator, SlidingAttentionCacheAllocator
from .distributed import DistributedHelper
from .initialization import resolve_max_memory_percent
from .requests import RequestState, RequestStatus, get_device_and_memory_breakdown, logger


def find_num_kv_heads(config: PreTrainedConfig) -> int:
    """Finds the number of key-value heads for the given config."""
    kv_heads = getattr(config, "num_key_value_heads", None)
    if kv_heads is not None:
        return kv_heads
    kv_heads = getattr(config, "num_attention_heads", None)
    if kv_heads is not None:
        return kv_heads
    raise ValueError(f"num_key_value_heads or num_attention_heads could not be found in the config:\n{config}")


def find_head_dim(config: PreTrainedConfig) -> int:
    """Finds the head dimension for the given config."""
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim
    hidden_size = getattr(config, "hidden_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    if hidden_size is not None and num_attention_heads is not None:
        return hidden_size // num_attention_heads
    raise ValueError(f"head_dim or (hidden_size and num_attention_heads) could not be found in the config:\n{config}")


def group_layers_by_attn_type(config: PreTrainedConfig) -> tuple[list[list[int]], list[str]]:
    """
    Group layers depending on the attention mix, according to VLLM's hybrid allocator rules:
        - Layers in each group need to have the same type of attention
        - All groups have the same number of layers

    For a model with the following layer types: ["sliding", "full", "full", "sliding", "full", "full", "full", "full"]
    We would get four groups: [0, 3], [1, 2], [4,5] and [6,7].
    """
    layer_types = getattr(config, "layer_types", None)
    if layer_types is None:
        attn_type = "sliding_attention" if getattr(config, "sliding_window", None) is not None else "full_attention"
        layer_types = [attn_type for _ in range(config.num_hidden_layers)]

    layer_counts = {}
    for i, layer_type in enumerate(layer_types):
        layer_counts[layer_type] = layer_counts.get(layer_type, []) + [i]

    group_size = gcd(*[len(indices) for indices in layer_counts.values()])

    layer_groups = []
    for layer_type, indices in layer_counts.items():
        for i in range(0, len(indices), group_size):
            layer_groups.append(indices[i : i + group_size])
    group_types = [layer_types[lg[0]] for lg in layer_groups]
    return layer_groups, group_types


class PagedAttentionCache:

    _min_block_size = 4

    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        device: torch.device | str,
        distributed_helper: DistributedHelper,
        tp_plan: dict[str, Any],
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize a paged attention cache for efficient memory usage. Also turns in prefix sharing if the model has
        only full attention layers.

        Args:
            config: Model configuration
            continuous_batching_config: Continuous batching configuration containing cache parameters
            device: Device for the cache tensors
            distributed_helper: TP-aware helper. Used to dispatch attention heads and ensure coherent cache size
            tp_plan: Tensor parallelism plan
            dtype: Data type of the activation and the cache (for now, these are the same)
        """
        self.config = config
        self.dtype = dtype
        self.device = device

        self.num_key_value_heads: int = find_num_kv_heads(config)
        self.head_dim: int = find_head_dim(config)

        self.block_size = continuous_batching_config.block_size
        if self.block_size < self._min_block_size:
            raise ValueError(f"Block size must be at least {self._min_block_size}, but got {self.block_size}")

        layer_groups, group_types = group_layers_by_attn_type(config)
        group_size = len(layer_groups[0])
        self.num_groups = len(layer_groups)

        self.sliding_windows = {}
        self.layer_index_to_group_indices = {}
        for i, group in enumerate(layer_groups):
            sliding_window = config.sliding_window if group_types[i] == "sliding_attention" else 1
            for j, layer in enumerate(group):
                self.layer_index_to_group_indices[layer] = (i, j)
                self.sliding_windows[layer] = sliding_window

        kv_is_tp = True
        for key in ["layers.*.self_attn.k_proj", "layers.*.self_attn.v_proj"]:
            if not (key in tp_plan or "model." + key in tp_plan):
                kv_is_tp = False
                break

        tp_size = distributed_helper.tp_size
        if tp_size > 1 and kv_is_tp:
            if self.num_key_value_heads % tp_size != 0:
                raise ValueError(
                    f"Number of key value heads {self.num_key_value_heads} must be divisible by tensor parallel size {tp_size}."
                )
            self.num_key_value_heads //= tp_size

        if continuous_batching_config.max_memory_percent is None:
            resolve_max_memory_percent(cb_config=continuous_batching_config, has_logit_processors=True)

        max_batch_tokens, num_blocks = PagedAttentionMemoryHandler(
            config=config,
            continuous_batching_config=continuous_batching_config,
            dtype=self.dtype,
            group_types=group_types,
            group_size=group_size,
        ).infer_max_batch_tokens_and_num_blocks()

        if tp_size > 1:
            sync = torch.tensor([max_batch_tokens, num_blocks], device=self.device, dtype=torch.int64)
            distributed_helper.tp_all_reduce_min(sync)
            max_batch_tokens, num_blocks = int(sync[0].item()), int(sync[1].item())

        self.max_batch_tokens = max_batch_tokens
        self.num_blocks = num_blocks
        self.num_pages = self.num_blocks * self.block_size
        logger.info(f"Paged cache initialized: {self.max_batch_tokens = }, {self.num_blocks = }, {self.block_size = }")

        max_blocks_per_request = continuous_batching_config.max_blocks_per_request
        if max_blocks_per_request is None:
            max_blocks_per_request = continuous_batching_config.fallback_max_blocks_per_request
        self.max_blocks_per_request = max_blocks_per_request

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []
        block_based_shape = (num_blocks + 2, self.block_size, self.num_key_value_heads, self.head_dim)

        self.cache_shape = ((num_blocks + 2) * self.block_size, self.num_key_value_heads, self.head_dim)
        self.read_trash_index = num_blocks * self.block_size
        self.sentinel_index = num_blocks * self.block_size + 1  # since block size >= 4 >= 2, this is safe
        self.write_trash_index = (num_blocks + 1) * self.block_size
        for _ in range(group_size):
            new_layer_key_cache = torch.empty(self.cache_shape, dtype=self.dtype, device=self.device)
            new_layer_value_cache = torch.empty(self.cache_shape, dtype=self.dtype, device=self.device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)
            new_layer_key_cache.view(block_based_shape)[num_blocks].fill_(0)
            new_layer_value_cache.view(block_based_shape)[num_blocks].fill_(0)
        logger.info(f"{self.cache_shape = } {self.key_cache[0].shape = } {self.key_cache[0].numel() = }")

        self.allow_block_sharing = continuous_batching_config.allow_block_sharing
        self.group_cache_managers: list[CacheAllocator] = []
        self.num_full_attention_groups = 0
        self.num_sliding_attention_groups = 0
        self.max_sliding_window_blocks_per_request = 0

        for i, group_type in enumerate(group_types):
            if group_type == "full_attention":
                cm = FullAttentionCacheAllocator(i, self.block_size, allow_block_sharing=self.allow_block_sharing)
                self.num_full_attention_groups += 1
            elif group_type == "sliding_attention":
                cm = SlidingAttentionCacheAllocator(
                    i, self.block_size, config.sliding_window, self.sentinel_index, self.write_trash_index
                )
                self.num_sliding_attention_groups += 1
                self.max_sliding_window_blocks_per_request = cm._max_blocks_per_request
            else:
                raise ValueError(f"Invalid group type: {group_type}")
            self.group_cache_managers.append(cm)

        self.use_prefix_sharing = self.allow_block_sharing and group_types == ["full_attention"]
        self._block_manager = BlockManager(num_blocks, self.block_size, tp_on=tp_size > 1)
        self._total_prefix_length: int = 0  # a counter to measure the impact of prefix sharing, also used in tests

        self._block_table_key = None

    def blocks_needed(self, num_requested_blocks: int, allocated_blocks: int) -> int:
        """Returns the number of physical blocks needed to allocate (num_requested_blocks) blocks to a request that
        already has (allocated_blocks) blocks. The number of newly allocated blocks needed is predicted by the
        following rules:
        - for full attention groups: since there is no sliding window for full attention layers, one requested block is
            always equivalent to one newly allocated block for EACH full attention group
        - for sliding window groups: because of the sliding window, the number of blocks allocated to a request is
            capped. Using the number of already (allocated_blocks) we can compute the number of new blocks to actually
            allocate to the request, which can be lower than the number of requested blocks. That number is the same for
            all sliding window groups, as only one sliding window size is supported.
        """
        needed_blocks = num_requested_blocks * self.num_full_attention_groups
        if self.num_sliding_attention_groups:
            blocks_left = max(self.max_sliding_window_blocks_per_request - allocated_blocks, 0)
            needed_blocks += min(blocks_left, num_requested_blocks) * self.num_sliding_attention_groups
        return needed_blocks

    def will_allocation_be_successful(self, num_requested_blocks: int, allocated_blocks: int) -> bool:
        """Returns a boolean indicating if the allocation of (num_requested_blocks) blocks will be successful."""
        return self.blocks_needed(num_requested_blocks, allocated_blocks) <= self.get_num_free_blocks()

    def blocks_in_use(self, request_id: str) -> int:
        pass

    def allocate_blocks(self, n_blocks: int, request_id: str, allocated_blocks: int) -> int | None:
        """Allocate cache blocks across all layer groups for a given request. Actual allocation is done by the cache
        managers, and this method only returns the maximum number of blocks actually allocated across all managers."""
        if not self.will_allocation_be_successful(n_blocks, allocated_blocks):
            return None
        max_allocated = 0
        for cm in self.group_cache_managers:
            num_allocated_blocks = cm.allocate_blocks(n_blocks, request_id, self._block_manager)
            if num_allocated_blocks is None:
                raise ValueError(f"Failed to allocate {n_blocks} blocks for request {request_id}")
            max_allocated = max(max_allocated, num_allocated_blocks)
        return max_allocated

    def free_blocks(self, request_id: str) -> None:
        """Free all allocated cache blocks for a given request across all layer groups. Actual deallocation is done
        by the cache managers."""
        for cm in self.group_cache_managers:
            cm.free_blocks(request_id, self._block_manager)

    def get_num_free_blocks(self) -> int:
        """Get the current number of unallocated blocks available for new requests."""
        return self._block_manager.num_free_blocks

    def extend_read_and_write_indices(
        self,
        request_id: str,
        past_length: int,
        query_length: int,
        read_index: list[list[int]] | None,
        write_index: list[list[int]],
    ) -> None:
        """Retrieve physical cache indices for reading KV states in the cache across all layer groups. This method
        coordinates with all cache managers to build the complete set of read indices needed for attention computation.
        When read_index is None, the batch has no cache reads and we only compute the write indices.
        """
        for cm, write_indices in zip(self.group_cache_managers, write_index):
            write_indices.extend(cm.get_write_indices(request_id, past_length, query_length))
        if read_index is not None:
            for cm, read_indices in zip(self.group_cache_managers, read_index):
                read_indices.extend(cm.get_read_indices(request_id, past_length, query_length))

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        for i, cm in enumerate(self.group_cache_managers):
            cm.fill_block_table(request_id, past_length, query_length, block_table[i])

    def get_seqlens_k(self, past_length: int, query_length: int) -> dict[str, int]:
        """Retrieve the key sequence length for the given request_id across all layer types. Returns a dictionary of
        layer types to their corresponding key sequence lengths."""
        seqlens_k = {}
        if self.num_full_attention_groups > 0:
            seqlens_k["full_attention"] = past_length + query_length
        if self.num_sliding_attention_groups > 0:
            seqlens_k["sliding_attention"] = query_length + min(past_length, self.config.sliding_window - 1)
        return seqlens_k

    def update(
        self,
        key_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_kv, head_dim]
        value_states: torch.Tensor,  # shape [1, num_kv_heads, seqlen_kv, head_dim]
        layer_idx: int,
        read_index: list[torch.Tensor],  # shape [num_layer_groups, seqlen_kv + past_length]
        write_index: list[torch.Tensor],  # shape [num_layer_groups, seqlen_q]
    ) -> tuple[torch.Tensor, torch.Tensor]:  # shape [seqlen_kv + past_length, num_kv_heads, head_dim]
        """Update the cache with new key-value states for a specific layer, and retrieves the relevant KV states from
        the cache for attention computation. The behavior differs based on the layer's attention type:

        - Full attention: New KV states are written to cache, then complete sequence is read from cache
        - Sliding window: Old KV is read from cache along with extra spaces for the new KV, then new KV is written to
            cache. This is because new KV might overwrite the old KV, so we need to read the old KV first.

        When the layer's read index is empty, the batch has no cache reads (all requests are non-chunked prefills): we
        only write to the cache and return the input KV states directly, skipping the index_select read-back.

        Returns the complete KV states (cached + new) for attention computation.
        """
        group_idx, layer_idx_in_group = self.layer_index_to_group_indices[layer_idx]
        layer_read_index = read_index[group_idx]
        layer_write_index = write_index[group_idx]
        k_cache = self.key_cache[layer_idx_in_group]
        v_cache = self.value_cache[layer_idx_in_group]
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)

        if layer_read_index.numel() == 0:
            k_cache.index_copy_(0, layer_write_index, key_states)
            v_cache.index_copy_(0, layer_write_index, value_states)
            return key_states, value_states

        sliding_window = self.sliding_windows[layer_idx]
        if sliding_window == 1:
            k_cache.index_copy_(0, layer_write_index, key_states)
            v_cache.index_copy_(0, layer_write_index, value_states)
            key_states_with_cache = torch.index_select(k_cache, 0, layer_read_index)
            value_states_with_cache = torch.index_select(v_cache, 0, layer_read_index)

        else:
            mask = (layer_read_index == self.sentinel_index).unsqueeze(-1).unsqueeze(-1)
            key_states_with_cache = torch.index_select(k_cache, 0, layer_read_index)
            key_states_with_cache.masked_scatter_(mask, key_states)
            value_states_with_cache = torch.index_select(v_cache, 0, layer_read_index)
            value_states_with_cache.masked_scatter_(mask, value_states)
            k_cache.index_copy_(0, layer_write_index, key_states)
            v_cache.index_copy_(0, layer_write_index, value_states)

        return key_states_with_cache, value_states_with_cache

    def get_block_table_key(self, flash_attn_with_kvcache_fn: Any) -> str:
        pass

    def search_prefix_match(self, request_id: str, prompt_ids: list[int]) -> int:
        pass

    def mark_shareable_blocks_as_complete(self, state: RequestState, num_complete_blocks: int) -> None:
        pass

    def copy_cache(self, list_source_blocks: list[int], list_forked_blocks: list[int]) -> None:
        pass

    def compute_max_num_forks(self, source_request_id: str) -> int:
        pass

    def fork_request(self, source_request_id: str, destination_request_ids: list[str]) -> tuple[list[int], list[int]]:
        pass

    def free_all_requests(self) -> None:
        """Free all blocks allocated to requests across all cache managers. This preserves prefix hashes in the block
        manager (blocks become initialized rather than uninitialized if they were complete), allowing prefix sharing
        to work across generation sessions."""
        all_request_ids = set()
        for cm in self.group_cache_managers:
            all_request_ids.update(cm.block_table.keys())
        for request_id in all_request_ids:
            self.free_blocks(request_id)


class PagedAttentionMemoryHandler:

    _min_max_batch_tokens = 256
    _default_max_batch_tokens = 8192

    def __init__(
        self,
        config: PreTrainedConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        dtype: torch.dtype,
        group_types: list[str],
        group_size: int,
    ) -> None:
        """Initialize the memory handler. Args:
        - config: the model configuration
        - continuous_batching_config: the continuous batching configuration
        - dtype: the data type of the activation and the cache
        - group_types: the list of all attention group types, formatted as strings
        - group_size: the size (in layers) of an attention group
        """
        self.config = config
        self.cb_config = continuous_batching_config
        self.cache_dtype = dtype
        self.activation_dtype = dtype
        self.block_size = continuous_batching_config.block_size
        self.page_size = find_head_dim(config) * find_num_kv_heads(config)
        self.num_groups = len(group_types)
        self.group_size = group_size

        if is_flash_attention_requested(self.config):
            self.num_attention_masks = 0
        else:
            self.num_attention_masks = 2 if "sliding_attention" in group_types else 1

        self.max_blocks_per_request = continuous_batching_config.max_blocks_per_request
        if self.max_blocks_per_request is None:
            self.max_blocks_per_request = continuous_batching_config.fallback_max_blocks_per_request
        self.num_output_rows = 2 if continuous_batching_config.return_logprobs else 1
        self.io_multiplier = 2 if continuous_batching_config.use_async_batching else 1
        self.available_memory = self.get_available_memory()

    @property
    def activation_peak(self) -> dict[str, tuple[int, ...]]:
        pass

    def get_available_memory(self) -> int:
        """Calculate available GPU memory for cache allocation in bytes, accouting for the maximum memory percent limit
        fixed by the continuous batching config."""
        _, total, reserved, allocated = get_device_and_memory_breakdown()
        available_memory = total - max(allocated, reserved)
        available_memory = int(available_memory * self.cb_config.max_memory_percent)
        logger.info(f"Memory available for cache allocation: {available_memory // 1024**2} MB")
        return available_memory

    def infer_max_batch_tokens_and_num_blocks(self) -> tuple[int, int]:
        """Infers max_batch_tokens and num_blocks based on the available memory and the size of the activation peaks.
        If neither value is provided, we use a default value of 8192 for max_batch_tokens, apply bounds depending on the
        available VRAM, and solve for num_blocks. If one value is provided, the other is found using a linear solve."""
        max_batch_tokens = self.cb_config.max_batch_tokens
        num_blocks = self.cb_config.num_blocks

        if max_batch_tokens is not None and num_blocks is not None:
            return self._check_footprint(max_batch_tokens, num_blocks)

        if max_batch_tokens is not None or num_blocks is not None:
            max_batch_tokens, num_blocks = self._solve_for_peaks(
                max_batch_tokens, num_blocks, cache_fill_per_batch=None
            )
            return self._check_footprint(max_batch_tokens, num_blocks)

        upper_bound_vram, _ = self._solve_for_peaks(
            max_batch_tokens=None,
            num_blocks=None,
            cache_fill_per_batch=0.1,  # each cache must fill 10% of the cache at most
        )
        max_batch_tokens = min(self._default_max_batch_tokens, upper_bound_vram)
        max_batch_tokens = max(max_batch_tokens, self._min_max_batch_tokens)
        max_batch_tokens, num_blocks = self._solve_for_peaks(max_batch_tokens, num_blocks, cache_fill_per_batch=None)
        return self._check_footprint(max_batch_tokens, num_blocks)

    def _solve_for_peaks(
        self,
        max_batch_tokens: int | None,
        num_blocks: int | None,
        cache_fill_per_batch: float | None,
    ) -> tuple[int, int]:
        """Returns max_batch_tokens and num_blocks so that their memory footprint is within the available memory for all
        activation peaks. If neither value is given, a value must be provided for cache_fill_per_batch: this means we
        solve for both varibles by saying each batch fill a certain percentage of the cache (eg, if cache_fill_per_batch
        is 0.01, each batch will fill 1% of the cache)."""
        solutions = []

        for peak_deltas in self.activation_peak.values():
            m, n = self._solve_for_peak(peak_deltas, max_batch_tokens, num_blocks, cache_fill_per_batch)
            solutions.append((m, n))

        final_m = min([solution[0] for solution in solutions])
        final_n = min([solution[1] for solution in solutions])
        return final_m, final_n

    def _solve_for_peak(
        self,
        peak: tuple[int, ...],
        max_batch_tokens: int | None,
        num_blocks: int | None,
        cache_fill_per_batch: float | None,
    ) -> tuple[int, int]:
        """Returns a couple of `(max_batch_tokens, num_blocks)` that satisfy the memory constraint for the given
        activation peak."""
        cm, cn, cmn, cmm = self._equation_coefficients(peak)

        if max_batch_tokens is None and num_blocks is None:
            if cache_fill_per_batch is None:
                raise ValueError("m must be provided if max_batch_tokens and num_blocks are None")
            m = cache_fill_per_batch  # as in, m is a substitute for big M, which is max_batch_tokens
            num_pages = self._solve_quadratic(cmn * m + cmm * m**2, cn + cm * m, -self.available_memory)
            max_batch_tokens = int(num_pages * m)
            num_blocks = int(num_pages) // self.block_size

        elif num_blocks is None:
            M = max_batch_tokens
            num_pages = floor((self.available_memory - cm * M - cmm * M**2) / (cn + cmn * M))
            num_blocks = num_pages // self.block_size

        elif max_batch_tokens is None:
            N = num_blocks * self.block_size
            max_batch_tokens = int(self._solve_quadratic(cmm, cm + cmn * N, cn * N - self.available_memory))

        return max_batch_tokens, num_blocks

    def _check_footprint(self, max_batch_tokens: int, num_blocks: int) -> tuple[int, int]:
        """Checks if the footprint of the cache is within the available memory."""
        memory_footprint = self.compute_memory_footprint(max_batch_tokens, num_blocks)
        if memory_footprint > self.available_memory:
            raise MemoryError(
                f"Memory footprint {memory_footprint} is more than available memory {self.available_memory}"
            )
        if max_batch_tokens <= 0 or num_blocks <= 0:
            raise ValueError(f"Invalid values: max_batch_tokens = {max_batch_tokens}, num_blocks = {num_blocks}")
        return max_batch_tokens, num_blocks

    def _solve_quadratic(self, a: float, b: float, c: float) -> int:
        """Largest positive root of a·x² + b·x + c = 0. Falls back to linear when a == 0. Rounded down."""
        if a == 0:
            return int(-c / b)
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            raise ValueError(f"No real solution (discriminant = {discriminant})")
        root = (-b + sqrt(discriminant)) / (2 * a)
        if root < 0:
            raise ValueError(f"No positive solution (root = {root})")
        return int(floor(root))

    def _equation_coefficients(self, peak_deltas: tuple[int, ...]) -> tuple[int, ...]:
        """Given some deltas corresponding to an activation peak, returns the coefficients for the memory polynomial of
        that peak. The memory polynomial is described in that class docstring."""
        delta_m, delta_n, delta_mm, delta_mn = peak_deltas

        i = torch.int32.itemsize             # size of int32 in bytes, used for index, input_ids, ...
        a = self.activation_dtype.itemsize             # for now, the cache and the activation have the same dtype
        c = self.cache_dtype.itemsize
        k = self.io_multiplier               # 1 sync, 2 async (IO tensors only)

        coeff_n = (
            delta_n                                      # activation peak: N-proportional part
            + 2 * self.group_size * self.page_size * c   # kv_cache: 2 * group_size * [N, page_size] * cache_dtype
            + k * self.num_groups * 8                    # read_index: [num_groups, N + M]  (N part only, int64)
        )
        coeff_m = (
            delta_m                                    # activation peak: M-proportional part
            + k * 7 * i                                # bulk_input: [7, M] int32, packed as 7 rows
            + k * self.num_output_rows * i             # output_ids: [num_output_rows, M] int32
            + k * self.num_groups                      # block_table: [bt_groups, M, max_blocks_per_req] int32
            * self.max_blocks_per_request * i          #   (zero when fast-decode is off)
            + k * self.num_groups * 8                  # write_index: [num_groups, M] int64
            + k * self.num_groups * 8                  # read_index: [num_groups, N + M] (M part only, int64)
        )
        coeff_mn = (
            delta_mn                             # activation peak: M·N-proportional part
            + k * self.num_attention_masks * a   # attention_mask: [1, 1, M, N + M] (N·M part only)
        )
        coeff_mm = (
            delta_mm                            # activation peak: M²-proportional part
            + k * self.num_attention_masks * a  # attention_mask: [1, 1, M, N + M] (M² part only)
        )

        return coeff_m, coeff_n, coeff_mn, coeff_mm

    def compute_memory_footprint(self, max_batch_tokens: int, num_blocks: int) -> int:
        """Evaluate the memory polynomial at concrete (N, M) values, taking the max across activation peaks."""
        M = max_batch_tokens
        N = num_blocks * self.block_size

        max_memory_footprint = 0
        for peak in self.activation_peak.values():
            cm, cn, cmn, cmm = self._equation_coefficients(peak)
            memory_footprint = cn * N + cm * M + cmn * N * M + cmm * M * M
            max_memory_footprint = max(max_memory_footprint, memory_footprint)
        return max_memory_footprint
