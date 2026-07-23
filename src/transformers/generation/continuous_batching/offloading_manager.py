
import logging
from contextlib import nullcontext
from itertools import chain

import torch

from ...utils import is_psutil_available
from .cache import PagedAttentionCache
from .distributed import DistributedHelper
from .requests import FutureRequestState, RequestState, RequestStatus, logger
from .scheduler import Scheduler


def contiguous_runs(indices: list[int]) -> list[tuple[int, int, int]]:
    pass


class OffloadingManager:

    def __init__(
        self,
        cache: PagedAttentionCache,
        scheduler: Scheduler,
        cpu_offload_space_gib: float | None,
        safety_threshold: float,
        compute_stream: torch.cuda.Stream | None,
        distributed_helper: DistributedHelper,
    ) -> None:
        self.cache = cache
        self.scheduler = scheduler
        self._compute_stream = compute_stream

        self._cpu_key_cache: list[torch.Tensor] = []
        self._cpu_value_cache: list[torch.Tensor] = []
        self._gpu_key_views: list[torch.Tensor] = []
        self._gpu_value_views: list[torch.Tensor] = []
        self._free_cpu_blocks: list[int] = []
        self._request_id_to_cpu_blocks: dict[str, list[int]] = {}
        self._request_id_to_group_block_counts: dict[str, list[int]] = {}

        num_cpu_blocks = self._compute_num_cpu_blocks(cpu_offload_space_gib, safety_threshold)
        num_cpu_blocks = torch.tensor(num_cpu_blocks, dtype=torch.int32, device="cpu")
        self._num_cpu_blocks = int(distributed_helper.tp_all_reduce_min(num_cpu_blocks, on_cpu=True).item())

        offloading_enabled = cpu_offload_space_gib is not None and cpu_offload_space_gib > 0
        if self._num_cpu_blocks == 0:
            if offloading_enabled:
                logger.warning(
                    f"cpu_offload_space={cpu_offload_space_gib:.1f} GiB is too small for even one block. "
                    "No CPU offloading."
                )
            return None

        cpu_cache_shape = (self._num_cpu_blocks, cache.block_size, cache.num_key_value_heads, cache.head_dim)
        for _ in cache.key_cache:
            self._cpu_key_cache.append(torch.empty(cpu_cache_shape, dtype=cache.dtype, pin_memory=True))
            self._cpu_value_cache.append(torch.empty(cpu_cache_shape, dtype=cache.dtype, pin_memory=True))

        block_shape = (-1, cache.block_size, cache.num_key_value_heads, cache.head_dim)
        for k_cache, v_cache in zip(cache.key_cache, cache.value_cache):
            self._gpu_key_views.append(k_cache.view(*block_shape))
            self._gpu_value_views.append(v_cache.view(*block_shape))

        self._free_cpu_blocks = list(range(self._num_cpu_blocks))

        cache_tensor = self._cpu_key_cache[0]
        size_in_bytes = 2 * cache_tensor.numel() * cache_tensor.element_size() * len(cache.key_cache)
        logger.info(
            f"CPU swap pool initialized: {self._num_cpu_blocks} blocks ({size_in_bytes / (1024**3):.2f} GiB pinned)"
        )

    def _compute_num_cpu_blocks(self, cpu_offload_space_gib: float | None, safety_threshold: float) -> int:
        """Returns the number of blocks that can fit in the CPU swap pool."""
        offload_bytes = int(cpu_offload_space_gib * (1024**3)) if cpu_offload_space_gib is not None else None

        if is_psutil_available():
            import psutil

            total_ram = psutil.virtual_memory().available
            max_bytes = int(total_ram * safety_threshold)
        else:
            max_bytes = None

        if offload_bytes is not None and max_bytes is not None:
            if offload_bytes > max_bytes:
                clamped_gib = max_bytes / (1024**3)
                logger.warning(
                    f"cpu_offload_space={cpu_offload_space_gib:.1f} GiB exceeds {safety_threshold:.0%} of total RAM "
                    f"({total_ram / (1024**3):.1f} GiB). Clamping to {clamped_gib:.1f} GiB."
                )
                offload_bytes = max_bytes
        elif offload_bytes is not None:
            logger.warning(
                "psutil is not available — cpu_offload_space_safety_threshold cannot be enforced. "
                "Install psutil to enable the safety cap."
            )
        elif max_bytes is not None:
            offload_bytes = max_bytes
            logger.warning(f"Auto-sizing CPU swap pool from safety threshold: {max_bytes / (1024**3):.2f} GiB.")
        else:
            raise ImportError(
                "cpu_offload_space=None requires psutil to auto-size the CPU swap pool. Install psutil or pass an "
                "explicit GiB value."
            )

        bytes_per_block = (
            2                                 # one for key, one for value
            * len(self.cache.key_cache)       # number of layers in a layer group
            * self.cache.block_size           # block size
            * self.cache.num_key_value_heads  # number of key value heads
            * self.cache.head_dim             # head dimension
            * self.cache.dtype.itemsize       # data type size in bytes
        )  # fmt: skip
        if bytes_per_block == 0:
            raise ValueError("The number of bytes per block is 0. This is not possible.")
        return offload_bytes // bytes_per_block

    def _stream_ctx(self):
        pass

    def offload_requests(self) -> int:
        pass

    def restore_scheduled_requests(self, requests_in_batch: list[FutureRequestState]) -> None:
        pass

    def free_request_cpu_cache(self, state: RequestState, keep_unsorted: bool = False) -> None:
        """Free CPU blocks for a single request (e.g., on cancellation)."""
        if state.is_cpu_offloaded:
            self._return_cpu_blocks(state.request_id)
            state.is_cpu_offloaded = False
            if not keep_unsorted:
                self._free_cpu_blocks.sort()

    def free_all_waiting_cpu_caches(self) -> None:
        """Free all CPU-offloaded caches in the waiting queue (e.g., on fail_all or reset)."""
        for state in self.scheduler.waiting_requests.values():
            self.free_request_cpu_cache(state, keep_unsorted=True)
        self._free_cpu_blocks.sort()

    def reset(self) -> None:
        """Reset CPU offloading state for a new generation session."""
        self.free_all_waiting_cpu_caches()
        self._request_id_to_cpu_blocks.clear()
        self._request_id_to_group_block_counts.clear()
        self._free_cpu_blocks = list(range(self._num_cpu_blocks))

    def _offload_to_cpu(self, victims: list[RequestState]) -> set[str]:
        pass

    def _return_cpu_blocks(self, request_id: str) -> tuple[list[int], list[int]]:
        """Return CPU blocks to the free pool without copying anything."""
        cpu_ids = self._request_id_to_cpu_blocks.pop(request_id)
        group_counts = self._request_id_to_group_block_counts.pop(request_id)
        self._free_cpu_blocks.extend(cpu_ids)
        return cpu_ids, group_counts
