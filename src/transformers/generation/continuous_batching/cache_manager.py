import hashlib
from abc import ABC, abstractmethod
from array import array
from collections import deque
from collections.abc import Iterator
from math import ceil
from typing import TypeVar

import torch

from .requests import logger


T = TypeVar("T")


def reverse_enumerate(xs: list[T]) -> Iterator[tuple[int, T]]:
    pass


class Block:  # TODO: rename to ShareableBlock and update the docs

    def __init__(self, id_: int, parent_id: int | None, group_id: int) -> None:
        self.id: int = id_
        self.parent_id: int | None = parent_id
        self.group_id: int = group_id
        self.hash: int | None = None
        self.ref_count: int = 1

    def __repr__(self) -> str:
        return f"Block(id={self.id}, parent_id={self.parent_id}, group_id={self.group_id}, hash={self.hash}, ref_count={self.ref_count})"

    @property
    def is_complete(self) -> bool:
        pass


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int, tp_on: bool) -> None:
        """Initializes the block manager with a given number of blocks (num_blocks) of size (block_size)."""
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.tp_on = tp_on
        self._uninit_block_ids = deque(range(num_blocks))
        self._init_block_ids: dict[int, None] = {}  # effectively act as an ordered set
        self._hash_to_id: dict[int, int] = {}
        self._id_to_block: dict[int, Block] = {}

    @property
    def num_free_blocks(self) -> int:
        pass

    def has_enough_free_blocks(self, n_blocks: int) -> bool:
        """Checks if there are enough free blocks to allocate the requested number of blocks (n_blocks). If there are
        not enough uninitialized blocks, we uninitialize the required number of initialized blocks."""
        if len(self._uninit_block_ids) >= n_blocks:
            return True
        block_to_uninitialize = n_blocks - len(self._uninit_block_ids)
        if len(self._init_block_ids) < block_to_uninitialize:
            return False
        for _ in range(block_to_uninitialize):
            id_to_uninitialize = self._init_block_ids.popitem()[0]
            block = self._id_to_block[id_to_uninitialize]
            self._hash_to_id.pop(block.hash)  # ty:ignore[invalid-argument-type]
            self._uninit_block_ids.append(id_to_uninitialize)
        return True

    def get_free_blocks(
        self, n_blocks: int, last_block_id: int | None, shareable: bool, group_id: int
    ) -> list[int] | None:
        """Returns a list of (n_blocks) free block and mark them as no longer free in the internal data structures.
        If the (shareable) flag is set to True, a Block object is created to keep track of the block, with the
        (last_block_id) to indicate the last block id in the sequence, also named the parent block. If the manager
        cannot find enough free blocks, it returns None."""
        if not self.has_enough_free_blocks(n_blocks):
            return None
        allocated_block_ids = [self._uninit_block_ids.popleft() for _ in range(n_blocks)]
        if shareable:
            for block_id in allocated_block_ids:
                block = Block(block_id, last_block_id, group_id)
                self._id_to_block[block_id] = block
                last_block_id = block_id
        return allocated_block_ids

    def fork_blocks(
        self, parent_blocks: list[int], num_forks: int, shareable: bool, group_id: int
    ) -> tuple[list[list[int]] | None, list[int], list[int]]:
        pass

    def increase_ref_count(self, block_id: int) -> None:
        pass

    def decrease_ref_count(self, block_id: int) -> None:
        """Decreases the reference count of a given (block_id). If the reference count reaches 0, the block is no longer
        in use, and becomes initialized (if it was complete) or uninitialized (if it was incomplete)."""
        block = self._id_to_block[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            if block.is_complete:
                self._init_block_ids[block_id] = None
            else:
                self._id_to_block.pop(block_id)
                self._uninit_block_ids.append(block_id)

    def free_blocks(self, blocks: list[int], shareable: bool) -> None:
        """Marks a list of (blocks) as free. If the blocks were not (shareable), we simply add them to the uninitialized
        blocks queue. Otherwise, their new state depends on whether they are complete."""
        if shareable:
            for block_id in blocks:
                self.decrease_ref_count(block_id)
        else:
            self._uninit_block_ids.extend(blocks)

    def uninitialize_unshared_block(self, block_id: int) -> None:
        pass

    def mark_shareable_blocks_as_complete(
        self, num_complete_blocks: int, allocated_blocks: list[int], prompt_ids: list[int]
    ) -> None:
        pass

    def compute_hash(self, parent_hash: int | None, tokens: list[int], group_id: int) -> int:
        pass


class CacheAllocator(ABC):

    _index: int
    block_table: dict[str, list[int]]  # request_id -> list of block_ids allocated to the request
    uses_block_sharing: bool  # flag to determine if the blocks are shareable

    @abstractmethod
    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> int | None:
        """Allocates (n_blocks) for a given (request_id) using the (block_manager). Returns the num of blocks allocated
        if successful and None otherwise."""

    def free_blocks(self, request_id: str, block_manager: BlockManager) -> None:
        """Frees all blocks associated with a (request_id) using the (block_manager)."""
        if request_id in self.block_table:
            blocks_to_free = self.block_table.pop(request_id)
            block_manager.free_blocks(blocks_to_free, shareable=self.uses_block_sharing)
        else:
            logger.warning(
                f"CacheAllocator {self._index} attempted to free blocks for non-existent request_id: {request_id}"
            )

    @abstractmethod
    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache in the cache tensor."""

    @abstractmethod
    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to write request_id's cache in the cache tensor."""

    @abstractmethod
    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills the block table for a given request_id, past_length and query_length."""

    def fork_blocks(
        self, parent_request_id: str, children_request_ids: list[str], block_manager: BlockManager
    ) -> tuple[list[int], list[int]]:
        pass


class FullAttentionCacheAllocator(CacheAllocator):

    def __init__(self, index: int, block_size: int, allow_block_sharing: bool) -> None:
        """Initializes the cache manager for a group of full attention layers.
        Args:
            - index: the index of the associated layer group
            - block_size: the size of the blocks in the cache
        """
        self._index = index
        self.uses_block_sharing = allow_block_sharing
        self.block_size = block_size
        self.block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> int | None:
        """Allocate (n_blocks) for a given (request_id) using the (block_manager). Returns the number of blocks
        allocated if successful and None otherwise. For group of full attention layers, we always allocate the number of
        requested blocks."""
        block_table = self.block_table.get(request_id, [])
        if block_table:
            last_block_id = block_table[-1]
        else:
            self.block_table[request_id] = block_table  # TODO: check the impact of making this a deque
            last_block_id = None
        allocated_blocks = block_manager.get_free_blocks(n_blocks, last_block_id, self.uses_block_sharing, self._index)
        if allocated_blocks is None:
            return None
        block_table.extend(allocated_blocks)
        return n_blocks

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache. For a group of full attention layers, we
        first write the new cache to the cache tensor and then read the entire cache from the beginning to the end."""
        block_table = self.block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        total_length = past_length + query_length
        num_full_blocks = total_length // self.block_size
        remainder = total_length % self.block_size
        physical_indices = []
        for b in range(num_full_blocks):
            start = block_table[b] * self.block_size
            physical_indices.extend(range(start, start + self.block_size))
        if remainder:
            start = block_table[num_full_blocks] * self.block_size
            physical_indices.extend(range(start, start + remainder))
        return physical_indices

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices for writing to the cache. For a group of full attention layers, we write the new
        cache as a continuation of the existing cache for the same request."""
        block_table = self.block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        start_block = past_length // self.block_size
        start_offset = past_length % self.block_size
        end_pos = past_length + query_length
        end_block = (end_pos - 1) // self.block_size  # -1 because if end_pos == block_size, we still end on block 0
        physical_indices = []
        for b in range(start_block, end_block + 1):
            block_start = block_table[b] * self.block_size
            local_start = start_offset if b == start_block else 0
            local_end = (end_pos - 1) % self.block_size + 1 if b == end_block else self.block_size
            physical_indices.extend(range(block_start + local_start, block_start + local_end))
        return physical_indices

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        """Fills the block table for a given request_id, past_length and query_length."""
        request_blocks = self.block_table.get(request_id)
        if request_blocks is None:
            raise ValueError(f"No block table found for request {request_id}")
        total_length = past_length + query_length
        num_blocks_needed = (total_length + self.block_size - 1) // self.block_size
        block_table[:num_blocks_needed] = torch.tensor(
            request_blocks[:num_blocks_needed], device=block_table.device, dtype=block_table.dtype
        )


class SlidingAttentionCacheAllocator(CacheAllocator):

    def __init__(
        self, index: int, block_size: int, sliding_window: int, sentinel_index: int, write_trash_index: int
    ) -> None:
        """Initializes the cache manager for a group of sliding window attention layers, with two special indices:
        - ``sentinel_index`` marks the spot of a new token in the read indices
        - ``write_trash_index`` is used by padding tokens to write their KV cache
        """
        self._index = index
        self.uses_block_sharing = False
        self.block_size = block_size
        self.sliding_window = sliding_window
        self.sentinel_index = sentinel_index
        self.write_trash_index = write_trash_index
        self._max_blocks_per_request = ceil(self.sliding_window / self.block_size)
        self.block_table = {}

    def allocate_blocks(self, n_blocks: int, request_id: str, block_manager: BlockManager) -> int | None:
        """Allocate (n_blocks) for a given (request_id) using the (block_manager). Returns the number of blocks
        allocated otherwise. For group of sliding window attention layers, we only allocate up to the point where we can
        fit an entire sliding window in the cache tensor."""
        if request_id not in self.block_table:
            self.block_table[request_id] = []
        already_allocated = len(self.block_table[request_id])
        if already_allocated == self._max_blocks_per_request:
            return 0
        after_allocation = min(already_allocated + n_blocks, self._max_blocks_per_request)
        actual_n_blocks = after_allocation - already_allocated
        allocated_blocks = block_manager.get_free_blocks(
            actual_n_blocks, None, self.uses_block_sharing, self._index
        )  # no block sharing w/ sliding window
        if allocated_blocks is None:
            return None
        self.block_table[request_id].extend(allocated_blocks)
        return actual_n_blocks

    def get_read_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to read request_id's cache in the cache tensor.
        For a group of sliding window attention layers, we read from the cache tensor before writing on it, because the
        new cache can overwrite the old one. To form the cache + new key / values states, we read the at most
        sliding_window - 1 cache page and then manually add the new key / values states after. Hence the sentinel
        indices which indicate where to store the new key or values indices."""
        block_table = self.block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        start_index = 0 if past_length < self.sliding_window else past_length % self.sliding_window
        cache_length = min(past_length, self.sliding_window - 1)
        physical_indices = []
        for i in range(start_index, start_index + cache_length):
            i %= self.sliding_window
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            physical_index = block_table[block_idx] * self.block_size + block_offset
            physical_indices.append(physical_index)
        return physical_indices + [self.sentinel_index] * query_length

    def get_write_indices(self, request_id: str, past_length: int, query_length: int) -> list[int]:
        """Returns the physical indices of where to write request_id's cache in the cache tensor. For a group of
        sliding window attention layers, we write the new cache in rolling-buffer kind of way: if we reach the end of
        the allocated physical cache, we start writing from the beginning of the physical cache again."""
        block_table = self.block_table.get(request_id)
        if block_table is None:
            raise ValueError(f"No block table found for request {request_id}")
        start_index = past_length % self.sliding_window
        cache_length = min(query_length, self.sliding_window)
        padding_length = query_length - cache_length
        physical_indices = []
        for i in range(start_index, start_index + cache_length):
            i %= self.sliding_window
            block_idx = i // self.block_size
            block_offset = i % self.block_size
            physical_index = block_table[block_idx] * self.block_size + block_offset
            physical_indices.append(physical_index)
        if padding_length > 0:
            physical_indices = [self.write_trash_index] * padding_length + physical_indices
        return physical_indices

    def fill_block_table(
        self, request_id: str, past_length: int, query_length: int, block_table: torch.Tensor
    ) -> None:
        raise NotImplementedError("Sliding window attention layers do not support block table")
