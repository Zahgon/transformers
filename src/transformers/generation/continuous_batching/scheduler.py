import threading
from abc import ABC, abstractmethod
from collections import deque

from .cache import PagedAttentionCache
from .requests import FutureRequestState, RequestState, RequestStatus, logger


class Scheduler(ABC):

    def __init__(self, cache: PagedAttentionCache, safety_margin: float, max_requests_per_batch: int):
        """Initializes the scheduler. The safety margin is the percentage of free blocks under which we stop
        scheduling new prefill requests, so safety_margin = 0.1 means that when there is less than 10% of free blocks,
        or equivalently when more than 90% of blocks are already allocated, we stop scheduling new prefill requests.
        Setting safety_margin to 0.0 means no safety margin is applied."""
        self.cache = cache
        self.safety_margin = safety_margin
        self.max_requests_per_batch = max_requests_per_batch
        self._cancellation_lock = threading.Lock()
        if safety_margin < 0 or safety_margin > 1:
            raise ValueError(f"Got {safety_margin = } but expected a value in [0, 1]")
        if max_requests_per_batch < 1:
            raise ValueError(f"Got {max_requests_per_batch = } but expected a value >= 1")
        self.read_cache_limit = None if self.cache.num_full_attention_groups else self.cache.config.sliding_window
        self.max_decode_fast_path_length = self.cache.max_blocks_per_request * self.cache.block_size
        self.reset()

    def reset(self) -> None:
        """Reset scheduler state for a new generation loop."""
        self.active_requests: dict[str, RequestState] = {}
        self.waiting_requests: dict[str, RequestState] = {}
        self.waiting_requests_order: deque[str] = deque()
        self._requests_to_cancel: set[str] = set()
        self._requests_to_fork: list[RequestState] = []
        self.block_new_requests = False
        self.starved_requests: list[tuple[RequestState, int]] = []

    def add_waiting_request(self, state: RequestState):
        pass

    @abstractmethod
    def schedule_batch(
        self, token_budget: int, cache_budget: int
    ) -> tuple[list[FutureRequestState] | None, bool, int, int]:
        """Schedules requests for the next batch based on available token and cache budgets. This method selects which
        requests should be processed in the current batch, considering the budgets and the scheduler's prioritization
        rules. The token_budget is the maximum number of tokens that can be processed in a batch, and the cache_budget
        is the maximum number of KV cache entries that can be read in a batch.
        Returns the list of scheduled requests in their "FutureRequestState" form, a boolean indicating if the decode
        fast path can be used, the total number of query tokens and the maximum number of kv tokens read."""

    def has_pending_requests(self) -> bool:
        pass

    def finish_request(self, request_id: str) -> None:
        pass

    def get_active_request_static_outputs(self, request_id: str) -> list[int]:
        pass

    def set_request_cancellation(self, request_id: str):
        pass

    def clear_cancelled_requests(self) -> list[RequestState]:
        pass

    def request_is_cancelled(self, request_id: str) -> bool:
        pass

    def _allocate_blocks_if_needed(self, state: RequestState, len_next_tokens: int) -> bool:
        pass

    def _infer_request_tokens(self, state: RequestState, request_ids_to_remove_from_waiting: set[str]) -> list[int]:
        pass

    def _schedule_request(
        self,
        state: RequestState,
        request_tokens: list[int],
        token_budget: int,
        request_ids_to_remove_from_waiting: set[str],
    ) -> None:
        pass

    def _process_candidates(
        self,
        candidates: list[RequestState],
        token_budget: int,
        cache_budget: int,
        request_ids_to_remove_from_waiting: set[str],
    ) -> tuple[list[FutureRequestState], bool, bool, int, int]:
        pass

    def _get_waiting_candidates(self) -> list[RequestState]:
        pass

    def _cleanup_waiting_queue(self, request_ids_to_remove_from_waiting: set[str]) -> None:
        pass


class FIFOScheduler(Scheduler):

    def __init__(self, cache: PagedAttentionCache, safety_margin: float | None, max_requests_per_batch: int):
        """Initializes the FIFO scheduler, with a default safety margin of 0.15 (ie. 15% of free blocks)."""
        if safety_margin is None:
            safety_margin = 0.15
        super().__init__(cache, safety_margin, max_requests_per_batch)

    def schedule_batch(
        self, token_budget: int, cache_budget: int
    ) -> tuple[list[FutureRequestState] | None, bool, int, int]:
        pass


class PrefillFirstScheduler(Scheduler):

    def __init__(self, cache: PagedAttentionCache, safety_margin: float | None, max_requests_per_batch: int):
        """Initializes the prefill first scheduler, with a default safety margin of 0.0 (no safety margin)."""
        if safety_margin is None:
            safety_margin = 0.0
        super().__init__(cache, safety_margin, max_requests_per_batch)

    def schedule_batch(
        self, token_budget: int, cache_budget: int
    ) -> tuple[list[FutureRequestState] | None, bool, int, int]:
        pass


SCHEDULER_MAPPING = {
    "fifo": FIFOScheduler,
    "prefill_first": PrefillFirstScheduler,
}
