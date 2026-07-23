import time
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum

import torch

from ...utils import is_psutil_available, is_torch_xpu_available
from ...utils.logging import logging


if is_psutil_available():
    import psutil

TMP_TOKEN_ID = -1


logger = logging.getLogger("ContinuousBatchingLogger")
if logger.propagate:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False


def get_device_and_memory_breakdown() -> tuple[torch.device, int, int, int]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free_memory, total_memory = torch.cuda.mem_get_info(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = total_memory - free_memory
    elif is_torch_xpu_available():
        device = torch.device("xpu")
        torch.xpu.empty_cache()
        torch.xpu.synchronize()
        total_memory = torch.xpu.get_device_properties(device).total_memory
        reserved_memory = torch.xpu.memory_reserved(device)
        allocated_memory = torch.xpu.memory_allocated(device)
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        total_memory = torch.mps.recommended_max_memory()
        allocated_memory = torch.mps.current_allocated_memory()
        reserved_memory = torch.mps.driver_allocated_memory()
    else:
        device = torch.device("cpu")
        if is_psutil_available():
            total_memory = psutil.virtual_memory().total
            allocated_memory = psutil.Process().memory_info().rss
            reserved_memory = allocated_memory
        else:
            logger.error(
                "Cannot get memory breakdown on CPU without psutil: returning 0 for all memory values. Please install "
                "psutil to get an actual memory breakdown."
            )
            total_memory = 0
            reserved_memory = 0
            allocated_memory = 0

    return device, total_memory, reserved_memory, allocated_memory


class RequestStatus(IntEnum):

    PENDING = 0
    PREFILLING = 1
    DECODING = 2
    FINISHED = 3
    FAILED = 4


@dataclass
class GenerationOutput:

    request_id: str
    prompt_ids: list[int] = field(default_factory=list)
    generated_tokens: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    error: str | None = None
    status: RequestStatus = RequestStatus.PENDING
    created_time: float = field(default_factory=time.perf_counter)
    lifespan: tuple[float, float] = (-1, -1)  # (time request was no longer pending, time request finished)
    timestamps: list[float] | None = None  # Timestamps of the generated tokens

    def is_finished(self) -> bool:
        return self.status == RequestStatus.FINISHED


@dataclass
class RequestState:

    request_id: str
    initial_tokens: list[int]  # Initial prompt tokens # TODO: rename this as prefill tokens

    streaming: bool = False  # Whether to stream tokens as they're generated
    record_timestamps: bool = False  # Whether to record timestamps for the generated tokens

    max_new_tokens: int | None = 20  # Maximum number of new tokens to generate. None means no limit. Default to 20.
    eos_token_id: int | list[int] | None = None  # ID(s) of the end-of-sequence tokens. Only used in post-init.
    num_children: int = 0  # Number of children requests
    logit_processor_kwargs: dict = field(default_factory=dict)  # Keyword arguments for the logits processor.

    tokens_to_process: list[int] = field(default_factory=list)  # Tokens IDs currently being processed
    generated_tokens: list[int] = field(default_factory=list)  # Generated tokens
    logprobs: list[float] = field(default_factory=list)  # Log probabilities of the generated tokens
    position_offset: int = 0  # Current position in the sequence for position_ids
    allocated_blocks: int = 0  # Number of blocks allocated to the request

    _status: RequestStatus = RequestStatus.PENDING  # Status of the request, hidden behind a property
    _eos_token_ids: set[int] = field(default_factory=set)  # IDs of the end-of-sequence tokens, formatted as a set

    created_time: float = field(default_factory=time.perf_counter)  # Time the request was created
    error: str | None = None  # Error message if the request failed
    lifespan: tuple[float, float] = (-1, -1)  # (time request was no longer pending, time request finished)
    _timestamps: list[float] = field(default_factory=list)  # Timestamps of the generated tokens
    _true_initial_tokens: int = 0  # The true number of initial tokens, useful when soft resetting requests

    _new_tokens_limit: int = 2147483647  # An int to check the max number of new tokens w/out always comparing w/ None
    remaining_prefill_tokens: list[int] = field(default_factory=list)  # Initial tokens left to process
    is_cpu_offloaded: bool = False  # True when the request's KV cache is in the CPU swap pool

    def __post_init__(self):
        self._new_tokens_limit = 2147483647 if self.max_new_tokens is None else self.max_new_tokens
        self.remaining_prefill_tokens = self.initial_tokens[:]
        if self.eos_token_id is None:
            pass
        elif isinstance(self.eos_token_id, int):
            if self.eos_token_id >= 0:
                self._eos_token_ids.add(self.eos_token_id)
        else:
            for token_id in self.eos_token_id:
                if token_id >= 0:
                    self._eos_token_ids.add(token_id)

    @property
    def status(self) -> RequestStatus:
        pass

    @status.setter
    def status(self, value: RequestStatus):
        pass

    @property
    def timestamps(self) -> list[float] | None:
        pass

    def log_end_of_request(self):
        pass

    def current_len(self) -> int:
        pass

    def generated_len(self) -> int:
        """Get the number of tokens generated so far."""
        return len(self.generated_tokens)

    def update_and_check_completion(self, token_id: int, logprob: float | None) -> bool:
        pass

    def __repr__(self):
        msg = [
            f"request_id={self.request_id}",
            f"status={self._status}",
            f"out_tokens={self.generated_len()}",
            f"query_length={len(self.tokens_to_process)}",
            f"remaining_tokens={len(self.remaining_prefill_tokens)}",
            f"kv_length={self.position_offset}",
            f"full_prompt_length={len(self.initial_tokens)}",
            f"allocated_blocks={self.allocated_blocks}",
            f"generated_tokens={self.generated_tokens}",
            f"logit_processor_kwargs={self.logit_processor_kwargs}",
        ]
        return "RequestState(\n\t" + ",\n\t".join(msg) + "\n)"

    def to_generation_output(self):
        pass

    def fork(self, new_request_id: str) -> "RequestState":
        pass

    def get_request_config(self) -> dict:
        pass

    def create_equivalent_initial_request(self) -> "RequestState":
        pass


class FutureRequestState:

    __slots__ = ("state", "has_new_token", "complete_blocks", "query_length")

    def __init__(self, state: RequestState, has_new_token: bool, complete_blocks: int, query_length: int) -> None:
        self.state = state
        self.has_new_token = has_new_token
        self.complete_blocks = complete_blocks
        self.query_length = query_length
