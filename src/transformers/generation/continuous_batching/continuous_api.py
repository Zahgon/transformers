import asyncio
import gc
import queue
import threading
from abc import abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from time import perf_counter
from typing import Any

import torch
from torch import nn
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ...configuration_utils import PretrainedConfig
from ...generation.configuration_utils import ContinuousBatchingConfig, GenerationConfig
from ...utils.generic import is_flash_attention_requested
from ...utils.import_utils import is_flash_attn_2_available, is_flash_attn_3_available
from ...utils.logging import logging
from ..logits_process import LogitsProcessorList
from .cache import PagedAttentionCache
from .cb_logits_processors import ContinuousBatchingLogitsProcessorList
from .distributed import DistributedHelper
from .initialization import resolve_continuous_batching_config, update_cb_config_after_cache_creation
from .input_outputs import ContinuousBatchingAsyncIOs, ContinuousBatchingIOs
from .model_runner import ModelRunner
from .offloading_manager import OffloadingManager
from .requests import GenerationOutput, RequestState, RequestStatus, logger
from .scheduler import SCHEDULER_MAPPING, FIFOScheduler, Scheduler
from .utils import WorkloadHints, drain_queue


"""
To enable cuda graphs, we need the dimensions of all tensors to be static, which is counter-intuitive for CB. In CB, as
generation goes on, there are two dimensions that change:
- the number of queries tokens (Q), which can vary from batch to batch
- the number of keys/values tokens (KV), which grows as the cache does

To solve this, we slice along those dimensions to fixed lengths. The size of the slices is controlled by interval sizes:
- q_padding_interval_size: the padding granularity for queries (in tokens)
- kv_padding_interval_size: the padding granularity for KV cache (in tokens)

For example, with q_padding_interval_size=64 and an actual query length of 100, we pad to 128 tokens.

Smaller intervals mean finer granularity and thus less padding, but more unique graph signatures. Since graphs take
memory and time to create, we use an LRU cache with a fixed size to limit memory usage. Good defaults:
- Q: 64 tokens gives ~4 graphs for max_batch_tokens=256, which is a good balance
- KV: 8192 tokens (256 blocks at block_size=32) gives reasonable granularity for large caches

All defaults are stored in ContinuousBatchingConfig.resolve_sentinel_values().
"""


class ProtoPretrainedModel(nn.Module):
    config: PretrainedConfig
    dtype: torch.dtype
    device: torch.device

    @abstractmethod
    def set_attn_implementation(self, attn_implementation: str) -> None:
        pass

    @abstractmethod
    def _get_logits_processor(self, generation_config: GenerationConfig) -> LogitsProcessorList:
        pass


class OutputRouter:

    def __init__(self) -> None:
        self.output_queue = queue.Queue()
        self.result_handlers: dict[str, tuple[Callable, asyncio.AbstractEventLoop]] = {}
        self._lock = threading.Lock()

    def deliver(self, output: GenerationOutput) -> None:
        pass

    def deliver_batch(self, outputs: list[GenerationOutput]) -> None:
        pass


class BackgroundThreadStatus:

    DONT_STOP = 0
    FLUSH_AND_STOP = 1
    HARD_STOP = 2
    STOPPED = 3

    def __init__(self) -> None:
        self._local_status_lock = threading.Lock()
        self._local_status = self.DONT_STOP
        self._tp_status = self.DONT_STOP

    def clear(self) -> None:
        """Clear the local and TP statuses. This method should ONLY be called by the main thread itself BEFORE starting
        the background thread."""
        self._tp_status = self.DONT_STOP
        with self._local_status_lock:
            self._local_status = self.DONT_STOP

    def request_stop(self, status: int, global_rank: int) -> None:
        """Request the background thread to stop. This does not take effect immediately, only after the TP group has
        communicated."""
        if status not in [self.FLUSH_AND_STOP, self.HARD_STOP]:
            raise ValueError(f"Invalid stop status {status} from rank {global_rank}")
        with self._local_status_lock:
            self._local_status = max(status, self._local_status, self._tp_status)
        logger.info(
            f"Rank {global_rank} requested background thread to stop with {status = }. Now {self._local_status = }"
        )

    def mark_as_stopped(self) -> None:
        pass

    def update_with_tp_status(self, tp_status: int) -> None:
        pass

    @property
    def local_status(self) -> int:
        pass

    @property
    def tp_status(self) -> int:
        pass


class ContinuousBatchProcessor:
    inputs_and_outputs: ContinuousBatchingIOs | ContinuousBatchingAsyncIOs
    scheduler: Scheduler

    def __init__(
        self,
        cache: PagedAttentionCache,
        config: PretrainedConfig,
        generation_config: GenerationConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        logit_processor: ContinuousBatchingLogitsProcessorList,
        input_queue: queue.Queue | None,
        cancel_queue: queue.Queue | None,
        output_router: OutputRouter,
        background_thread_status: BackgroundThreadStatus,
        model_device: torch.device,
        model_dtype: torch.dtype,
        scheduler: Scheduler,
        distributed_helper: DistributedHelper,
    ) -> None:
        """Initialize the continuous batch processor.

        Args:
            cache: A [`PagedAttentionCache`] object
            config: The model configuration
            generation_config: The generation configuration
            continuous_batching_config: The continuous batching configuration
            logit_processor: The [`ContinuousBatchingLogitsProcessorList`] object used to process the logits.
            input_queue: Queue for incoming requests. Is None if this process is not a TP driver.
            cancel_queue: Queue for cancellation request_ids. Is None if this process is not a TP driver.
            output_router: An [`OutputRouter`] object that routes outputs to handlers or the output queue.
            background_thread_status: A [`BackgroundThreadStatus`] object to track the background thread status.
            model_device: Device for model inputs/outputs
            model_dtype: Data type for model inputs/outputs
            scheduler: The [`Scheduler`] to use
            distributed_helper: The [`DistributedHelper`] to use
        """
        self.cache = cache
        self.config = config
        self.cb_config = continuous_batching_config
        self.logit_processor = logit_processor
        self.input_queue = input_queue
        self.cancel_queue = cancel_queue
        self.output_router = output_router
        self.background_thread_status = background_thread_status
        self.model_device = model_device
        self.model_dtype = model_dtype
        self.scheduler = scheduler
        self.distributed_helper = distributed_helper

        self.do_sample = getattr(generation_config, "do_sample", True)
        self.return_logprobs = continuous_batching_config.return_logprobs

        self.distributed_helper.set_tp_seed(continuous_batching_config.seed, model_device)

        self.sliding_window = 1 if getattr(config, "sliding_window", None) is None else config.sliding_window

        self.max_batch_tokens = cache.max_batch_tokens

        io_kwargs = {
            "cache": cache,
            "config": config,
            "continuous_batching_config": continuous_batching_config,
            "device": model_device,
            "model_dtype": model_dtype,
            "logit_processor": self.logit_processor,
        }
        self.use_async_batching = self.cb_config.use_async_batching

        if self.use_async_batching:
            self.inputs_and_outputs = ContinuousBatchingAsyncIOs(**io_kwargs)
        else:
            self.inputs_and_outputs = ContinuousBatchingIOs(**io_kwargs)

        self.offloading_manager = OffloadingManager(
            cache=cache,
            scheduler=scheduler,
            cpu_offload_space_gib=continuous_batching_config.cpu_offload_space,
            safety_threshold=continuous_batching_config.cpu_offload_space_safety_threshold,
            compute_stream=self.inputs_and_outputs.compute_stream,
            distributed_helper=self.distributed_helper,
        )

        self.model_runner = ModelRunner(
            logit_processor=self.logit_processor,
            cb_config=self.cb_config,
            cache=self.cache,
            inputs_and_outputs=self.inputs_and_outputs,
            do_sample=self.do_sample,
            return_logprobs=self.return_logprobs,
        )

    def __repr__(self) -> str:
        return (
            f"ContinuousBatchProcessor(input_queue={self.input_queue}, "
            f"active_requests={self.scheduler.active_requests}, waiting_requests={self.scheduler.waiting_requests})"
            + self.inputs_and_outputs.get_model_kwargs().__repr__()
        )

    def __del__(self) -> None:
        self.inputs_and_outputs = None  # clean up CUDA graphs in priority
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self) -> None:
        """Reset the batch processor for a new generation loop."""
        self.offloading_manager.reset()
        self.scheduler.reset()
        self.inputs_and_outputs.reset()
        self.cache.free_all_requests()

    def _update_tp_group_state(self) -> bool:
        pass

    def _handle_request_error(self, error: Exception, state: RequestState) -> None:
        pass

    def prepare_next_batch(self) -> bool:
        pass

    def update_batch(self) -> None:
        pass

    def has_pending_requests(self) -> bool:
        pass

    def handle_batch_error(self, error):
        pass

    def fail_all_requests(self, error: Exception) -> None:
        pass

    @torch.no_grad()
    def _generation_step(self, model: nn.Module) -> None:
        pass

    @torch.no_grad()
    def warmup(self, model: nn.Module) -> None:
        """Pre-capture CUDA graphs (or trigger compile warmup) for varlen and decode paths. In async mode, both IO
        pairs are warmed up since each has its own graph buffer and static tensors. The varlen path is warmed up at
        the largest possible `(q, kv)` sizes so subsequent captures fit inside it without growing the pool."""
        self.model_runner.warmup(model)


class ContinuousBatchingManager:

    def __init__(
        self,
        model: ProtoPretrainedModel,
        generation_config: GenerationConfig,
        continuous_batching_config: ContinuousBatchingConfig,
        workload_hints: WorkloadHints | None = None,
    ) -> None:
        """Initialize the continuous batching manager.

        Args:
            model: The language model for generation
            generation_config: Configuration for generation parameters
            continuous_batching_config: Configuration for continuous batching parameters
            workload_hints: Workload hints for the continuous batching initialization (optional)
        """
        self.input_queue = queue.Queue(maxsize=continuous_batching_config.max_queue_size)
        self.cancel_queue: queue.Queue[str] = queue.Queue()
        self._request_counter = 0
        self._request_lock = threading.Lock()
        self._has_new_requests = threading.Event()

        self.background_thread_status = BackgroundThreadStatus()
        self.output_router = OutputRouter()
        self.batch_processor: ContinuousBatchProcessor | None = None
        self._generation_thread = None

        self.fatal_error: Exception | None = None
        self.warmed_up = False  # Set to True after warmup is completed. Useful for persistent managers.

        self._original_attn_impl = None  # needs to be set before the model is switched to paged attention
        self.switch_to_cb_friendly_attn(model)
        self.model = model.eval()

        self.generation_config = generation_config
        num_return_sequences = getattr(generation_config, "num_return_sequences", None)
        self.num_return_sequences = num_return_sequences if num_return_sequences is not None else 1

        self.distributed_helper = DistributedHelper(
            device_mesh=getattr(self.model, "_device_mesh", None),
            cpu_group_timeout=continuous_batching_config.cpu_group_timeout,
        )
        self.is_tp_driver = self.distributed_helper.is_tp_driver
        if continuous_batching_config.disable_nccl_graph_mixing:
            self.distributed_helper.maybe_warn_nccl_graph_mixing()

        self.logit_processor = ContinuousBatchingLogitsProcessorList(
            logits_processor=self.model._get_logits_processor(generation_config),
            per_request_processors=continuous_batching_config.per_request_processors,
            drop_unsupported_processors=continuous_batching_config.drop_unsupported_processors,
        )

        self.continuous_batching_config = resolve_continuous_batching_config(
            config=self.model.config,
            cb_config=continuous_batching_config,
            workload_hints=workload_hints,
            has_logit_processors=self.logit_processor.do_processing,
        )
        self._use_prefix_sharing = self.continuous_batching_config.allow_block_sharing

    def switch_to_cb_friendly_attn(self, model: ProtoPretrainedModel) -> None:
        """Switch the attn implementation to one that is CB friendly: try to find a flash implementation if flash is
        requested and, in any cases, switch to a paged implementation."""
        original_attn_impl = model.config._attn_implementation
        target_implem = original_attn_impl

        is_flash = is_flash_attention_requested(requested_attention_implementation=target_implem)
        is_paged = "paged|" in target_implem
        if not is_flash and not is_paged and model._supports_flash_attn:
            if is_flash_attn_3_available(kernels_fallback_ok=True):
                version = 3
            elif is_flash_attn_2_available(kernels_fallback_ok=True):
                version = 2
            else:
                version = None
            msg = "Continuous batching is much better when using flash attention."
            if version is not None:
                target_implem = f"flash_attention_{version}"  # no "paged|" prefix here to enter the branch below
                logger.warning(
                    f"{msg} Switching from {original_attn_impl} to {target_implem}. "
                    "If you need to use eager or sdpa, use paged|eager or paged|sdpa as the `attn_implementation`."
                )
            else:
                logger.warning(f"{msg} Consider using a flash `attn_implementation` when loading the model.")

        if "paged|" not in target_implem:
            model.set_attn_implementation(f"paged|{target_implem}")
            self._original_attn_impl = original_attn_impl

    def warmup(self) -> None:
        """Pre-capture CUDA graphs for varlen and decode paths by running dummy batches. Initializes the batch
        processor if not already done."""
        if self.batch_processor is None:
            self.batch_processor = self._create_batch_processor()
        self.batch_processor.warmup(self.model)
        self.warmed_up = True


    def is_running(self) -> bool:
        """Returns True if the background generation thread has been started and is still alive."""
        return self._generation_thread is not None and self._generation_thread.is_alive()

    def start(self) -> None:
        """Start the background generation thread."""
        if self.is_running():
            logger.warning("Manager thread is already running.")
            return None
        self.background_thread_status.clear()
        self.fatal_error = None
        self._generation_thread = threading.Thread(target=self._run_generation_loop)
        self._generation_thread.start()

    def stop(
        self,
        block: bool = True,
        timeout: float | None = None,
        keep_for_next_session: bool = False,
        hard_stop: bool = False,
    ) -> None:
        """Stop the background generation thread. If the `block` flag is set to True, then this method waits for the
        thread to stop for a maximum time of `timeout` seconds (None means no timeout). If the `keep_for_next_session`
        flag is set to True, then the manager is cached on the model for future use. If the `hard_stop` flag is set,
        the background generation thread will be stopped immediately and pending requests will be failed."""

        if self.batch_processor is None:
            logger.warning("\nBatch processor was not initialized.")

        if self._generation_thread is None:
            msg = "Manager not started."
            if keep_for_next_session:
                msg += " Hence the unstarted manager will not be kept for next session."
            logger.warning(msg)
            return None

        stop_trigger_time = perf_counter()
        stop_status = BackgroundThreadStatus.HARD_STOP if hard_stop else BackgroundThreadStatus.FLUSH_AND_STOP
        self.background_thread_status.request_stop(stop_status, self.distributed_helper.global_rank)
        if block:
            self.join(stop_trigger_time, timeout)

        if not keep_for_next_session:
            self.batch_processor = None
        else:
            logger.info("Continuous batching manager will be kept for next session.")
            self.model._cached_continuous_batching_manager = self

        if self._original_attn_impl is not None:
            self.model.set_attn_implementation(self._original_attn_impl)
            self._original_attn_impl = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def join(self, stop_trigger_time: float, timeout: float | None = None) -> None:
        """Wait for the background thread to finish. Wait can be capped using the timeout argument (in seconds)."""
        if self._generation_thread is None:
            return None
        self._generation_thread.join(timeout=timeout)
        if self._generation_thread.is_alive():
            logger.warning(f"Generation thread did not exit after join timeout ({timeout}).")
        else:
            end = perf_counter()
            logger.info(f"Background generation thread stopped after {end - stop_trigger_time:.2f}s.")
            self._generation_thread = None

    def destroy(self) -> None:
        """Terminate the manager and release distributed resources. Safe to call multiple times. After calling this,
        the manager cannot be restarted."""
        if self.is_running():
            self.stop(block=True, keep_for_next_session=False)
        self.distributed_helper.destroy_cpu_comm_group()


    def add_request(
        self,
        input_ids: list[int],
        request_id: str | None = None,
        max_new_tokens: int | None = None,
        streaming: bool = False,
        record_timestamps: bool = False,
        eos_token_id: int | list[int] | None = None,
        **logit_processor_kwargs: Any,
    ) -> str | None:
        """Add a new generation request to the queue. If the process is not a TP driver, this is a no-op.

        Args:
            input_ids: Input token IDs to use as prompt
            request_id: Optional custom request ID (auto-generated if None)
            max_new_tokens: Maximum number of new tokens to generate
            streaming: Whether to stream tokens as they're generated
            record_timestamps: Whether to record timestamps for each generated token
            eos_token_id: End-of-sequence token ID(s)
            logit_processor_kwargs: Keyword arguments for the logits processor.

        Returns:
            str | None: The request ID if the process is a TP driver, None otherwise.
        """
        if not self.is_tp_driver:
            return None
        if self.background_thread_status.local_status >= BackgroundThreadStatus.FLUSH_AND_STOP:
            preview = f"{input_ids[:3]}"[:-1] + ", ..., " + f"{input_ids[-3:]}"[1:]
            logger.warning(f"Background thread is stopping. Request with ids {preview} will be dropped.")
            return None

        if request_id is None:
            with self._request_lock:
                request_id = f"req_{self._request_counter}"
                self._request_counter += 1
        max_new_tokens = self.generation_config.max_new_tokens if max_new_tokens is None else max_new_tokens
        eos_token_id = self.generation_config.eos_token_id if eos_token_id is None else eos_token_id

        state = RequestState(
            request_id=request_id,
            initial_tokens=list(input_ids),
            num_children=self.num_return_sequences - 1,
            record_timestamps=record_timestamps,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            streaming=streaming,
            logit_processor_kwargs=logit_processor_kwargs,
        )

        self.input_queue.put(state, block=True, timeout=10)
        self._has_new_requests.set()
        return request_id

    def add_requests(
        self,
        inputs: list[list[int]],
        max_new_tokens: int | None = None,
        streaming: bool = False,
        record_timestamps: bool = False,
        **logit_processor_kwargs: Any,
    ) -> list[str]:
        """Utility function to batch `add_request` and return their IDs. Check its documentation for more details."""
        num_requests = len(inputs)
        with self._request_lock:
            request_ids = [f"req_{i}" for i in range(self._request_counter, self._request_counter + num_requests)]
            self._request_counter += num_requests
        ids_and_inputs = list(zip(request_ids, inputs))
        if self._use_prefix_sharing:
            ids_and_inputs = sorted(ids_and_inputs, key=lambda x: x[1], reverse=True)
        eos_token_id = self.generation_config.eos_token_id
        eos_token_id = self.model.config.eos_token_id if eos_token_id is None else eos_token_id
        eos_token_id = -1 if eos_token_id is None else eos_token_id
        for request_id, input_ids in ids_and_inputs:
            self.add_request(
                input_ids=input_ids,
                request_id=request_id,
                max_new_tokens=max_new_tokens,
                streaming=streaming,
                record_timestamps=record_timestamps,
                eos_token_id=eos_token_id,
                **logit_processor_kwargs,
            )
        return request_ids

    def cancel_request(self, request_id: str) -> None:
        """Cancel a request by its ID. If this called from a process that is not a TP driver, it's a no-op: only TP
        driver processes interact with the manager."""
        if self.is_tp_driver:
            self.cancel_queue.put(request_id)
            self._has_new_requests.set()

    def get_result(self, request_id: str | None = None, timeout: float | None = None) -> GenerationOutput | None:
        """Retrieve one result from the output queue. If an ID is provided, returns the first matching request. If a
        timeout is provided, returns None after the timeout (in seconds)."""
        if self._generation_thread is None and self.output_router.output_queue.empty():
            return None
        try:
            result = self.output_router.output_queue.get(block=True, timeout=timeout)
            if request_id is not None and result.request_id != request_id:
                self.output_router.output_queue.put(result)
                return None
            return result
        except queue.Empty:
            return None

    def __iter__(self):
        """Iterate over results as they become available."""
        while self._generation_thread is not None and self._generation_thread.is_alive():
            result = self.get_result(timeout=0.1)
            if result is not None:
                yield result

    def request_id_iter(self, request_id: str) -> Generator[GenerationOutput]:
        pass

    def register_result_handler(self, request_id: str, callback: Callable) -> None:
        """Register a callback for result delivery (streaming or non-streaming).

        The callback is invoked on the event loop via ``call_soon_threadsafe`` each time a result is produced for this
        request. For streaming requests, this happens on every token; for non-streaming, only on completion. The handler
        is automatically cleaned up when the request finishes.

        Args:
            request_id (`str`): The request ID to receive outputs for.
            callback (`callable`): Called with a ``GenerationOutput`` for each result.
        """
        loop = asyncio.get_running_loop()

        def _auto_cleanup(result):
            pass

        with self.output_router._lock:
            self.output_router.result_handlers[request_id] = (_auto_cleanup, loop)


    @torch.no_grad()
    def _run_generation_loop(self) -> None:
        pass

    def _generation_step(self) -> None:
        pass

    def _create_batch_processor(self) -> ContinuousBatchProcessor:
        """Create a new batch processor. If an already initialized batch processor exists, it is reset and returned."""
        batch_processor = getattr(self, "batch_processor", None)
        if isinstance(batch_processor, ContinuousBatchProcessor):
            batch_processor.reset()
            return batch_processor

        paged_attention_cache = PagedAttentionCache(
            config=self.model.config,
            continuous_batching_config=self.continuous_batching_config,
            device=self.model.device,
            distributed_helper=self.distributed_helper,
            tp_plan=getattr(self.model, "tp_plan", {}),
            dtype=self.model.dtype,
        )
        self._use_prefix_sharing = paged_attention_cache.use_prefix_sharing
        update_cb_config_after_cache_creation(
            cb_config=self.continuous_batching_config,
            num_blocks=paged_attention_cache.num_blocks,
            max_batch_tokens=paged_attention_cache.max_batch_tokens,
            use_prefix_sharing=self._use_prefix_sharing,
        )

        if paged_attention_cache.num_sliding_attention_groups > 0:
            self.continuous_batching_config.max_blocks_per_request = 0

        scheduler_type = self.continuous_batching_config.scheduler_type
        scheduler_cls = SCHEDULER_MAPPING.get(scheduler_type, None)
        if scheduler_cls is None:
            logger.warning(f"Scheduler '{scheduler_type}' not found. Defaulting to FIFO.")
            scheduler_cls = FIFOScheduler
        scheduler = scheduler_cls(
            cache=paged_attention_cache,
            safety_margin=self.continuous_batching_config.safety_margin,
            max_requests_per_batch=self.continuous_batching_config.max_requests_per_batch,
        )

        batch_processor = ContinuousBatchProcessor(
            cache=paged_attention_cache,
            config=self.model.config,
            generation_config=self.generation_config,
            continuous_batching_config=self.continuous_batching_config,
            logit_processor=self.logit_processor,
            input_queue=self.input_queue if self.is_tp_driver else None,
            cancel_queue=self.cancel_queue if self.is_tp_driver else None,
            output_router=self.output_router,
            background_thread_status=self.background_thread_status,
            model_device=self.model.device,
            model_dtype=self.model.dtype,
            scheduler=scheduler,
            distributed_helper=self.distributed_helper,
        )
        return batch_processor

    def _handle_critical_error(self, error: Exception, batch_processor: ContinuousBatchProcessor | None) -> None:
        pass

    def _fail_all_remaining_requests(self, error: Exception, batch_processor: ContinuousBatchProcessor | None) -> None:
        pass


class ContinuousMixin:

    generation_config: GenerationConfig

    @torch.no_grad()
    def init_continuous_batching(
        self,
        generation_config: GenerationConfig | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        workload_hints: WorkloadHints | None = None,
    ) -> ContinuousBatchingManager:
        """Initialize a manager for continuous batching inference.

        Args:
            generation_config: An optional generation configuration, which may contain a CompileConfig object
            continuous_batching_config: An optional continuous batching configuration
            workload_hints: Optional WorkloadHints to help the continuous batching manager make better decisions for
                default values
        Returns:
            `ContinuousBatchingManager`: The manager instance to add requests and retrieve results.
        """
        if not hasattr(self, "config") or not hasattr(self, "device") or not hasattr(self, "dtype"):
            raise AttributeError("Model must have 'config', 'device', and 'dtype' attributes.")

        cached_manager = getattr(self, "_cached_continuous_batching_manager", None)
        if isinstance(cached_manager, ContinuousBatchingManager):
            logger.info(
                "Cached continuous batching manager found: it will be re-used instead of creating a new one. If you"
                " want to create a new manager, you should call `destroy_cached_continuous_batching_manager` first."
            )
            cached_manager.switch_to_cb_friendly_attn(self)  # might have switched in .stop
            return cached_manager

        gen_config = generation_config if generation_config is not None else self.generation_config
        if gen_config is None:
            raise ValueError("A GenerationConfig must be provided or set in the model.")
        if gen_config.eos_token_id is None:
            logger.warning("`eos_token_id` not set in GenerationConfig. Setting to -1 (disabled).")
            gen_config.eos_token_id = -1

        if continuous_batching_config is None:
            if isinstance(getattr(gen_config, "continuous_batching_config", None), ContinuousBatchingConfig):
                continuous_batching_config = gen_config.continuous_batching_config
            else:
                continuous_batching_config = ContinuousBatchingConfig()

        return ContinuousBatchingManager(
            model=self,
            generation_config=gen_config,
            continuous_batching_config=continuous_batching_config,
            workload_hints=workload_hints,
        )

    def destroy_cached_continuous_batching_manager(self) -> None:
        pass

    @contextmanager
    @torch.no_grad()
    def continuous_batching_context_manager(
        self,
        generation_config: GenerationConfig | None = None,
        block: bool = True,
        timeout: float | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        persistent_manager: bool = False,
        warmup: bool = True,
        workload_hints: WorkloadHints | None = None,
    ) -> Generator[ContinuousBatchingManager]:
        """A context manager to safely use the continuous batching manager. Arguments are similar to the ones of
        `init_continuous_batching`, except for:
            - block: whether to block the thread when stopping the manager. Default is True.
            - timeout: maximum time to wait for the thread to stop. Default is None (no timeout).
            - warmup: whether to pre-capture CUDA graphs at the largest sizes before running. Default is True.
        """
        manager = self.init_continuous_batching(
            generation_config=generation_config,
            continuous_batching_config=continuous_batching_config,
            workload_hints=workload_hints,
        )
        if warmup and not manager.warmed_up:
            logger.warning("Warming up for continuous batching...")
            start = perf_counter()
            manager.warmup()
            logger.warning(f"Warming up completed in {perf_counter() - start:.2f}s.")
        manager.start()
        try:
            yield manager
        finally:
            logger.debug("Continuous batching loop finished")
            manager.stop(block=block, timeout=timeout, keep_for_next_session=persistent_manager)
            if not persistent_manager:
                manager.destroy()

    @torch.no_grad()
    def generate_batch(
        self,
        inputs: list[list[int]],
        generation_config: GenerationConfig | None = None,
        continuous_batching_config: ContinuousBatchingConfig | None = None,
        record_timestamps: bool = False,
        progress_bar: bool = True,
        persistent_manager: bool = False,
        warmup: bool = True,
        **kwargs,
    ) -> dict[str, GenerationOutput]:
        """Generate sequences for a batch of prompts using continuous batching.

        Args:
            inputs: List of input token sequences (prompts)
            generation_config: Optional generation configuration
            continuous_batching_config: Optional continuous batching configuration
            record_timestamps: If set to true, the requests will have a timestamp for each token generated
            progress_bar: If set to true, a progress bar will be displayed
            persistent_manager: whether to persist the manager after the generation is finished. Default is False.
            warmup: whether to pre-capture CUDA graphs before processing requests. Default is True.
        Returns:
            `dict[str, GenerationOutput]`: a dictionary of request ids to GenerationOutput objects
        """
        if not inputs:
            return {}

        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.warning("Progress bar is disabled when logger level is less than DEBUG")
            progress_bar = False

        gen_cfg = self.generation_config if generation_config is None else generation_config
        num_return_sequences = gen_cfg.num_return_sequences if gen_cfg.num_return_sequences is not None else 1
        num_requests = len(inputs) * num_return_sequences

        max_new_tokens = kwargs.pop("max_new_tokens", None)
        max_new_tokens = gen_cfg.max_new_tokens if max_new_tokens is None else max_new_tokens

        workload_hints = WorkloadHints(
            max_prompt_length=max(len(input_ids) for input_ids in inputs),
            max_generated_length=max_new_tokens if max_new_tokens is not None else 0,
            num_requests=num_requests,
        )
        if persistent_manager:
            logger.warning(
                "Since you passed `persistent_manager=True`, the manager will be kept alive after the generation is "
                "finished. However, it was sized specifically for the requests passed in `generate_batch`. If you plan "
                "to reuse the manager for a very different workload, you might want to create a new manager instead."
            )

        manager_cm = self.continuous_batching_context_manager(
            generation_config=generation_config,
            continuous_batching_config=continuous_batching_config,
            block=True,
            timeout=5,
            persistent_manager=persistent_manager,
            warmup=warmup,
            workload_hints=workload_hints,
        )
        logging_cm = logging_redirect_tqdm([logger])
        pbar_cm = tqdm(
            total=num_requests,
            disable=(not progress_bar),
            desc=f"Solving {num_requests} requests",
            unit="request",
        )

        results = {}
        finished_count = 0
        with manager_cm as manager, logging_cm, pbar_cm as pbar:
            try:
                request_ids = manager.add_requests(
                    inputs=inputs, max_new_tokens=max_new_tokens, record_timestamps=record_timestamps
                )
                while finished_count < num_requests:
                    result = manager.get_result(timeout=1)
                    if result:
                        req_id = result.request_id
                        if result.is_finished():
                            results[req_id] = result
                            finished_count += 1
                            pbar.update(1)
                    elif not manager.is_running():
                        logger.error("Generation thread terminated unexpectedly.")
                        print("Returning results of generate_batch despite unexpected termination.")
                        break

            except Exception as e:
                logger.error(f"Error during batch generation: {e}", exc_info=True)

        reordered_results = {}
        missing_keys = []
        for req_id in request_ids:
            result = results.get(req_id)
            if result is not None:
                reordered_results[req_id] = result
            else:
                missing_keys.append(req_id)
        if missing_keys:
            logger.error(f"Requests {missing_keys} not found in results.")
        return reordered_results
