from abc import ABC, abstractmethod

import torch

from ..logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from .requests import FutureRequestState, logger


class ContinuousBatchingLogitsProcessor(ABC):
    supported_kwargs: dict[str, type]
    ignored_kwargs: tuple[str, ...]

    @abstractmethod
    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default values for this processor."""
        pass

    @abstractmethod
    def prepare_tensor_args(self, requests_with_new_token: list[FutureRequestState]) -> torch.Tensor:
        pass

    @abstractmethod
    def __call__(self, scores: torch.FloatTensor, tensor_arg: torch.Tensor) -> torch.FloatTensor:
        """Applies the logits processor in a per-token manner.
        Args:
            - scores (torch.FloatTensor): The scores to process, with shape [num_tokens, vocab_size]
            - tensor_arg (torch.Tensor): The tensor argument to use for the logits processor, with shape
                [max_num_tokens] and dtype torch.int32. The dtype might not be representative of the actual data, for
                instance it's common to have a float32 tensor viewed as int32 (eg. temperature)
        Returns:
            - torch.FloatTensor: The processed scores, with shape [num_tokens, vocab_size]
        """
        pass


class ContinuousBatchingLogitsProcessorList:

    def __init__(
        self,
        logits_processor: LogitsProcessorList,
        per_request_processors: bool = False,
        drop_unsupported_processors: bool = True,
    ) -> None:
        self.logits_processor = logits_processor
        self.tensors_required = 0  # number of tensors required to store CB logits processors arguments
        if per_request_processors:
            self._convert_to_per_request_processors()
        self._validate_processors(drop_unsupported_processors)
        self._retrieve_processors_kwargs()
        self.do_processing = len(self.logits_processor) > 0

    def __repr__(self) -> str:
        return f"ContinuousBatchingLogitsProcessorList(logits_processor={self.logits_processor}, tensors_required={self.tensors_required})"

    def clear(self) -> None:
        self.logits_processor = LogitsProcessorList()
        self.tensors_required = 0
        self.supported_keys = {}
        self.ignored_keys = set()
        self.do_processing = False

    def _convert_to_per_request_processors(self) -> None:
        """Replaces the compatible logits processors with their per-request versions."""
        for i, processor in enumerate(self.logits_processor):
            for regular_cls, cb_cls in CLASSIC_TO_CB_PROCESSORS_MAP.items():
                if isinstance(processor, regular_cls):
                    self.logits_processor[i] = cb_cls(processor)
                    self.tensors_required += 1  # in the future, this might be more than 1 (will be stored in mapping)
                    break

    def _validate_processors(self, drop_unsupported: bool) -> None:
        """Validates the logits processors and optionally removes unsupported ones. When drop_unsupported is True,
        processors explicitly marked as unsupported are removed. Otherwise, all processors are kept but warnings are
        logged for unsupported or unknown ones.
        """
        filtered_processors = []
        for processor in self.logits_processor:
            class_name = processor.__class__.__name__
            supported = getattr(processor, "supports_continuous_batching", None)

            if isinstance(processor, ContinuousBatchingLogitsProcessor) or supported:
                filtered_processors.append(processor)
            elif supported is None:
                logger.warning(f"Processor {class_name} might not be supported by CB.")
                filtered_processors.append(processor)
            elif drop_unsupported:
                logger.warning(f"Processor {class_name} isn't supported by CB. Dropping it.")
            else:
                logger.warning(f"Processor {class_name} isn't supported by CB. Kept it because {drop_unsupported = }.")
                filtered_processors.append(processor)

        self.logits_processor = LogitsProcessorList(filtered_processors)

    def _retrieve_processors_kwargs(self) -> None:
        """Retrieves the supported (with types) and ignored kwargs from continuous batching processors."""
        self.supported_keys: dict[str, type] = {}
        self.ignored_keys = set()
        for processor in self.logits_processor:
            if isinstance(processor, ContinuousBatchingLogitsProcessor):
                self.supported_keys.update(processor.supported_kwargs)
                self.ignored_keys.update(processor.ignored_kwargs)

    def check_kwargs(self, kwargs: dict) -> None:
        pass

    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default values for this processor."""
        i = 0
        for processor in self.logits_processor:
            if isinstance(processor, ContinuousBatchingLogitsProcessor):
                processor.fill_defaults(int32_tensor[i])
                i += 1

    def prepare_tensor_args(
        self, requests_in_batch: list[FutureRequestState], arg_storage: torch.Tensor
    ) -> torch.Tensor:
        requests_with_new_token = [request for request in requests_in_batch if request.has_new_token]
        current_arg_id = 0
        for processor in self.logits_processor:
            if isinstance(processor, ContinuousBatchingLogitsProcessor):
                tensorized_arg = processor.prepare_tensor_args(requests_with_new_token)
                arg_storage[current_arg_id, : tensorized_arg.size(0)] = tensorized_arg.to(arg_storage.device)
                current_arg_id += 1
        return arg_storage

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, logits_processor_args: torch.Tensor
    ) -> torch.FloatTensor:
        current_arg_id = 0
        for processor in self.logits_processor:
            if isinstance(processor, ContinuousBatchingLogitsProcessor):
                scores = processor(scores, logits_processor_args[current_arg_id])
                current_arg_id += 1
            else:
                scores = processor(input_ids, scores)
        return scores


class ContinuousBatchingTemperatureLogitsWarper(ContinuousBatchingLogitsProcessor):
    supported_kwargs: dict[str, type] = {"temperature": float}
    ignored_kwargs: tuple[str, ...] = ()

    def __init__(self, temperature_processor: TemperatureLogitsWarper) -> None:
        self.temperature = temperature_processor.temperature

    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default temperature."""
        default = torch.empty_like(int32_tensor, dtype=torch.float32)
        default.fill_(self.temperature)
        int32_tensor.copy_(default.view(dtype=torch.int32))

    def prepare_tensor_args(self, requests_with_new_token: list[FutureRequestState]) -> torch.Tensor:
        data = []
        for request in requests_with_new_token:
            temp = request.state.logit_processor_kwargs.get("temperature", self.temperature)
            data.append(temp)
        tensorized = torch.tensor(data, dtype=torch.float32, device="cpu")
        return tensorized.view(dtype=torch.int32)

    def __call__(self, scores: torch.FloatTensor, tensor_arg: torch.Tensor) -> torch.FloatTensor:
        temperatures = tensor_arg[: scores.size(0)].view(dtype=torch.float32)  # shape [B]
        return scores / temperatures.unsqueeze(-1)  # broadcast [B, 1] over [B, V]


class ContinuousBatchingTopKLogitsWarper(ContinuousBatchingLogitsProcessor):
    supported_kwargs: dict[str, type] = {"top_k": int}
    ignored_kwargs: tuple[str, ...] = ("filter_value", "min_tokens_to_keep")

    def __init__(self, top_k_processor: TopKLogitsWarper):
        self.top_k = top_k_processor.top_k
        self.filter_value = top_k_processor.filter_value
        self.min_tokens_to_keep = top_k_processor.min_tokens_to_keep

    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default top_k."""
        int32_tensor.fill_(self.top_k)

    def prepare_tensor_args(self, requests_with_new_token: list[FutureRequestState]) -> torch.Tensor:
        top_ks = []
        for request in requests_with_new_token:
            top_k = request.state.logit_processor_kwargs.get("top_k", self.top_k)
            top_k = max(top_k, self.min_tokens_to_keep)
            top_ks.append(top_k)
        tensor_args = torch.tensor(top_ks, dtype=torch.int32, device="cpu")
        return tensor_args

    def __call__(self, scores: torch.FloatTensor, tensor_arg: torch.Tensor) -> torch.FloatTensor:
        """Applies top-k selection to the scores tensor (shape [B, V])."""
        top_k = tensor_arg[: scores.size(0)]  # shape [B]
        sorted_scores = torch.sort(scores, dim=-1, descending=True)[0]  # [B, V]
        top_k_indices = (top_k - 1).unsqueeze(-1).to(dtype=torch.int64)  # [B, 1]
        thresholds = sorted_scores.gather(dim=-1, index=top_k_indices)  # [B, 1]
        return scores.masked_fill(scores < thresholds, self.filter_value)


class ContinuousBatchingTopPLogitsWarper(ContinuousBatchingLogitsProcessor):
    supported_kwargs: dict[str, type] = {"top_p": float}
    ignored_kwargs: tuple[str, ...] = ("filter_value", "min_tokens_to_keep")

    def __init__(self, top_p_processor: TopPLogitsWarper):
        self.top_p = top_p_processor.top_p
        self.filter_value = top_p_processor.filter_value
        self.min_tokens_to_keep = top_p_processor.min_tokens_to_keep

    def fill_defaults(self, int32_tensor: torch.Tensor) -> None:
        """Fills the given tensor int32 tensor with the default top_p."""
        default = torch.empty_like(int32_tensor, dtype=torch.float32)
        default.fill_(self.top_p)
        int32_tensor.copy_(default.view(dtype=torch.int32))

    def prepare_tensor_args(self, requests_with_new_token: list[FutureRequestState]) -> torch.Tensor:
        top_ps = []
        for request in requests_with_new_token:
            top_p = request.state.logit_processor_kwargs.get("top_p", self.top_p)
            top_ps.append(top_p)
        tensorized = torch.tensor(top_ps, dtype=torch.float32, device="cpu")
        return tensorized.view(dtype=torch.int32)

    def __call__(self, scores: torch.FloatTensor, tensor_arg: torch.Tensor) -> torch.FloatTensor:
        """Applies top-p (nucleus) sampling to the scores tensor (shape [B, V])."""
        top_p = tensor_arg[: scores.size(0)].view(dtype=torch.float32)  # shape [B]

        sorted_logits, sorted_indices = torch.sort(scores, descending=False, dim=-1)  # [B, V]
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # [B, V]

        threshold = (1 - top_p).unsqueeze(-1)  # [B, 1]
        sorted_indices_to_remove = cumulative_probs <= threshold  # [B, V]

        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = False

        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        return scores.masked_fill(indices_to_remove, self.filter_value)


CLASSIC_TO_CB_PROCESSORS_MAP = {
    TemperatureLogitsWarper: ContinuousBatchingTemperatureLogitsWarper,
    TopKLogitsWarper: ContinuousBatchingTopKLogitsWarper,
    TopPLogitsWarper: ContinuousBatchingTopPLogitsWarper,
}
