
import copy
import weakref
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
import torch
import torch.nn as nn

from ..pytorch_utils import prune_linear_layer
from ..utils import ModelOutput, is_sklearn_available
from .configuration_utils import GenerationConfig
from .logits_process import LogitsProcessorList, MinLengthLogitsProcessor, SuppressTokensLogitsProcessor


if is_sklearn_available():
    from sklearn.metrics import roc_curve

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..tokenization_utils_base import PreTrainedTokenizerBase
    from .configuration_utils import GenerationConfig


class CandidateGenerator:

    requires_model_outputs: bool = False

    def get_candidates(self, input_ids: torch.LongTensor, **kwargs) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Fetches the candidates to be tried for the current input.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)

        Return:
            `torch.LongTensor` of shape `(batch_size, candidate_length)` containing the candidate sequences to be
            assessed by the model and, optionally, a `torch.FloatTensor` of shape `(batch_size, candidate_length,
            vocabulary_size)` containing the logits associated to each candidate.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call `get_candidates`."
        )

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        """
        Updates the candidate generation strategy based on the outcomes.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, candidate_length, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            num_matches (`int`):
                The number of matches between the candidate sequences and the model predictions.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can call "
            "`update_candidate_strategy`."
        )


class AssistedCandidateGenerator(CandidateGenerator):

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: dict,
        inputs_tensor: torch.Tensor | None = None,
        logits_processor: Optional["LogitsProcessorList"] = None,
    ):
        device = assistant_model.device
        input_ids = input_ids.to(device)
        if inputs_tensor is not None:
            inputs_tensor = inputs_tensor.to(device)

        self.assistant_model = assistant_model

        self.assistant_generation_config = copy.deepcopy(assistant_model.generation_config)
        global_defaults = self.assistant_generation_config._get_default_generation_params()
        self.assistant_generation_config.update(**global_defaults, defaults_only=True)
        self.num_assistant_tokens = self.assistant_generation_config.num_assistant_tokens
        self.assistant_confidence_threshold = self.assistant_generation_config.assistant_confidence_threshold

        self.assistant_generation_config.eos_token_id = generation_config.eos_token_id

        assistant_kwargs = {}
        for key, value in model_kwargs.items():  # deepcopy crashes if we attempt to copy encoder outputs with grads
            if key not in ("encoder_outputs", "past_key_values"):
                assistant_kwargs[key] = (
                    value.detach().to(device) if isinstance(value, torch.Tensor) else copy.deepcopy(value)
                )

        if "logits_to_keep" in assistant_kwargs and not assistant_model._supports_logits_to_keep():
            del assistant_kwargs["logits_to_keep"]

        if assistant_model.config.is_encoder_decoder:
            inputs_tensor, model_input_name, assistant_kwargs = assistant_model._prepare_model_inputs(
                inputs_tensor, self.assistant_generation_config.bos_token_id, assistant_kwargs
            )
            assistant_kwargs = assistant_model._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, assistant_kwargs, model_input_name, self.assistant_generation_config
            )
        elif "encoder_outputs" in model_kwargs:
            assistant_kwargs["encoder_outputs"] = model_kwargs["encoder_outputs"]
        self.assistant_kwargs = assistant_kwargs

        if assistant_model.config.is_encoder_decoder:
            self.input_ids_key = "decoder_input_ids"
        elif "encoder_outputs" in assistant_kwargs:
            self.input_ids_key = "input_ids"
            self.assistant_kwargs["attention_mask"] = self.assistant_kwargs.get(
                "decoder_attention_mask",
                torch.ones((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.long),
            )
        else:
            self.input_ids_key = "input_ids"

        self.logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        self.generation_config = copy.deepcopy(generation_config)

        self.generation_config.return_dict_in_generate = True
        self.generation_config.output_scores = True
        self.generation_config.assistant_confidence_threshold = self.assistant_confidence_threshold
        self.generation_config.is_assistant = True

        self.main_model_min_length = self.generation_config.min_length
        self.generation_config.min_length = None
        self.generation_config.min_new_tokens = None
        self.main_model_max_length = self.generation_config.max_length
        self.generation_config.max_length = None
        self.logits_processor = [
            processor for processor in self.logits_processor if not isinstance(processor, MinLengthLogitsProcessor)
        ]

        if (
            is_sklearn_available()
            and self.assistant_generation_config.assistant_confidence_threshold
            and type(self) is AssistedCandidateGenerator
        ):
            self.probs = []
            self.matches = []

    def get_candidates(self, input_ids: torch.LongTensor, **kwargs) -> tuple[torch.LongTensor, torch.FloatTensor]:
        pass

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        pass

    def _calculate_new_tokens(self, input_ids: torch.LongTensor) -> tuple[int, int]:
        pass

    def _update_past_and_masks(
        self, input_ids: torch.LongTensor, remove_from_pkv: int = 0, num_added_tokens: int = 1
    ) -> bool:
        pass

    def _prepare_generation_args(self, input_ids: torch.LongTensor, min_new_tokens: int, max_new_tokens: int) -> dict:
        pass

    def _generate_candidates(self, generation_args: dict) -> tuple[torch.LongTensor, torch.FloatTensor | None]:
        pass


class AssistedCandidateGeneratorDifferentTokenizers(AssistedCandidateGenerator):

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        target_tokenizer: "PreTrainedTokenizerBase",
        assistant_tokenizer: "PreTrainedTokenizerBase",
        generation_config: "GenerationConfig",
        model_kwargs: dict,
        inputs_tensor: torch.Tensor | None = None,
        logits_processor: Optional["LogitsProcessorList"] = None,
    ):
        super().__init__(input_ids, assistant_model, generation_config, model_kwargs, inputs_tensor, logits_processor)

        self.target_tokenizer = target_tokenizer
        self.assistant_tokenizer = assistant_tokenizer
        self.prev_target_ids_len: int | None = None
        self.prev_assistant_ids: torch.LongTensor | None = None
        self.target_lookbehind = self.assistant_generation_config.target_lookbehind
        self.assistant_lookbehind = self.assistant_generation_config.assistant_lookbehind

    @staticmethod
    def _get_longest_diag_dict(input_matrix, nonzero_idx):
        pass

    @staticmethod
    def _get_longest_diag_index(input_matrix):
        pass

    @staticmethod
    def _get_tokens_diag(prompt, prompt_plus_new_tokens):
        pass

    def convert_source_tokens_to_target_tokens(
        self,
        input_ids,
        source_tokenizer,
        destination_tokenizer,
    ):
        pass

    def get_candidates(self, input_ids: torch.LongTensor, **kwargs) -> tuple[torch.LongTensor, torch.FloatTensor]:
        pass

    def _prepare_assistant_input_ids(self, input_ids: torch.LongTensor) -> tuple[torch.LongTensor, int]:
        pass

    def _process_assistant_outputs(
        self, input_ids: torch.LongTensor, assistant_sequences: torch.LongTensor
    ) -> torch.LongTensor:
        pass


class _PruneReindexingLMHead(nn.Module):

    def __init__(self, original_lm_head, assistant_overlap_token_ids):
        super().__init__()
        self.pruned_lm_head = prune_linear_layer(original_lm_head, assistant_overlap_token_ids).to(
            original_lm_head.weight.dtype
        )

    def forward(self, hidden_states):
        pruned_logits = self.pruned_lm_head(hidden_states)
        return pruned_logits


class _MapInputEmbedding(nn.Module):
    def __init__(self, original_embedding: nn.Embedding, assistant_overlap_token_ids):
        """
        Wraps an existing embedding layer and remaps token IDs before lookup.

        Args:
            original_embedding (nn.Embedding): Pre-trained or existing embedding layer.
            assistant_overlap_token_ids (dict): Mapping from original token IDs to new token IDs.
                          Example: {old_id: new_id}
        """
        super().__init__()
        self.original_embedding = original_embedding
        self.weight = original_embedding.weight
        self.assistant_overlap_token_ids = assistant_overlap_token_ids
        self.map = False

    def forward(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Args:
            input_ids (torch.LongTensor): Tensor of token IDs (batch_size, seq_len).

        Returns:
            torch.FloatTensor: Corresponding input embeddings.
        """
        if self.map:
            my_input_ids = self.assistant_overlap_token_ids[input_ids[0, -1]].unsqueeze(0).unsqueeze(0)
        else:
            self.map = True
            my_input_ids = input_ids

        return self.original_embedding(my_input_ids)


class AssistantToTargetTranslator:

    FILTER_VALUE: float = -float("Inf")  # The value used to filter out unmapped tokens in the logits.
    SUPPRESS_TOKEN_ID: int = -1  # The ID used to mark suppressed tokens in the mapping.

    def __init__(
        self,
        target_tokenizer: "PreTrainedTokenizerBase",
        assistant_tokenizer: "PreTrainedTokenizerBase",
        target_vocab_size: int,  # required since target_vocab_size can be different from the length of target_tokenizer.get_vocab()
        assistant_model: Optional["PreTrainedModel"] = None,
        assistant_prune_lm_head: bool = False,
    ):
        self._target_tokenizer: PreTrainedTokenizerBase = target_tokenizer
        self._assistant_tokenizer: PreTrainedTokenizerBase = assistant_tokenizer
        self._assistant_model_device = assistant_model.device if assistant_model is not None else "cpu"
        self.target_vocab_size: int = target_vocab_size
        self._assistant_to_target_input_ids, self.target_to_assistant_input_ids = (
            self._get_assistant_to_target_input_ids()
        )
        self._suppress_input_ids: list[int] = self._get_suppress_input_ids()
        self.logits_processors: LogitsProcessorList | None = None
        self.assistant_prune_lm_head = assistant_prune_lm_head and assistant_model is not None
        if len(self._suppress_input_ids) > 0:
            if self.assistant_prune_lm_head and assistant_model is not None:
                self.assistant_overlap_token_ids = torch.tensor(
                    list(self.target_to_assistant_input_ids.values()),
                    dtype=torch.long,
                    device=self._assistant_model_device,
                )
                original_lm_head = assistant_model.get_output_embeddings()
                pruned_lm_head = _PruneReindexingLMHead(original_lm_head, self.assistant_overlap_token_ids)
                del original_lm_head
                assistant_model.set_output_embeddings(pruned_lm_head)

                original_input_embeddings = assistant_model.get_input_embeddings()
                map_input_embeddings = _MapInputEmbedding(original_input_embeddings, self.assistant_overlap_token_ids)
                del original_input_embeddings
                assistant_model.set_input_embeddings(map_input_embeddings)
                self.map_input_embeddings = map_input_embeddings
            else:
                self.logits_processors = LogitsProcessorList(
                    [SuppressTokensLogitsProcessor(self._get_suppress_input_ids(), self._assistant_model_device)]
                )

    def unmap_input_ids(self):
        pass

    def _get_assistant_to_target_input_ids(self):
        target_vocab = self._target_tokenizer.get_vocab()
        assistant_vocab = self._assistant_tokenizer.get_vocab()

        space_str = " "
        target_space_ids = self._target_tokenizer(space_str, add_special_tokens=False)["input_ids"]
        if len(target_space_ids) > 0:
            target_space_sign = self._target_tokenizer.convert_ids_to_tokens(target_space_ids)[0][0]

            assistant_space_ids = self._assistant_tokenizer(space_str, add_special_tokens=False)["input_ids"]
            if len(assistant_space_ids) > 0:
                assistant_space_sign = self._assistant_tokenizer.convert_ids_to_tokens(assistant_space_ids)[0][0]

                if target_space_sign != assistant_space_sign:
                    assistant_vocab = {
                        (
                            tok.replace(assistant_space_sign, target_space_sign, 1)
                            if tok.startswith(assistant_space_sign)
                            else tok
                        ): idx
                        for tok, idx in assistant_vocab.items()
                    }

        max_assistant_index = max(assistant_vocab.values())
        assistant_to_target_input_ids = torch.full((max_assistant_index + 1,), self.SUPPRESS_TOKEN_ID, dtype=int)
        target_to_assistant_input_ids: dict[int, int] = {}
        for tok, assistant_id in assistant_vocab.items():
            target_id = target_vocab.get(tok)
            if target_id is not None:
                assistant_to_target_input_ids[assistant_id] = target_id
                target_to_assistant_input_ids[target_id] = assistant_id
        return assistant_to_target_input_ids.to(self._assistant_model_device), target_to_assistant_input_ids

    def _get_suppress_input_ids(self) -> list[int]:
        """
        Get the input ids that are in the assistant vocab but not in the target vocab.
        """
        return torch.where(self._assistant_to_target_input_ids == self.SUPPRESS_TOKEN_ID)[0]

    def get_target_ids(
        self, assistant_input_ids, target_input_ids, assistant_candidate_ids: torch.LongTensor
    ) -> torch.LongTensor:
        """
        Return the target candidate ids that correspond to the assistant candidate ids.
        Note that we have already the target ids for the prompt and we only need to find the target ids for the new tokens.
        Moreover, assistant ids of the original prompt does not necessarily appear in _assistant_to_target_input_ids.
        """

        num_new_tokens = len(assistant_candidate_ids[0]) - assistant_input_ids.shape[1]
        if num_new_tokens == 0:
            return target_input_ids
        else:
            last_candidate_ids = assistant_candidate_ids[0, -num_new_tokens:]
            if self.assistant_prune_lm_head:
                last_candidate_ids = self.assistant_overlap_token_ids[last_candidate_ids]
            transformed_slice = self._assistant_to_target_input_ids[last_candidate_ids]
            return torch.cat((target_input_ids, transformed_slice.unsqueeze(0)), dim=1)

    def get_target_logits(self, assistant_logits: torch.FloatTensor) -> torch.FloatTensor:
        pass


class AssistantVocabTranslatorCache:

    _cache = weakref.WeakKeyDictionary()

    @classmethod
    def get_translator(
        cls,
        target_tokenizer: "PreTrainedTokenizerBase",
        assistant_tokenizer: "PreTrainedTokenizerBase",
        target_vocab_size: int,
        assistant_model: Optional["PreTrainedModel"] = None,
        assistant_prune_lm_head: bool = False,
    ) -> AssistantToTargetTranslator:
        pass

    @classmethod
    def cleanup(cls):
        """
        Clean up dead references in the cache.
        This removes entries where either the target_tokenizer or assistant_tokenizer
        has been garbage collected.
        """
        dead_keys = [key for key in cls._cache if key is None]
        for key in dead_keys:
            del cls._cache[key]

        for assistant_dict in cls._cache.values():
            dead_keys = [key for key in assistant_dict if key is None]
            for key in dead_keys:
                del assistant_dict[key]


class UniversalSpeculativeDecodingGenerator(AssistedCandidateGeneratorDifferentTokenizers):

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        target_tokenizer: "PreTrainedTokenizerBase",
        assistant_tokenizer: "PreTrainedTokenizerBase",
        generation_config: "GenerationConfig",
        model_kwargs: dict,
        atm_translator: AssistantToTargetTranslator,
        inputs_tensor: torch.Tensor | None = None,
        logits_processor: Optional["LogitsProcessorList"] = None,
    ):
        self._atm_translator = atm_translator
        super().__init__(
            input_ids,
            assistant_model,
            target_tokenizer,
            assistant_tokenizer,
            generation_config,
            model_kwargs,
            inputs_tensor,
            logits_processor,
        )
        self._target_seq_len_with_candidates: int = 0
        self._prev_assistant_ids: torch.LongTensor | None = None

    def get_candidates(self, input_ids: torch.LongTensor, **kwargs) -> tuple[torch.LongTensor, torch.FloatTensor]:
        pass

    def _update_past_and_masks(self, assistant_input_ids: torch.LongTensor, num_added_tokens: int = 1) -> bool:
        pass

    def _prepare_assistant_input_ids(self, target_input_ids: torch.LongTensor) -> torch.LongTensor:
        pass


class PromptLookupCandidateGenerator(CandidateGenerator):

    def __init__(
        self,
        eos_token_id: torch.Tensor | None = None,
        num_output_tokens: int = 10,
        max_matching_ngram_size: int = 2,
        max_length: int = 20,
        logits_processor: Optional["LogitsProcessorList"] = None,
        vocab_size: int | None = None,
    ):
        self.num_output_tokens = num_output_tokens
        self.max_matching_ngram_size = max_matching_ngram_size
        self.max_length = max_length
        self.eos_token_id = eos_token_id
        self.logits_processor = logits_processor
        self.vocab_size = vocab_size

        if self.max_matching_ngram_size <= 0 or self.num_output_tokens <= 0:
            raise ValueError("Invalid max_matching_ngram_size or num_output_tokens")

    def get_candidates(self, input_ids: torch.LongTensor, **kwargs) -> tuple[torch.LongTensor, torch.FloatTensor]:
        pass

    def update_candidate_strategy(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int):
        pass


class EarlyExitCandidateGenerator(AssistedCandidateGenerator):

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: dict,
        inputs_tensor: torch.Tensor | None = None,
        logits_processor: Optional["LogitsProcessorList"] = None,
    ):
        super().__init__(
            input_ids=input_ids,
            assistant_model=assistant_model,
            generation_config=generation_config,
            model_kwargs=model_kwargs,
            inputs_tensor=inputs_tensor,
            logits_processor=logits_processor,
        )
        self.assistant_early_exit = self.generation_config.assistant_early_exit
        self.generation_config.assistant_early_exit = None

    def get_candidates(self, input_ids: torch.LongTensor, **kwargs) -> tuple[torch.LongTensor, torch.FloatTensor]:
        pass


class SinglePositionMultiTokenCandidateGenerator(AssistedCandidateGenerator):

    requires_model_outputs: bool = True
    model_kwargs_overrides: dict[str, Any] = {
        "output_hidden_states": True,
        "return_shared_kv_states": True,
    }

    def __init__(
        self,
        input_ids: torch.LongTensor,
        assistant_model: "PreTrainedModel",
        target_model_input_embeddings: nn.Embedding,
        generation_config: "GenerationConfig",
        model_kwargs: dict,
        inputs_tensor: torch.Tensor | None = None,
        logits_processor: Optional["LogitsProcessorList"] = None,
        eos_token_id: int | list[int] | torch.Tensor | None = None,
    ):
        if (
            "Gemma4Assistant" not in assistant_model.__class__.__name__
            and "Gemma4UnifiedAssistant" not in assistant_model.__class__.__name__
        ):
            raise ValueError(
                f"Expected assistant_model to be a Gemma4AssistantForCausalLM or Gemma4UnifiedAssistantForCausalLM. Got {assistant_model.__class__.__name__}"
                " This candidate generator requires that the assistant model is able to work from a shared_kv_states"
                " dictionary. Currently, only the Gemma4AssistantForCausalLM and Gemma4UnifiedAssistantForCausalLM support this."
            )

        super().__init__(input_ids, assistant_model, generation_config, model_kwargs, inputs_tensor, logits_processor)
        self.target_model_input_embeddings = target_model_input_embeddings

        if eos_token_id is None:
            eos_token_id: set = set()

            if isinstance(self.generation_config.eos_token_id, Iterable):
                eos_token_id.update(self.generation_config.eos_token_id)
            elif isinstance(self.generation_config.eos_token_id, int):
                eos_token_id.add(self.generation_config.eos_token_id)

            if isinstance(self.assistant_generation_config.eos_token_id, Iterable):
                eos_token_id.update(self.assistant_generation_config.eos_token_id)
            elif isinstance(self.assistant_generation_config.eos_token_id, int):
                eos_token_id.add(self.assistant_generation_config.eos_token_id)

            self.eos_token_id = torch.tensor(list(eos_token_id), dtype=torch.long) if eos_token_id else None
        elif not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            self.eos_token_id = torch.tensor(eos_token_id, dtype=torch.long)
        else:
            self.eos_token_id = eos_token_id.long()

    def get_candidates(
        self,
        input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
        model_outputs: ModelOutput,
        is_first_iteration: bool,
        n_last_matches: int,
        **kwargs,
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]:
        pass


class MTPCandidateGenerator(AssistedCandidateGenerator):
    requires_model_outputs: bool = True
    model_kwargs_overrides: dict[str, Any] = {"output_hidden_states": True}

    def __init__(
        self,
        main_model: "PreTrainedModel",
        generation_config: "GenerationConfig",
        model_kwargs: dict[str, Any],
        logits_processor: Optional["LogitsProcessorList"] = None,
    ):
        from ..cache_utils import MtpCache
        from ..modeling_layers import MtpModel

        self.num_mtp_layers = getattr(main_model.config.get_text_config(), "num_mtp_layers", None)
        if self.num_mtp_layers is None:
            raise ValueError(
                "Could not find `num_mtp_layers` in the model config. This model probably has no associated "
                "mtp weights."
            )

        base_model = main_model.get_decoder()
        self.device = next(x.device for x in base_model.layers[-1].parameters())  # type: ignore
        self.mtp_model = MtpModel.from_pretrained(main_model, device_map={"": self.device})

        self.mtp_cache = MtpCache(config=main_model.config.get_mtp_config())
        self.mtp_cache.activate_past_recording()

        self.do_sample = generation_config.do_sample
        self.logits_processor = logits_processor

        self.is_main_model_prefill = True

    def get_candidates(
        self,
        input_ids: torch.LongTensor,
        model_kwargs: dict[str, Any],
        model_outputs: ModelOutput,
        is_first_iteration: bool,
        n_last_matches: int,
        **kwargs,
    ) -> tuple[torch.LongTensor, torch.FloatTensor | None]:
        pass

    def update_candidate_strategy(self, *args, **kwargs):
        pass


def _prepare_attention_mask(model_kwargs: dict[str, Any], new_length: int, is_encoder_decoder: bool) -> dict[str, Any]:
    """Expands or crops the model's mask for decoding purposes, to the defined length"""

    mask_key = "decoder_attention_mask" if is_encoder_decoder else "attention_mask"
    if mask_key not in model_kwargs:
        return model_kwargs

    mask = model_kwargs[mask_key]
    mask_length_diff = new_length - mask.shape[1]

    if mask_length_diff < 0:
        model_kwargs[mask_key] = mask[:, :mask_length_diff]
    elif mask_length_diff > 0:
        model_kwargs[mask_key] = torch.cat([mask, mask.new_ones((mask.shape[0], mask_length_diff))], dim=-1)

    if "cross_attention_mask" in model_kwargs:
        cross_mask = model_kwargs["cross_attention_mask"]
        if mask_length_diff < 0:
            model_kwargs["cross_attention_mask"] = cross_mask[:, :mask_length_diff]
        elif mask_length_diff > 0:
            new_mask = cross_mask[:, -1:, :, :].repeat(1, mask_length_diff, 1, 1)
            model_kwargs["cross_attention_mask"] = torch.cat([cross_mask, new_mask], dim=1)
    elif "image_attention_mask" in model_kwargs:
        cross_mask = model_kwargs["image_attention_mask"]
        if mask_length_diff < 0:
            model_kwargs["image_attention_mask"] = cross_mask[:, :mask_length_diff]
        elif mask_length_diff > 0:
            new_mask = cross_mask[:, -1:, :].repeat(1, mask_length_diff, 1)
            model_kwargs["image_attention_mask"] = torch.cat([cross_mask, new_mask], dim=1)

    return model_kwargs


def _prepare_position_ids(model_kwargs: dict[str, Any], new_length: int, is_encoder_decoder: bool) -> dict[str, Any]:
    pass


def _prepare_token_type_ids(model_kwargs: dict[str, Any], new_length: int) -> dict[str, Any]:
    pass
