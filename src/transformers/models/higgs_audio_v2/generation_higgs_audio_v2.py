
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn

from ...generation import (
    GenerateDecoderOnlyOutput,
    GenerationConfig,
    GenerationMixin,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from ...generation.logits_process import (
    InfNanRemoveLogitsProcessor,
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from ...generation.streamers import BaseStreamer
from ...generation.utils import GenerateNonBeamOutput
from ...utils import add_start_docstrings, logging


logger = logging.get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class HiggsAudioV2DelayPatternLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        delay_pattern: list[int],
        audio_bos_token_id: int,
        audio_eos_token_id: int,
        audio_stream_bos_id: int,
        audio_stream_eos_id: int,
        num_codebooks: int,
        codebook_size: int,
    ):
        self.delay_pattern = torch.tensor(delay_pattern)
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.bos_delay_pattern = None
        self.eos_delay_pattern = None
        self.vocab_mask_bos = torch.arange(codebook_size) != audio_stream_bos_id
        self.vocab_mask_eos = torch.arange(codebook_size) != audio_stream_eos_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.clone().reshape(-1, self.num_codebooks, self.codebook_size)
        batch_size = scores.shape[0]

        delay_pattern_size = len(self.delay_pattern)
        input_ids = input_ids[:, -delay_pattern_size:]

        if self.bos_delay_pattern is None:
            self.bos_delay_pattern = self.delay_pattern.repeat(batch_size, 1)
            audio_bos_idxs = (input_ids == self.audio_bos_token_id).nonzero()

            if len(audio_bos_idxs) > 0:
                batch_idxs = audio_bos_idxs[:, 0]
                is_first = torch.cat([batch_idxs.new_ones(1, dtype=torch.bool), batch_idxs[1:] != batch_idxs[:-1]])
                min_bos_idxs = audio_bos_idxs[is_first]
                current_after_bos = (delay_pattern_size - min_bos_idxs[:, 1]).unsqueeze(-1)
                unique_batch_idxs = batch_idxs.unique().to(self.bos_delay_pattern.device)
                self.bos_delay_pattern[unique_batch_idxs] = self.bos_delay_pattern[
                    unique_batch_idxs
                ] - current_after_bos.to(self.bos_delay_pattern.device)
            else:
                self.bos_delay_pattern = torch.zeros_like(self.bos_delay_pattern)

        if self.eos_delay_pattern is None:
            self.eos_delay_pattern = self.delay_pattern.repeat(batch_size, 1)
            audio_eos_idxs = (input_ids == self.audio_eos_token_id).nonzero()

            if len(audio_eos_idxs) > 0:
                batch_idxs = audio_eos_idxs[:, 0]
                is_first = torch.cat([batch_idxs.new_ones(1, dtype=torch.bool), batch_idxs[1:] != batch_idxs[:-1]])
                min_eos_idxs = audio_eos_idxs[is_first]
                current_after_eos = (delay_pattern_size - min_eos_idxs[:, 1]).unsqueeze(-1)
                unique_batch_idxs = batch_idxs.unique().to(self.eos_delay_pattern.device)
                self.eos_delay_pattern[unique_batch_idxs] = self.eos_delay_pattern[
                    unique_batch_idxs
                ] - current_after_eos.to(self.eos_delay_pattern.device)

        row_mask = self.bos_delay_pattern >= 0
        scores[(row_mask[..., None] & self.vocab_mask_bos).to(scores.device)] = -float("inf")
        self.bos_delay_pattern[row_mask] -= 1

        self.eos_delay_pattern[input_ids[:, -1].to(self.eos_delay_pattern.device) == self.audio_eos_token_id] -= 1
        row_mask = self.eos_delay_pattern <= 0
        scores[(row_mask[..., None] & self.vocab_mask_eos).to(scores.device)] = -float("inf")

        return scores.reshape(-1, self.codebook_size)


@dataclass
class HiggsAudioV2GenerationOutput(GenerateDecoderOnlyOutput):

    audio_sequences: list[torch.LongTensor] | None = None


class HiggsAudioV2GenerationMixin(GenerationMixin):
    _supported_logits_processor_types = (
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
        InfNanRemoveLogitsProcessor,
    )

    def _get_logits_processor(self, *args, **kwargs):
        parent_processors = super()._get_logits_processor(*args, **kwargs)

        unsupported = [p for p in parent_processors if not isinstance(p, self._supported_logits_processor_types)]
        if unsupported:
            unsupported_names = [type(p).__name__ for p in unsupported]
            raise ValueError(
                f"HiggsAudioV2 generates audio codebook logits, not text logits. "
                f"The following logits processors are not compatible: {unsupported_names}. "
                f"Only the following processors are supported: "
                f"{[t.__name__ for t in self._supported_logits_processor_types]}."
            )

        delay_pattern_processor = HiggsAudioV2DelayPatternLogitsProcessor(
            delay_pattern=[el + 1 for el in range(self.config.num_codebooks)],
            audio_bos_token_id=self.config.audio_bos_token_id,
            audio_eos_token_id=self.config.audio_delay_token_id,
            audio_stream_bos_id=self.config.audio_stream_bos_id,
            audio_stream_eos_id=self.config.audio_stream_eos_id,
            num_codebooks=self.config.num_codebooks,
            codebook_size=self.config.codebook_size,
        )

        logits_processor = LogitsProcessorList()
        logits_processor.append(delay_pattern_processor)
        logits_processor.extend(parent_processors)
        return logits_processor

    def _prepare_generation_config(
        self, generation_config: GenerationConfig | None, **kwargs: Any
    ) -> tuple[GenerationConfig, dict]:
        generation_config, model_kwargs = super()._prepare_generation_config(generation_config, **kwargs)
        original_get_generation_mode = generation_config.get_generation_mode

        def patched_get_generation_mode(assistant_model=None):
            pass

        generation_config.get_generation_mode = patched_get_generation_mode

        return generation_config, model_kwargs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        model_forward = (
            self.get_compiled_call(generation_config.compile_config)
            if self._valid_auto_compile_criteria(model_kwargs, generation_config)
            else self.__call__
        )

        prefill_consumed = False
        outputs = self._prefill(
            input_ids,
            generation_config,
            model_kwargs,
            is_first_iteration=not generation_config.is_assistant,
        )

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            if prefill_consumed:
                next_sequence_length = 1 if model_kwargs["use_cache"] else None
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, next_sequence_length=next_sequence_length, **model_kwargs
                )
                with self._optimize_model_for_decode():
                    outputs = model_forward(**model_inputs, return_dict=True)
            prefill_consumed = True
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            next_token_scores = logits_processor(input_ids, next_token_logits)

            if return_dict_in_generate:
                if output_scores:
                    scores += (
                        next_token_scores.reshape(batch_size, self.config.num_codebooks, self.config.codebook_size),
                    )
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (outputs.hidden_states,)

            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            next_token_logits = next_token_logits.reshape(-1, self.config.num_codebooks, self.config.codebook_size)
            next_tokens = next_tokens.reshape(batch_size, self.config.num_codebooks)

            ras_win_len = generation_config.ras_win_len if hasattr(generation_config, "ras_win_len") else None
            ras_win_max_num_repeat = (
                generation_config.ras_win_max_num_repeat
                if hasattr(generation_config, "ras_win_max_num_repeat")
                else None
            )
            audio_input_ids = model_kwargs.get("audio_input_ids")
            if ras_win_len is not None and ras_win_max_num_repeat is not None and audio_input_ids is not None:
                audio_inputs_ids_window = audio_input_ids[:, -ras_win_len:, :]
                repetition_mask = audio_inputs_ids_window == next_tokens.unsqueeze(1)

                not_excluded_mask = (audio_inputs_ids_window != self.config.audio_stream_bos_id) & (
                    audio_inputs_ids_window != self.config.audio_stream_eos_id
                )
                repetition_mask = repetition_mask & not_excluded_mask
                rep_num = repetition_mask.sum(dim=1)

                replacement_mask = rep_num >= ras_win_max_num_repeat
                replacement_tokens = (
                    next_token_logits[replacement_mask].softmax(dim=-1).multinomial(1, replacement=True).view(-1)
                )
                next_tokens[replacement_mask] = replacement_tokens

            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences[:, None] + self.config.audio_stream_eos_id * (
                    1 - unfinished_sequences[:, None]
                )

            has_audio_stream_eos = (next_tokens == self.config.audio_stream_eos_id).any(dim=-1)
            has_all_audio_stream_eos = (next_tokens == self.config.audio_stream_eos_id).all(dim=-1)
            next_tokens = next_tokens[:, None, :]

            if audio_input_ids is not None:
                model_kwargs["audio_input_ids"] = torch.cat([audio_input_ids, next_tokens], dim=1)
            else:
                model_kwargs["audio_input_ids"] = next_tokens

            next_audio_input_ids_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=next_tokens.device)
            next_audio_input_ids_mask[has_all_audio_stream_eos] = 0
            audio_input_ids_mask = model_kwargs.get("audio_input_ids_mask")
            if audio_input_ids_mask is not None:
                model_kwargs["audio_input_ids_mask"] = torch.cat(
                    [audio_input_ids_mask, next_audio_input_ids_mask], dim=1
                )
            else:
                model_kwargs["audio_input_ids_mask"] = next_audio_input_ids_mask

            next_tokens_flat = input_ids.new_ones(batch_size) * self.config.audio_token_id
            next_tokens_flat[has_audio_stream_eos | (input_ids[:, -1] == self.config.audio_delay_token_id)] = (
                self.config.audio_delay_token_id
            )
            if self.config.eos_token_id is not None:
                next_tokens_flat[has_all_audio_stream_eos] = self.config.eos_token_id
            next_tokens = next_tokens_flat

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return HiggsAudioV2GenerationOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
                audio_sequences=model_kwargs.get("audio_input_ids"),
            )
        else:
            return model_kwargs.get("audio_input_ids")
