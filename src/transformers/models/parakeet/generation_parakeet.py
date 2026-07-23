
from dataclasses import dataclass

import torch

from ...generation import GenerationMixin, StoppingCriteria
from ...utils import ModelOutput


class ParakeetRNNTDecoderCache:
    def __init__(self, config):
        self.config = config
        self.cache: torch.Tensor | None = None
        self.hidden_state: torch.Tensor | None = None
        self.cell_state: torch.Tensor | None = None
        self.is_initialized: bool = False

    def lazy_initialization(self, hidden_states):
        self.cache = torch.zeros(
            hidden_states.shape[0],
            1,
            self.config.decoder_hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.hidden_state = torch.zeros(
            self.config.num_decoder_layers,
            hidden_states.shape[0],
            self.config.decoder_hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        self.cell_state = torch.zeros(
            self.config.num_decoder_layers,
            hidden_states.shape[0],
            self.config.decoder_hidden_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        torch._dynamo.mark_static_address(self.cache)
        torch._dynamo.mark_static_address(self.hidden_state)
        torch._dynamo.mark_static_address(self.cell_state)

        self.is_initialized = True

    def update(
        self,
        decoder_output,
        hidden_state,
        cell_state,
        mask=None,
    ):
        if not self.is_initialized:
            self.lazy_initialization(decoder_output)

        if mask is None:
            self.hidden_state.copy_(hidden_state)
            self.cell_state.copy_(cell_state)
            self.cache.copy_(decoder_output)
        else:
            mask = mask.to(decoder_output.device)
            batch_size = decoder_output.shape[0]
            mask_h = mask.view(1, batch_size, 1)
            mask_d = mask.view(batch_size, 1, 1)
            self.cache = torch.where(mask_d, decoder_output, self.cache)
            self.hidden_state = torch.where(mask_h, hidden_state, self.hidden_state)
            self.cell_state = torch.where(mask_h, cell_state, self.cell_state)


class ParakeetTDTDecoderCache(ParakeetRNNTDecoderCache): ...


@dataclass
class ParakeetRNNTGenerateOutput(ModelOutput):

    sequences: torch.LongTensor
    durations: torch.LongTensor | None = None
    attentions: tuple[tuple[torch.FloatTensor]] | None = None
    hidden_states: tuple[tuple[torch.FloatTensor]] | None = None


class EncoderExhaustedCriteria(StoppingCriteria):

    def __init__(self, model):
        self.model = model

    def __call__(self, input_ids, scores, **kwargs):
        if self.model._encoder_finished is None:
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        return self.model._encoder_finished


class ParakeetRNNTGenerationMixin(GenerationMixin):

    def _get_stopping_criteria(self, *args, **kwargs):
        criteria = super()._get_stopping_criteria(*args, **kwargs)
        criteria.append(EncoderExhaustedCriteria(self))
        return criteria

    def _update_model_kwargs_for_generation(self, outputs, *args, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, *args, **kwargs)

        logits = outputs.logits[:, -1, :]
        tokens = logits.argmax(dim=-1)
        blank_mask = tokens == self.config.blank_token_id

        if self._symbols_at_frame is None:
            self._symbols_at_frame = torch.zeros_like(tokens)
        symbols = torch.where(blank_mask, torch.zeros_like(self._symbols_at_frame), self._symbols_at_frame + 1)
        force_advance = symbols >= self.max_symbols_per_step
        self._symbols_at_frame = torch.where(blank_mask | force_advance, torch.zeros_like(symbols), symbols)

        advance = (blank_mask | force_advance).long()
        model_kwargs["encoder_frame_idxs"] = model_kwargs["encoder_frame_idxs"] + advance
        self._step_durations.append(advance)
        self._encoder_finished = model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]

        return model_kwargs

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        has_default_min_length,
        model_input_name,
        input_ids_length,
        inputs_tensor,
    ):
        if has_default_max_length and generation_config.max_new_tokens is None:
            encoder_seq_len = self.encoder._get_subsampling_output_length(
                torch.tensor([inputs_tensor.shape[1]], device=inputs_tensor.device)
            ).item()
            generation_config.max_length = self.max_symbols_per_step * encoder_seq_len
            has_default_max_length = False  # prevent super() from overwriting
        return super()._prepare_generated_length(
            generation_config,
            has_default_max_length,
            has_default_min_length,
            model_input_name,
            input_ids_length,
            inputs_tensor,
        )

    def _prepare_model_inputs(self, *args, **kwargs):
        inputs, input_name, model_kwargs = super()._prepare_model_inputs(*args, **kwargs)
        explicit = {"input_features", "attention_mask", "output_attention_mask"}
        irrelevant_prefix = ("decoder_", "cross_attn", "use_cache", "past_key_values", "cache_params")
        encoder_kwargs = {
            key: value
            for key, value in model_kwargs.items()
            if key not in explicit and not key.startswith(irrelevant_prefix)
        }

        encoder_outputs = self.get_audio_features(
            input_features=inputs,
            attention_mask=model_kwargs.get("attention_mask", None),
            output_attention_mask=True,
            **encoder_kwargs,
        )
        model_kwargs["encoder_outputs"] = encoder_outputs

        if encoder_outputs.attention_mask is not None:
            encoder_valid_lengths = encoder_outputs.attention_mask.sum(-1)
        else:
            batch_size = encoder_outputs.last_hidden_state.shape[0]
            encoder_valid_lengths = torch.full(
                (batch_size,),
                encoder_outputs.last_hidden_state.shape[1],
                dtype=torch.long,
                device=encoder_outputs.last_hidden_state.device,
            )
        model_kwargs["encoder_valid_lengths"] = encoder_valid_lengths

        model_kwargs["encoder_frame_idxs"] = torch.zeros(
            inputs.shape[0],
            device=inputs.device,
            dtype=torch.long,
        )

        return inputs, input_name, model_kwargs

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        model_kwargs["decoder_cache"] = ParakeetRNNTDecoderCache(self.config)

    def prepare_inputs_for_generation(self, input_ids, *args, **kwargs):
        from .modeling_parakeet import ParakeetEncoderModelOutput

        model_inputs = super().prepare_inputs_for_generation(input_ids, *args, **kwargs)
        encoder_frame_idxs = model_inputs.pop("encoder_frame_idxs").to(
            model_inputs["encoder_outputs"].pooler_output.device
        )

        pooler_output = model_inputs["encoder_outputs"].pooler_output
        batch_size, max_encoder_len = pooler_output.shape[0], pooler_output.shape[1]
        encoder_frame_idxs = encoder_frame_idxs.clamp(max=max_encoder_len - 1)
        model_inputs["encoder_outputs"] = ParakeetEncoderModelOutput(
            pooler_output=pooler_output[torch.arange(batch_size), encoder_frame_idxs, None],
        )

        return model_inputs

    def generate(self, inputs=None, generation_config=None, **kwargs):
        self._encoder_finished = None
        self._symbols_at_frame = None
        self._step_durations = []

        outputs = super().generate(inputs=inputs, generation_config=generation_config, **kwargs)

        durations = torch.stack(self._step_durations, dim=1)  # (batch, steps)
        durations = torch.cat(
            [torch.zeros(durations.shape[0], 1, dtype=durations.dtype, device=durations.device), durations], dim=1
        )
        del self._encoder_finished, self._symbols_at_frame, self._step_durations

        return ParakeetRNNTGenerateOutput(
            sequences=outputs.sequences if isinstance(outputs, ModelOutput) else outputs,
            durations=durations,
        )


class ParakeetTDTGenerationMixin(ParakeetRNNTGenerationMixin):

    def _update_model_kwargs_for_generation(self, outputs, *args, **kwargs):
        model_kwargs = GenerationMixin._update_model_kwargs_for_generation(self, outputs, *args, **kwargs)

        logits = outputs.logits[:, -1, :]
        tokens = logits[:, : self.config.vocab_size].argmax(dim=-1)
        durations = logits[:, self.config.vocab_size :].argmax(dim=-1)

        blank_mask = tokens == self.config.blank_token_id
        durations = torch.where(blank_mask & (durations == 0), torch.ones_like(durations), durations)
        model_kwargs["encoder_frame_idxs"] = model_kwargs["encoder_frame_idxs"] + durations
        self._step_durations.append(durations)

        self._encoder_finished = model_kwargs["encoder_frame_idxs"] >= model_kwargs["encoder_valid_lengths"]

        return model_kwargs
