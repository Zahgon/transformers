from __future__ import annotations

import os
import re
from functools import partial
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from safetensors import safe_open

from .cache_utils import Cache
from .conversion_mapping import get_model_conversion_mapping
from .core_model_loading import WeightRenaming, convert_and_load_state_dict_in_model
from .masking_utils import LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING, create_causal_mask
from .modeling_outputs import (
    BaseModelOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from .modeling_utils import LoadStateDictConfig, PreTrainedModel, _get_resolved_checkpoint_files
from .models.auto import AutoModel
from .processing_utils import Unpack
from .utils import ContextManagers, TransformersKwargs, auto_docstring, can_return_tuple, logging
from .utils.loading_report import log_state_dict_report


if TYPE_CHECKING:
    from .cache_utils import MtpCache
    from .configuration_utils import PreTrainedConfig
    from .generation.logits_process import LogitsProcessorList


logger = logging.get_logger(__name__)


class GradientCheckpointingLayer(nn.Module):

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning_once(message)

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)


@auto_docstring
class GenericForSequenceClassification:
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.score = nn.Linear(config.get_text_config().hidden_size, self.num_labels, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        transformer_outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.get_text_config().pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.get_text_config().pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            non_pad_mask = (input_ids != self.config.get_text_config().pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class GenericForQuestionAnswering:
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.post_init()

    def get_input_embeddings(self):
        return getattr(self, self.base_model_prefix).embed_tokens

    def set_input_embeddings(self, value):
        getattr(self, self.base_model_prefix).embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        start_positions: torch.LongTensor | None = None,
        end_positions: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> QuestionAnsweringModelOutput:
        outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class GenericForTokenClassification:
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(
            config.get_text_config().hidden_size,
            config.num_labels,
            bias=getattr(config, "token_classification_bias", True),
        )

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> TokenClassifierOutput:
        outputs: BaseModelOutputWithPast = getattr(self, self.base_model_prefix)(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MtpLayer(nn.Module):
    def __init__(
        self,
        config: PreTrainedConfig,
        decoder_layer_cls: type[nn.Module],
        norm_cls: type[nn.Module],
        layer_idx: int,
        use_post_norm: bool = True,
    ):
        super().__init__()
        self.config = config
        self.use_post_norm = use_post_norm
        self.enorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.mtp_block = decoder_layer_cls(config, layer_idx)
        self.post_norm = norm_cls(config.hidden_size, eps=config.rms_norm_eps) if use_post_norm else None

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        previous_hidden_state: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        past_key_values: Cache | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.config, "mtp_hidden_states_first", False):
            projection_input = torch.cat([self.hnorm(previous_hidden_state), self.enorm(inputs_embeds)], dim=-1)
        else:
            projection_input = torch.cat([self.enorm(inputs_embeds), self.hnorm(previous_hidden_state)], dim=-1)
        hidden_states = self.eh_proj(projection_input)
        hidden_states = self.mtp_block(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        if self.use_post_norm:
            hidden_states = self.post_norm(hidden_states)

        return hidden_states


class MtpModel(PreTrainedModel):
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_flash_attn = True
    _keys_to_ignore_on_load_unexpected = ["shared_head.head.weight", "embed_tokens.weight"]
    _keys_to_ignore_on_load_missing = ["shared_head.weight", "embed_tokens.weight"]

    def __init__(self, main_model: PreTrainedModel, num_mtp_layers: int):
        super().__init__(main_model.config.get_mtp_config())
        self.loss_type = "ForCausalLM"
        self.num_mtp_layers = num_mtp_layers
        base_model = main_model.get_decoder()
        layer_cls = type(base_model.layers[-1])
        norm_cls = next(
            type(module)
            for name, module in base_model.layers[-1].named_modules()  # type: ignore
            if "norm" in name
        )
        self.use_post_norm = True
        self.use_shared_post_norm = False
        if hasattr(self.config, "chain_hidden_post_norm"):
            self.use_post_norm = False
            self.use_shared_post_norm = self.config.chain_hidden_post_norm

        self.layers = nn.ModuleList(
            [MtpLayer(self.config, layer_cls, norm_cls, k, self.use_post_norm) for k in range(num_mtp_layers)]
        )
        if self.use_shared_post_norm:
            self.shared_post_norm = norm_cls(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.tie_with_main_model(main_model)

        self.post_init()

    def tie_with_main_model(self, main_model: PreTrainedModel):
        """Tie the embedding/head/rotary layer with the main model."""
        self.embed_tokens = main_model.get_input_embeddings()
        self.shared_head = main_model.lm_head
        base_model = main_model.get_decoder()
        self.rotary_emb = getattr(base_model, "rotary_emb", None)

    def _project_to_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the shared head the same way the main model does (muP scaling, unpadded vocab slice)."""
        multiplier = getattr(self.config, "logits_mup_width_multiplier", None)
        if multiplier is not None:
            hidden_states = hidden_states / multiplier
        logits = self.shared_head(hidden_states)
        unpadded_vocab_size = getattr(self.config, "unpadded_vocab_size", None)
        if unpadded_vocab_size is not None and unpadded_vocab_size < logits.shape[-1]:
            logits = logits[..., :unpadded_vocab_size]
        return logits

    def create_masks_for_mtp_layer(
        self, layer_idx: int, inputs_embeds: torch.Tensor, mtp_cache: MtpCache, position_ids: torch.Tensor
    ):
        """
        Create the (potentially several) masks required for layer `layer_idx`. This relies on the `layer_type`
        attribute of the mtp layer if any, otherwise simply create a causal mask for full attention.
        """
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": None,
            "past_key_values": mtp_cache,
            "position_ids": position_ids,
            "layer_idx": layer_idx,
        }

        mtp_layer_type = getattr(self.layers[layer_idx], "layer_type", None)
        masks = {}
        if mtp_layer_type is not None and mtp_layer_type in LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING:
            mask_function = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[mtp_layer_type]
            if isinstance(mask_function, dict):
                for actual_pattern, actual_function in mask_function.items():
                    masks[actual_pattern] = actual_function(**mask_kwargs)
            else:
                masks[mtp_layer_type] = mask_function(**mask_kwargs)
        else:
            masks["full_attention"] = create_causal_mask(**mask_kwargs)

        if len(masks) > 2:
            raise ValueError("You should have at most 2 masks, 1 for attention, and 1 for linear attention")

        internal_layer_expected_kwarg_mapping = {
            "full_attention": "attention_mask",
            "sliding_attention": "attention_mask",
            "linear_attention": "conv_mask",
        }
        masks = {internal_layer_expected_kwarg_mapping[k]: v for k, v in masks.items()}

        return masks

    def forward(
        self,
        input_ids: torch.Tensor,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        mtp_cache: MtpCache | None,
        labels: torch.LongTensor | None = None,
        do_sample: bool = False,
        logits_processor: LogitsProcessorList | None = None,
        full_input_ids: torch.Tensor | None = None,  # needed as input for the logits_processor
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample 1 new token for each mtp layers present in this model. Note that the inputs are assumed to be already sliced and correct
        here, i.e. if the main model just processed inputs corresponding to tokens at positions [N-1, N] in the sequence, then from it
        you draft a new token for position N+1, and the `input_ids`/`position_ids`/`attention_mask` here are assumed to correspond to
        data for tokens at positions [N, N+1], i.e. shifted by 1 from the main model, by the newly drafted token. The `last_hidden_states`
        though will correspond to the same as the main model, i.e. positions [N-1, N] in the sequence length dimension.

        `full_input_ids` correspond to the full sequence of `input_ids`, which is used in case we have any `logits_processor` as some
        processors may require to check the length/value of the full previous sequence of ids.
        """
        batch_size = input_ids.shape[0]

        drafted_logits = []
        drafted_tokens = []
        loss = None
        for i, mtp_layer in enumerate(self.layers):
            inputs_embeds = self.embed_tokens(input_ids).to(last_hidden_states.device)
            position_embeddings = (
                self.rotary_emb(inputs_embeds, position_ids=position_ids) if self.rotary_emb is not None else None
            )

            masks = self.create_masks_for_mtp_layer(i, inputs_embeds, mtp_cache, position_ids)

            last_hidden_states = mtp_layer(
                inputs_embeds,
                last_hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=mtp_cache,
                **masks,
                **kwargs,
            )
            if self.use_shared_post_norm:
                last_hidden_states = self.shared_post_norm(last_hidden_states)

            slice_indices = slice(-1, None) if labels is None else slice(None, None)
            logits = self._project_to_logits(last_hidden_states[:, slice_indices, :])

            if labels is not None:
                shift_labels = nn.functional.pad(labels, (0, i), value=-100)[..., i:].contiguous()
                loss += self.loss_function(
                    logits, labels, vocab_size=self.config.vocab_size, shift_labels=shift_labels, **kwargs
                )

            drafted_logits.append(logits)
            next_token_logits = logits[:, -1, :].to(device=input_ids.device)
            if logits_processor is not None and full_input_ids is not None:
                next_token_scores = logits_processor(full_input_ids, next_token_logits.to(torch.float32))
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1, dtype=torch.float32)
                next_mtp_token = torch.multinomial(probs, num_samples=1)
            else:
                next_mtp_token = torch.argmax(next_token_scores, dim=-1, keepdim=True)
            drafted_tokens.append(next_mtp_token)

            input_ids = torch.cat([input_ids[:, 1:], next_mtp_token], dim=-1)
            attention_mask = torch.cat([attention_mask[:, 1:], attention_mask.new_ones(batch_size, 1)], dim=-1)  # type: ignore
            position_ids = torch.cat([position_ids[:, 1:], position_ids[:, -1:] + 1], dim=-1)

            if full_input_ids is not None:
                full_input_ids = torch.cat([full_input_ids, next_mtp_token], dim=-1)

        new_candidate_ids = torch.cat(drafted_tokens, dim=1)
        candidate_logits = torch.cat(drafted_logits, dim=1)
        return new_candidate_ids, candidate_logits, loss

    @classmethod
    def from_pretrained(cls, main_model: PreTrainedModel, device_map=None, **kwargs) -> MtpModel:
        pretrained_model_name_or_path = main_model.config.name_or_path
        num_hidden_layers = main_model.config.get_text_config().num_hidden_layers
        mtp_patterns = main_model._keys_to_ignore_on_load_unexpected.copy()  # type: ignore
        final_mtp_patterns = []
        for pattern in mtp_patterns:
            match_object = re.search(r"\.(\d+)", pattern)
            if match_object is not None and int(match_object.group(1)) < num_hidden_layers:
                continue
            final_mtp_patterns.append(pattern)
        if len(final_mtp_patterns) == 0:
            raise ValueError(f"{main_model.__class__.__name__} does not seem to register any known MTP layer patterns")
        mtp_regex = re.compile("|".join(rf"({pattern})" for pattern in final_mtp_patterns))

        num_mtp_layers = main_model.config.get_text_config().num_mtp_layers
        contexts = cls.get_init_context(main_model.config.dtype, False, False, None)
        with ContextManagers(contexts):
            mtp_model = cls(main_model, num_mtp_layers)

        checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            variant=None,
            gguf_file=None,
            use_safetensors=True,
            user_agent=None,
            is_remote_code=False,
        )
        mtp_files = checkpoint_files
        mtp_weight_map = None
        if sharded_metadata is not None:
            mtp_weight_map = {
                k: v for k, v in sharded_metadata["weight_map"].items() if mtp_regex.search(k) is not None
            }
            mtp_files = [file for file in checkpoint_files if os.path.basename(file) in mtp_weight_map.values()]

        mtp_state_dict = {}
        all_pointer = set()
        for file in mtp_files:
            file_pointer = safe_open(file, framework="pt", device="cpu")
            all_pointer.add(file_pointer)
            for k in file_pointer.keys():
                if (mtp_weight_map is not None and k in mtp_weight_map.keys()) or (
                    mtp_weight_map is None and mtp_regex.search(k) is not None
                ):
                    mtp_state_dict[k] = file_pointer.get_slice(k)  # don't materialize yet

        weight_conversions = [
            WeightRenaming(
                source_patterns=f"layers.{N}.", target_patterns=f"layers.{N - num_hidden_layers}.mtp_block."
            )
            for N in range(num_hidden_layers, num_hidden_layers + num_mtp_layers)
        ]
        weight_conversions.extend(get_model_conversion_mapping(mtp_model, add_legacy=False))
        weight_conversions.extend(main_model._weight_conversions)

        loading_info, _ = convert_and_load_state_dict_in_model(
            model=mtp_model,
            state_dict=mtp_state_dict,
            load_config=LoadStateDictConfig(
                weight_mapping=weight_conversions, device_map=device_map, dtype=main_model.config.dtype
            ),
            tp_plan=None,
        )
        for k in all_pointer:
            k.__exit__(None, None, None)

        mtp_model._adjust_missing_and_unexpected_keys(loading_info)

        if loading_info.missing_keys:
            raise RuntimeError(
                f"The following {cls.__name__} weights are missing from {pretrained_model_name_or_path} "
                f"(checkpoint keys not matching the conversion mapping?): {sorted(loading_info.missing_keys)}"
            )

        mtp_model.tie_with_main_model(main_model)

        log_state_dict_report(
            model=mtp_model,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            ignore_mismatched_sizes=False,
            loading_info=loading_info,
            logger=logger,
        )

        return mtp_model

    @classmethod
    def _can_set_attn_implementation(cls) -> bool:
        return True

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return True
