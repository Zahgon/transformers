
import torch
import torch.nn as nn
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...modeling_outputs import CausalLMOutputWithPast
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
from ..granite.configuration_granite import GraniteConfig
from ..granite.modeling_granite import (
    GraniteAttention,
    GraniteDecoderLayer,
    GraniteForCausalLM,
    GraniteModel,
    GranitePreTrainedModel,
    GraniteRMSNorm,
    GraniteRotaryEmbedding,
)


@auto_docstring(checkpoint="naver-hyperclovax/HyperCLOVAX-SEED-Think-14B")
@strict
class HyperCLOVAXConfig(GraniteConfig):

    model_type = "hyperclovax"

    head_dim: int | None = None

    attention_multiplier: float | None = None

    use_post_norm: bool = True

    def __post_init__(
        self,
        **kwargs,
    ):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        super().__post_init__(**kwargs)

        if self.attention_multiplier is None:
            self.attention_multiplier = self.head_dim**-0.5

    def validate_architecture(self):
        pass


class HyperCLOVAXRMSNorm(GraniteRMSNorm):
    pass


class HyperCLOVAXRotaryEmbedding(GraniteRotaryEmbedding):
    pass


class HyperCLOVAXAttention(GraniteAttention):
    pass


class HyperCLOVAXDecoderLayer(GraniteDecoderLayer):
    def __init__(self, config: HyperCLOVAXConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.post_norm1 = (
            HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_norm else nn.Identity()
        )
        self.post_norm2 = (
            HyperCLOVAXRMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_norm else nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.post_norm1(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_norm2(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states


@auto_docstring
class HyperCLOVAXPreTrainedModel(GranitePreTrainedModel):
    pass


@auto_docstring
class HyperCLOVAXModel(GraniteModel):
    pass


@auto_docstring
class HyperCLOVAXForCausalLM(GraniteForCausalLM):
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
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        Example:

        ```python
        >>> from transformers import AutoTokenizer, HyperCLOVAXForCausalLM

        >>> model = HyperCLOVAXForCausalLM.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-14B")
        >>> tokenizer = AutoTokenizer.from_pretrained("naver-hyperclovax/HyperCLOVAX-SEED-Think-14B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me? Are you okay?" The man was confused and answered, "Yes." Then the woman asked.
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :]) * self.config.logits_scaling

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HyperCLOVAXConfig",
    "HyperCLOVAXPreTrainedModel",
    "HyperCLOVAXModel",
    "HyperCLOVAXForCausalLM",
]
