

import torch
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache
from ...utils import auto_docstring
from ..gemma2.configuration_gemma2 import Gemma2Config
from ..gemma2.modeling_gemma2 import Gemma2Attention, Gemma2DecoderLayer, Gemma2ForCausalLM, Gemma2MLP, Gemma2RMSNorm


@auto_docstring(checkpoint="google/vaultgemma-1b")
@strict
class VaultGemmaConfig(Gemma2Config):

    use_bidirectional_attention = AttributeError()


class VaultGemmaRMSNorm(Gemma2RMSNorm):
    pass


class VaultGemmaMLP(Gemma2MLP):
    pass


class VaultGemmaAttention(Gemma2Attention):

    def __init__(self, config: VaultGemmaConfig, layer_idx: int):
        super().__init__()
        self.is_causal = True


class VaultGemmaDecoderLayer(Gemma2DecoderLayer):
    def __init__(self, **super_kwargs):
        super().__init__(**super_kwargs)
        del self.post_attention_layernorm
        del self.post_feedforward_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VaultGemmaForCausalLM(Gemma2ForCausalLM):
    pass


__all__ = [
    "VaultGemmaConfig",
    "VaultGemmaForCausalLM",
    "VaultGemmaModel",  # noqa: F822
    "VaultGemmaPreTrainedModel",  # noqa: F822
]
