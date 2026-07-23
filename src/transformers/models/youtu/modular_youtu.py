

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ..deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from ..deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
from ..llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..qwen3.modeling_qwen3 import Qwen3MLP


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="tencent/Youtu-LLM-2B")
@strict
class YoutuConfig(DeepseekV3Config):

    model_type = "youtu"
    base_model_tp_plan = {
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    attribute_map = {}

    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    max_position_embeddings: int = 131072
    initializer_range: float | None = None
    embedding_initializer_range: float | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128001
    tie_word_embeddings: bool = True

    n_shared_experts = AttributeError()
    n_routed_experts = AttributeError()
    routed_scaling_factor = AttributeError()
    n_group = AttributeError()
    topk_group = AttributeError()
    num_experts_per_tok = AttributeError()
    first_k_dense_replace = AttributeError()
    norm_topk_prob = AttributeError()
    pretraining_tp = AttributeError()
    moe_intermediate_size = AttributeError()
    num_mtp_layers = AttributeError()

    def __post_init__(self, **kwargs):
        if self.initializer_range is None:
            if self.hidden_size != 0:
                self.initializer_range = 2.0 / (5.0 * self.hidden_size) ** 0.5
            else:
                self.initializer_range = 0.02

        self.embedding_initializer_range = self.embedding_initializer_range or 2.0 * self.initializer_range
        super().__post_init__(**kwargs)


class YoutuRMSNorm(LlamaRMSNorm):
    pass


class YoutuRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class YoutuMLP(Qwen3MLP):
    pass


class YoutuAttention(DeepseekV3Attention):
    pass


class YoutuDecoderLayer(LlamaDecoderLayer):
    pass


class YoutuPreTrainedModel(LlamaPreTrainedModel, PreTrainedModel):
    @torch.no_grad()
    def _init_weights(self, module):
        PreTrainedModel._init_weights(self, module)
        std = getattr(self.config, "initializer_range", 0.02)
        embed_std = getattr(self.config, "embedding_initializer_range", 2 * std)
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=embed_std)
            if module.padding_idx is not None:
                init.zeros_(module.weight.data[module.padding_idx])


class YoutuModel(LlamaModel):
    pass


class YoutuForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "YoutuConfig",
    "YoutuPreTrainedModel",
    "YoutuModel",
    "YoutuForCausalLM",
]
