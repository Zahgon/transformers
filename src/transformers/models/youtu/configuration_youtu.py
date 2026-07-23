

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="tencent/Youtu-LLM-2B")
@strict
class YoutuConfig(PreTrainedConfig):

    model_type = "youtu"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    attribute_map = {}

    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 1536
    qk_rope_head_dim: int = 64
    v_head_dim: int | None = 128
    qk_nope_head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float | None = None
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 128000
    eos_token_id: int | list[int] | None = 128001
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    rope_interleave: bool | None = True
    attention_bias: bool = False
    attention_dropout: float | int | None = 0.0
    embedding_initializer_range: float | None = None

    def __post_init__(self, **kwargs):
        if self.initializer_range is None:
            if self.hidden_size != 0:
                self.initializer_range = 2.0 / (5.0 * self.hidden_size) ** 0.5
            else:
                self.initializer_range = 0.02

        self.embedding_initializer_range = self.embedding_initializer_range or 2.0 * self.initializer_range
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_rope_head_dim
        super().__post_init__(**kwargs)


__all__ = ["YoutuConfig"]
