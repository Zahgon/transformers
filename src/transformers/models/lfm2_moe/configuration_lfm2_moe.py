

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="LiquidAI/LFM2-8B-A1B")
@strict
class Lfm2MoeConfig(PreTrainedConfig):

    model_type = "lfm2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_ep_plan = {
        "layers.*.feed_forward.gate": "ep_router",
        "layers.*.feed_forward.experts.gate_up_proj": "grouped_gemm",
        "layers.*.feed_forward.experts.down_proj": "grouped_gemm",
        "layers.*.feed_forward.experts": "moe_tp_experts",
    }
    default_theta = 1000000.0

    vocab_size: int = 65536
    hidden_size: int = 2048
    intermediate_size: int = 7168
    moe_intermediate_size: int = 1792
    num_hidden_layers: int = 32
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    max_position_embeddings: int = 128_000
    initializer_range: float = 0.02
    use_cache: bool = True
    norm_eps: float = 0.00001
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    conv_bias: bool = False
    conv_L_cache: int = 3
    num_dense_layers: int = 2
    num_experts_per_tok: int = 4
    num_experts: int = 32
    use_expert_bias: bool = True
    routed_scaling_factor: float = 1.0
    norm_topk_prob: bool = True
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.tie_word_embeddings = kwargs.pop("tie_embedding", self.tie_word_embeddings)
        super().__post_init__(**kwargs)


__all__ = ["Lfm2MoeConfig"]
