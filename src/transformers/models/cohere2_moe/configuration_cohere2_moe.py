
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="CohereLabs/command-a-plus-05-2026")
@strict
class Cohere2MoeConfig(PreTrainedConfig):

    model_type = "cohere2_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
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

    vocab_size: int = 256000
    hidden_size: int = 8192
    intermediate_size: int = 22528
    logit_scale: float = 0.0625
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int | None = None
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 5
    eos_token_id: int | list[int] | None = 255001
    tie_word_embeddings: bool = True
    rope_theta: float | int = 10000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int | None = 4096
    num_experts_per_tok: int = 2
    num_experts: int = 8
    num_shared_experts: int = 0
    shared_expert_combination_strategy: str = "average"
    expert_selection_fn: str = "softmax"
    layer_types: list[str] | None = None
    mlp_layer_types: list | None = None
    prefix_dense_sliding_window_pattern: int = 1
    norm_topk_prob: bool = True
    prefix_dense_intermediate_size: int | None = None
    rms_norm_eps: float | None = None
    sliding_window_pattern: int = 4

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.standardize_rope_params()
        self.validate_rope()

        first_k_dense_replace = kwargs.pop("first_k_dense_replace", 0)

        if self.layer_types is None:
            prefix_layers = [
                "sliding_attention" if ((i + 1) % self.prefix_dense_sliding_window_pattern) != 0 else "full_attention"
                for i in range(first_k_dense_replace)
            ]
            rest_layers = [
                "sliding_attention" if ((i + 1) % self.sliding_window_pattern) != 0 else "full_attention"
                for i in range(self.num_hidden_layers - first_k_dense_replace)
            ]
            self.layer_types = prefix_layers + rest_layers

        self.validate_layer_type()

        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


__all__ = ["Cohere2MoeConfig"]
