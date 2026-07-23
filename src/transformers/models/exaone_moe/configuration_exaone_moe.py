from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="LGAI-EXAONE/K-EXAONE-236B-A23B")
@strict
class ExaoneMoeConfig(PreTrainedConfig):

    model_type = "exaone_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
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

    vocab_size: int = 102400
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 53
    pad_token_id: int | None = 0
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    attention_dropout: float | int = 0.0
    sliding_window: int = 4096
    sliding_window_pattern: str | int | None = 4
    layer_types: list[str] | None = None

    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    mlp_layer_types: list[str] | None = None
    first_k_dense_replace: int = 1
    moe_intermediate_size: int = 1024
    num_experts: int = 64
    num_experts_per_tok: int = 8
    num_shared_experts: int = 1
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 2.5
    n_group: int = 1
    topk_group: int = 1

    def __post_init__(self, **kwargs):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = [
                "dense" if i < self.first_k_dense_replace else "sparse" for i in range(self.num_hidden_layers)
            ]
        if self.sliding_window is None:
            self.sliding_window_pattern = 0
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if ((i + 1) % (self.sliding_window_pattern) != 0 and i < self.num_hidden_layers)
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


__all__ = ["ExaoneMoeConfig"]
