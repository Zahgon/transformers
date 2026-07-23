
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="upstage/Solar-Open-100B")
@strict
class SolarOpenConfig(PreTrainedConfig):

    model_type = "solar_open"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
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
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 196608
    hidden_size: int = 4096
    num_hidden_layers: int = 48
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 1280
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    default_theta = 1_000_000.0
    head_dim: int = 128

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 1.0)
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["SolarOpenConfig"]
