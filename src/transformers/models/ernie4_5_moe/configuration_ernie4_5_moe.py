
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="baidu/ERNIE-4.5-21B-A3B-PT")
@strict
class Ernie4_5_MoeConfig(PreTrainedConfig):

    model_type = "ernie4_5_moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_experts": "moe_num_experts", "num_experts_per_tok": "moe_k"}
    default_theta = 500000.0

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
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

    vocab_size: int = 103424
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    hidden_size: int = 2560
    intermediate_size: int = 12288
    num_hidden_layers: int = 28
    num_attention_heads: int = 20
    num_key_value_heads: int | None = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    use_bias: bool | None = False
    moe_intermediate_size: int = 1536
    moe_k: int | None = 6
    moe_num_experts: int | None = 64
    moe_num_shared_experts: int | None = 2
    moe_layer_start_index: int | None = 1
    moe_layer_end_index: int | None = -1
    moe_layer_interval: int | None = 1
    moe_norm_min: float | None = 1e-12
    output_router_logits: bool | None = False
    router_aux_loss_coef: float | None = 0.001

    def __post_init__(self, **kwargs):
        self.moe_layer_end_index = (
            self.num_hidden_layers - 1 if self.moe_layer_end_index == -1 else self.moe_layer_end_index
        )
        super().__post_init__(**kwargs)


__all__ = ["Ernie4_5_MoeConfig"]
