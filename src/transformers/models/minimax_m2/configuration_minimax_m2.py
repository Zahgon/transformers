

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="MiniMaxAI/MiniMax-Text-01-hf")
@strict
class MiniMaxM2Config(PreTrainedConfig):

    model_type = "minimax_m2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",
        "layers.*.self_attn.k_proj": "colwise_gather_output",
        "layers.*.self_attn.v_proj": "colwise_gather_output",
        "layers.*.self_attn.o_proj": "rowwise_split_input",
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
        "num_experts": "num_local_experts",
    }
    default_theta = 5000000.0

    vocab_size: int = 200064
    hidden_size: int = 3072
    intermediate_size: int = 1536
    num_hidden_layers: int = 62
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 196608
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-06
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 200034
    eos_token_id: int | list[int] | None = 200020
    tie_word_embeddings: bool = False
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 8
    num_local_experts: int = 256
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    rope_parameters: RopeParameters | dict | None = None


__all__ = ["MiniMaxM2Config"]
