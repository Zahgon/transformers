
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig, remap_legacy_layer_types
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="Qwen/Qwen3-Next-80B-A3B-Instruct")
@strict
class Qwen3NextConfig(PreTrainedConfig):

    model_type = "qwen3_next"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.shared_expert.gate_proj": "colwise",
        "layers.*.mlp.shared_expert.up_proj": "colwise",
        "layers.*.mlp.shared_expert.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.linear_attn.in_proj_qkvz": "colwise_gather_output",
        "layers.*.linear_attn.in_proj_ba": "colwise_gather_output",
        "layers.*.linear_attn.out_proj": "colwise_gather_output",
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

    vocab_size: int = 151936
    hidden_size: int = 2048
    intermediate_size: int = 5632
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    head_dim: int = 256
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 10
    num_experts: int = 512
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: list[int] | None = None
    layer_types: list[str] | None = None
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.25)  # assign default for BC
        self.mlp_only_layers = [] if self.mlp_only_layers is None else self.mlp_only_layers
        if self.layer_types is None:
            interval_pattern = kwargs.pop("full_attention_interval", 4)
            self.layer_types = [
                "linear_attention" if bool((i + 1) % interval_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = remap_legacy_layer_types(self.layer_types)

        super().__post_init__(**kwargs)


__all__ = ["Qwen3NextConfig"]
