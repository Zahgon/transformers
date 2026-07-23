

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="allenai/FlexOlmo-7x7B-1T")
@strict
class FlexOlmoConfig(PreTrainedConfig):

    model_type = "flex_olmo"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "num_experts"}
    default_theta = 500000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.k_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.v_proj": "colwise_gather_output",  # we need to replicate here due to the added norm on q and k
        "layers.*.self_attn.o_proj": "rowwise_split_input",  # input is replicated due to the added norm on q and k
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

    vocab_size: int = 100352
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-06
    use_cache: bool = True
    pad_token_id: int | None = 100277
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = 100257
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | None = 0.0
    num_experts_per_tok: int = 5
    num_experts: int = 7
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.01
    norm_topk_prob: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["FlexOlmoConfig"]
