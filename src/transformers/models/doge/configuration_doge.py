from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="SmallDoge/Doge-320M")
@strict
class DogeConfig(PreTrainedConfig):

    model_type = "doge"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.dt_proj": "rowwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.router_gate": "colwise_gather_output",
        "layers.*.mlp.down_embed": "rowwise_split_input",
        "layers.*.mlp.up_embed": "rowwise_split_input",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 32768
    hidden_size: int = 1024
    intermediate_size: int = 2048
    num_hidden_layers: int = 32
    hidden_dropout: float | int = 0.0
    hidden_act: str = "silu"
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-06
    use_cache: bool = True
    tie_word_embeddings: bool = False
    max_position_embeddings: int = 2048
    rope_parameters: RopeParameters | dict | None = None
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    attention_bias: bool = False
    attention_dropout: float | None = 0.0
    mlp_bias: bool = False
    sliding_window: int | None = None
    keep_window_size: int = 2048
    is_moe: bool = False
    num_experts: int = 16384
    num_experts_per_tok: int = 64
    norm_topk_prob: bool = False
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


__all__ = ["DogeConfig"]
