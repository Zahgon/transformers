

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="THUDM/glm-4-9b-chat")
@strict
class GlmConfig(PreTrainedConfig):

    model_type = "glm"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_gather_output",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_split_input",  # input is replicated due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 151552
    hidden_size: int = 4096
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 2
    head_dim: int | None = 128
    hidden_act: str = "silu"
    attention_dropout: float | int | None = 0.0
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 0.00000015625
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int | None = 151329
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    attention_bias: bool = True

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        if self.eos_token_id is None:
            self.eos_token_id = [151329, 151336, 151338]
        super().__post_init__(**kwargs)


__all__ = ["GlmConfig"]
