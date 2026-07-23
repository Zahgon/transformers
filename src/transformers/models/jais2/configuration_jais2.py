

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring
from ...utils.type_validators import interval


@auto_docstring(checkpoint="inceptionai/Jais-2-8B-Chat")
@strict
class Jais2Config(PreTrainedConfig):

    model_type = "jais2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 150272
    hidden_size: int = 3328
    intermediate_size: int = 26624
    num_hidden_layers: int = 32
    num_attention_heads: int = 26
    num_key_value_heads: int | None = None
    hidden_act: str = "relu2"
    max_position_embeddings: int = 8192
    initializer_range: float = interval(min=0.0, max=1.0)(default=0.02)
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 150024
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = True
    attention_dropout: int | float | None = 0.0
    mlp_bias: bool = True
    head_dim: int | None = None
    layer_norm_eps: float = 1e-5

    def __post_init__(self, **kwargs):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["Jais2Config"]
