

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/phi-1")
@strict
class PhiConfig(PreTrainedConfig):

    model_type = "phi"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.dense": "rowwise",
        "layers.*.mlp.fc1": "colwise",
        "layers.*.mlp.fc2": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "embed_dropout": (["inputs_embeds"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "final_layernorm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 51200
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    resid_pdrop: float | int = 0.0
    embd_pdrop: float | int = 0.0
    attention_dropout: float | int | None = 0.0
    hidden_act: str = "gelu_new"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    qk_layernorm: bool = False
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


__all__ = ["PhiConfig"]
