from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="CohereForAI/c4ai-command-r-v01")
@strict
class Cohere2Config(PreTrainedConfig):

    model_type = "cohere2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
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

    vocab_size: int = 256000
    hidden_size: int = 8192
    intermediate_size: int = 22528
    logit_scale: float = 0.0625
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 5
    eos_token_id: int | list[int] | None = 255001
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        self.head_dim = self.hidden_size // self.num_attention_heads

        if self.layer_types is None:
            _sliding_window_pattern = kwargs.pop("sliding_window_pattern", 4)
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % _sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


__all__ = ["Cohere2Config"]
