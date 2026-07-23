
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="bigcode/starcoder2-7b")
@strict
class Starcoder2Config(PreTrainedConfig):

    model_type = "starcoder2"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.c_fc": "colwise",
        "layers.*.mlp.c_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 49152
    hidden_size: int = 3072
    intermediate_size: int = 12288
    num_hidden_layers: int = 30
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    hidden_act: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 4096
    initializer_range: float = 0.018042
    norm_epsilon: float = 1e-5
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    pad_token_id: int | None = None
    rope_parameters: RopeParameters | dict | None = None
    sliding_window: int | None = None
    attention_dropout: float | int = 0.0
    residual_dropout: float | int = 0.0
    embedding_dropout: float | int = 0.0
    use_bias: bool = True
    tie_word_embeddings: bool = True


__all__ = ["Starcoder2Config"]
