
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/gpt_bigcode")
@strict
class GPTBigCodeConfig(PreTrainedConfig):

    model_type = "gpt_bigcode"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 50257
    n_positions: int = 1024
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_inner: int | None = None
    activation_function: str = "gelu_pytorch_tanh"
    resid_pdrop: float | int = 0.1
    embd_pdrop: float | int = 0.1
    attn_pdrop: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    pad_token_id: int | None = None
    attention_softmax_in_fp32: bool = True
    scale_attention_softmax_in_fp32: bool = True
    multi_query: bool = True
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = 1 if self.multi_query else self.n_head
        super().__post_init__(**kwargs)


__all__ = ["GPTBigCodeConfig"]
