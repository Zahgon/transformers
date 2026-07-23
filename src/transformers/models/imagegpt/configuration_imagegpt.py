
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="openai/imagegpt-small")
@strict
class ImageGPTConfig(PreTrainedConfig):

    model_type = "imagegpt"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 512 + 1  # add one for start of sentence (sos) token
    n_positions: int = 32 * 32
    n_embd: int = 512
    n_layer: int = 24
    n_head: int = 8
    n_inner: int | None = None
    activation_function: str = "quick_gelu"
    resid_pdrop: float | int = 0.1
    embd_pdrop: float | int = 0.1
    attn_pdrop: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    tie_word_embeddings: bool = False
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    add_cross_attention: bool = False
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None


__all__ = ["ImageGPTConfig"]
