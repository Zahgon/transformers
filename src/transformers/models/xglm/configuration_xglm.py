
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/xglm-564M")
@strict
class XGLMConfig(PreTrainedConfig):

    model_type = "xglm"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_attention_heads": "attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    vocab_size: int = 256008
    max_position_embeddings: int = 2048
    d_model: int = 1024
    ffn_dim: int = 4096
    num_layers: int = 24
    attention_heads: int = 16
    activation_function: str = "gelu"
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    layerdrop: float | int = 0.0
    init_std: float = 0.02
    scale_embedding: bool = True
    use_cache: bool = True
    decoder_start_token_id: int = 2
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True


__all__ = ["XGLMConfig"]
