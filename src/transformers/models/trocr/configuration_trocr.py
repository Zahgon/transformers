
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/trocr-base-handwritten")
@strict
class TrOCRConfig(PreTrainedConfig):

    model_type = "trocr"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "decoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "decoder_layers",
    }

    vocab_size: int = 50265
    d_model: int = 1024
    decoder_layers: int = 12
    decoder_attention_heads: int = 16
    decoder_ffn_dim: int = 4096
    activation_function: str = "gelu"
    max_position_embeddings: int = 512
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    decoder_start_token_id: int = 2
    init_std: float = 0.02
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    scale_embedding: bool = False
    use_learned_position_embeddings: bool = True
    layernorm_embedding: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    cross_attention_hidden_size: int | None = None
    is_decoder: bool = False
    tie_word_embeddings: bool = True


__all__ = ["TrOCRConfig"]
