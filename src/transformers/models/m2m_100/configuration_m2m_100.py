
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/m2m100_418M")
@strict
class M2M100Config(PreTrainedConfig):

    model_type = "m2m_100"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 128112
    max_position_embeddings: int = 1024
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.05
    decoder_layerdrop: float | int = 0.05
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "relu"
    d_model: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int | None = 2
    scale_embedding: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True


__all__ = ["M2M100Config"]
