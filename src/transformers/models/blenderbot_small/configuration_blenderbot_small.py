
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/blenderbot_small-90M")
@strict
class BlenderbotSmallConfig(PreTrainedConfig):

    model_type = "blenderbot-small"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 50265
    max_position_embeddings: int = 512
    encoder_layers: int = 8
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 16
    decoder_layers: int = 8
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 512
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int | None = 1
    scale_embedding: bool = False
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    forced_eos_token_id: int | list[int] | None = 2
    is_decoder: bool = False
    tie_word_embeddings: bool = True


__all__ = ["BlenderbotSmallConfig"]
