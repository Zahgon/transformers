
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="allenai/led-base-16384")
@strict
class LEDConfig(PreTrainedConfig):

    model_type = "led"
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
        "initializer_range": "init_std",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 50265
    max_encoder_position_embeddings: int = 16384
    max_decoder_position_embeddings: int = 1024
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int = 2
    classifier_dropout: float | int = 0.0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    attention_window: list[int] | int = 512
    tie_word_embeddings: bool = True


__all__ = ["LEDConfig"]
