
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="Helsinki-NLP/opus-mt-en-de")
@strict
class MarianConfig(PreTrainedConfig):

    model_type = "marian"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 58101
    decoder_vocab_size: int | None = None
    max_position_embeddings: int = 1024
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
    decoder_start_token_id: int = 58100
    scale_embedding: bool = False
    pad_token_id: int | None = 58100
    eos_token_id: int | list[int] | None = 0
    bos_token_id: int | None = None
    forced_eos_token_id: int | list[int] | None = 0
    share_encoder_decoder_embeddings: bool = True
    is_decoder: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.decoder_vocab_size = self.decoder_vocab_size or self.vocab_size
        super().__post_init__(**kwargs)


__all__ = ["MarianConfig"]
