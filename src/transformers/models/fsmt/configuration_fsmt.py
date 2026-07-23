
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/wmt19-en-ru")
@strict
class FSMTConfig(PreTrainedConfig):

    model_type = "fsmt"
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "vocab_size": "tgt_vocab_size",
        "num_hidden_layers": "encoder_layers",
    }

    langs: list[str] | tuple[str, ...] = ("en", "de")
    src_vocab_size: int = 42024
    tgt_vocab_size: int = 42024
    activation_function: str = "relu"
    d_model: int = 1024
    max_length: int = 200
    max_position_embeddings: int = 1024
    encoder_ffn_dim: int = 4096
    encoder_layers: int = 12
    encoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.0
    decoder_ffn_dim: int = 4096
    decoder_layers: int = 12
    decoder_attention_heads: int = 16
    decoder_layerdrop: float | int = 0.0
    attention_dropout: float | int = 0.0
    dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int | None = 2
    is_encoder_decoder: bool = True
    scale_embedding: bool = True
    tie_word_embeddings: bool = False
    num_beams: int = 5
    length_penalty: float = 1.0
    early_stopping: bool = False
    use_cache: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    forced_eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs):
        kwargs.pop("decoder", None)  # delete unused kwargs
        super().__post_init__(**kwargs)


__all__ = ["FSMTConfig"]
