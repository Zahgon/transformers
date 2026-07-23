
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/prophetnet-large-uncased")
@strict
class ProphetNetConfig(PreTrainedConfig):

    model_type = "prophetnet"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "num_encoder_attention_heads",
    }

    activation_dropout: float | int = 0.1
    activation_function: str = "gelu"
    vocab_size: int = 30522
    hidden_size: int = 1024
    encoder_ffn_dim: int = 4096
    num_encoder_layers: int = 12
    num_encoder_attention_heads: int = 16
    decoder_ffn_dim: int = 4096
    num_decoder_layers: int = 12
    num_decoder_attention_heads: int = 16
    attention_dropout: float | int = 0.1
    dropout: float | int = 0.1
    max_position_embeddings: int = 512
    init_std: float = 0.02
    is_encoder_decoder: bool = True
    add_cross_attention: bool = True
    decoder_start_token_id: int | None = 0
    ngram: int = 2
    num_buckets: int = 32
    relative_max_distance: int = 128
    disable_ngram_loss: bool = False
    eps: float = 0.0
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    is_decoder: bool = False
    tie_word_embeddings: bool = True

    @property
    def num_hidden_layers(self) -> int:
        pass

    @num_hidden_layers.setter
    def num_hidden_layers(self, value):
        raise NotImplementedError(
            "This model does not support the setting of `num_hidden_layers`. Please set `num_encoder_layers` and"
            " `num_decoder_layers`."
        )


__all__ = ["ProphetNetConfig"]
