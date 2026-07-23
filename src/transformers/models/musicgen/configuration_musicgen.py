
from typing import ClassVar

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="facebook/musicgen-small")
@strict
class MusicgenDecoderConfig(PreTrainedConfig):
    model_type = "musicgen_decoder"
    base_config_key = "decoder_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    vocab_size: int = 2048
    max_position_embeddings: int = 2048
    num_hidden_layers: int = 24
    ffn_dim: int = 4096
    num_attention_heads: int = 16
    layerdrop: float | int = 0.0
    use_cache: bool = True
    activation_function: str = "gelu"
    hidden_size: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    initializer_factor: float = 0.02
    scale_embedding: bool = False
    num_codebooks: int = 4
    audio_channels: int = 1
    pad_token_id: int | None = 2048
    bos_token_id: int | None = 2048
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = False
    is_decoder: bool = False
    add_cross_attention: bool = False
    cross_attention_hidden_size: int | None = None

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="facebook/musicgen-small")
@strict
class MusicgenConfig(PreTrainedConfig):

    model_type: ClassVar[str] = "musicgen"
    sub_configs: ClassVar[dict[str, type[PreTrainedConfig]]] = {
        "text_encoder": AutoConfig,
        "audio_encoder": AutoConfig,
        "decoder": MusicgenDecoderConfig,
    }
    has_no_defaults_at_init: ClassVar[bool] = True

    text_encoder: dict | PreTrainedConfig = None
    audio_encoder: dict | PreTrainedConfig = None
    decoder: dict | PreTrainedConfig = None
    initializer_factor: float = 0.02

    def __post_init__(self, **kwargs):
        if isinstance(self.text_encoder, dict):
            text_encoder_model_type = self.text_encoder.pop("model_type")
            self.text_encoder = AutoConfig.for_model(text_encoder_model_type, **self.text_encoder)
        elif self.text_encoder is None:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because text_encoder is not passed"
            )

        if isinstance(self.audio_encoder, dict):
            audio_encoder_model_type = self.audio_encoder.pop("model_type")
            self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **self.audio_encoder)
        elif self.audio_encoder is None:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because audio_encoder is not passed"
            )

        if isinstance(self.decoder, dict):
            self.decoder = MusicgenDecoderConfig(**self.decoder)
        elif self.decoder is None:
            self.decoder = MusicgenDecoderConfig()

        self.is_encoder_decoder = True
        super().__post_init__(**kwargs)

    @property
    def sampling_rate(self):
        pass


__all__ = ["MusicgenConfig", "MusicgenDecoderConfig"]
