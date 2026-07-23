
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="nari-labs/Dia-1.6B")
@strict
class DiaEncoderConfig(PreTrainedConfig):
    model_type = "dia_encoder"

    max_position_embeddings: int = 1024
    num_hidden_layers: int = 12
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 16
    head_dim: int = 128
    intermediate_size: int = 4096
    norm_eps: float = 1e-5
    vocab_size: int = 256
    hidden_act: str = "silu"
    rope_parameters: dict | None = None
    initializer_range: float = 0.02


@auto_docstring(checkpoint="nari-labs/Dia-1.6B")
@strict
class DiaDecoderConfig(PreTrainedConfig):

    model_type = "dia_decoder"

    max_position_embeddings: int = 3072
    num_hidden_layers: int = 18
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    cross_num_attention_heads: int = 16
    cross_head_dim: int = 128
    cross_num_key_value_heads: int = 16
    cross_hidden_size: int = 1024
    norm_eps: float = 1e-5
    vocab_size: int = 1028
    hidden_act: str = "silu"
    num_channels: int = 9
    rope_parameters: RopeParameters | dict | None = None
    initializer_range: float = 0.02
    use_cache: bool = True
    is_encoder_decoder: bool = True
    pad_token_id: int | None = 1025
    eos_token_id: int | list[int] | None = 1024
    bos_token_id: int | None = 1026


@auto_docstring(checkpoint="nari-labs/Dia-1.6B")
@strict
class DiaConfig(PreTrainedConfig):

    model_type = "dia"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"encoder_config": DiaEncoderConfig, "decoder_config": DiaDecoderConfig}

    encoder_config: DiaEncoderConfig | dict | None = None
    decoder_config: DiaDecoderConfig | dict | None = None
    norm_eps: float = 1e-5
    is_encoder_decoder: bool = True
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    bos_token_id: int | None = None
    delay_pattern: list[int] | None = None
    initializer_range: float = 0.02
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = DiaEncoderConfig(**self.encoder_config)
        if isinstance(self.decoder_config, dict):
            self.decoder_config = DiaDecoderConfig(**self.decoder_config)

        self.encoder_config = self.encoder_config if self.encoder_config is not None else DiaEncoderConfig()
        self.decoder_config = self.decoder_config if self.decoder_config is not None else DiaDecoderConfig()
        self.delay_pattern = (
            self.delay_pattern if self.delay_pattern is not None else [0, 8, 9, 10, 11, 12, 13, 14, 15]
        )

        if self.pad_token_id is not None:
            logger.warning_once(
                "Passing `pad_token_id` to `DiaConfig` is deprecated. "
                "Please set it directly on `DiaDecoderConfig` instead."
            )
            self.decoder_config.pad_token_id = self.pad_token_id

        if self.eos_token_id is not None:
            logger.warning_once(
                "Passing `eos_token_id` to `DiaConfig` is deprecated. "
                "Please set it directly on `DiaDecoderConfig` instead."
            )
            self.decoder_config.eos_token_id = self.eos_token_id

        if self.bos_token_id is not None:
            logger.warning_once(
                "Passing `bos_token_id` to `DiaConfig` is deprecated. "
                "Please set it directly on `DiaDecoderConfig` instead."
            )
            self.decoder_config.bos_token_id = self.bos_token_id

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    def get_text_config(self, *args, **kwargs):
        """Defaulting to audio config as it's the decoder in this case which is usually the text backbone"""
        return self.decoder_config


__all__ = ["DiaConfig", "DiaEncoderConfig", "DiaDecoderConfig"]
