
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="espnet/fastspeech2_conformer")
@strict
class FastSpeech2ConformerConfig(PreTrainedConfig):

    model_type = "fastspeech2_conformer"
    base_config_key = "model_config"
    attribute_map = {"num_hidden_layers": "encoder_layers", "num_attention_heads": "encoder_num_attention_heads"}

    hidden_size: int = 384
    vocab_size: int = 78
    num_mel_bins: int = 80
    encoder_num_attention_heads: int = 2
    encoder_layers: int = 4
    encoder_linear_units: int = 1536
    decoder_layers: int = 4
    decoder_num_attention_heads: int = 2
    decoder_linear_units: int = 1536
    speech_decoder_postnet_layers: int = 5
    speech_decoder_postnet_units: int = 256
    speech_decoder_postnet_kernel: int = 5
    positionwise_conv_kernel_size: int = 3
    encoder_normalize_before: bool = False
    decoder_normalize_before: bool = False
    encoder_concat_after: bool = False
    decoder_concat_after: bool = False
    reduction_factor: int = 1
    speaking_speed: float = 1.0
    use_macaron_style_in_conformer: bool = True
    use_cnn_in_conformer: bool = True
    encoder_kernel_size: int = 7
    decoder_kernel_size: int = 31
    duration_predictor_layers: int = 2
    duration_predictor_channels: int = 256
    duration_predictor_kernel_size: int = 3
    energy_predictor_layers: int = 2
    energy_predictor_channels: int = 256
    energy_predictor_kernel_size: int = 3
    energy_predictor_dropout: float | int = 0.5
    energy_embed_kernel_size: int = 1
    energy_embed_dropout: float | int = 0.0
    stop_gradient_from_energy_predictor: bool = False
    pitch_predictor_layers: int = 5
    pitch_predictor_channels: int = 256
    pitch_predictor_kernel_size: int = 5
    pitch_predictor_dropout: float | int = 0.5
    pitch_embed_kernel_size: int = 1
    pitch_embed_dropout: float | int = 0.0
    stop_gradient_from_pitch_predictor: bool = True
    encoder_dropout_rate: float | int = 0.2
    encoder_positional_dropout_rate: float | int = 0.2
    encoder_attention_dropout_rate: float | int = 0.2
    decoder_dropout_rate: float | int = 0.2
    decoder_positional_dropout_rate: float | int = 0.2
    decoder_attention_dropout_rate: float | int = 0.2
    duration_predictor_dropout_rate: float | int = 0.2
    speech_decoder_postnet_dropout: float | int = 0.5
    max_source_positions: int = 5000
    use_masking: bool = True
    use_weighted_masking: bool = False
    num_speakers: int | None = None
    num_languages: int | None = None
    speaker_embed_dim: int | None = None
    is_encoder_decoder: bool = True
    convolution_bias: bool = True

    def __post_init__(self, **kwargs):
        self.encoder_config = {
            "num_attention_heads": self.encoder_num_attention_heads,
            "layers": self.encoder_layers,
            "kernel_size": self.encoder_kernel_size,
            "attention_dropout_rate": self.encoder_attention_dropout_rate,
            "dropout_rate": self.encoder_dropout_rate,
            "positional_dropout_rate": self.encoder_positional_dropout_rate,
            "linear_units": self.encoder_linear_units,
            "normalize_before": self.encoder_normalize_before,
            "concat_after": self.encoder_concat_after,
        }
        self.decoder_config = {
            "num_attention_heads": self.decoder_num_attention_heads,
            "layers": self.decoder_layers,
            "kernel_size": self.decoder_kernel_size,
            "attention_dropout_rate": self.decoder_attention_dropout_rate,
            "dropout_rate": self.decoder_dropout_rate,
            "positional_dropout_rate": self.decoder_positional_dropout_rate,
            "linear_units": self.decoder_linear_units,
            "normalize_before": self.decoder_normalize_before,
            "concat_after": self.decoder_concat_after,
        }
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="espnet/fastspeech2_conformer_with_hifigan")
@strict
class FastSpeech2ConformerHifiGanConfig(PreTrainedConfig):

    model_type = "fastspeech2_conformer_hifigan"
    base_config_key = "vocoder_config"

    model_in_dim: int = 80
    upsample_initial_channel: int = 512
    upsample_rates: list[int] | tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (16, 16, 4, 4)
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple | None = None
    initializer_range: float = 0.01
    leaky_relu_slope: float = 0.1
    normalize_before: bool = True

    def __post_init__(self, **kwargs):
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="espnet/fastspeech2_conformer_with_hifigan")
@strict
class FastSpeech2ConformerWithHifiGanConfig(PreTrainedConfig):

    model_type = "fastspeech2_conformer_with_hifigan"
    sub_configs = {"model_config": FastSpeech2ConformerConfig, "vocoder_config": FastSpeech2ConformerHifiGanConfig}

    model_config: dict | PreTrainedConfig | None = None
    vocoder_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.model_config is None:
            self.model_config = FastSpeech2ConformerConfig()
            logger.info("model_config is None. initializing the model with default values.")
        elif isinstance(self.model_config, dict):
            self.model_config = FastSpeech2ConformerConfig(**self.model_config)

        if self.vocoder_config is None:
            self.vocoder_config = FastSpeech2ConformerHifiGanConfig()
            logger.info("vocoder_config is None. initializing the coarse model with default values.")
        elif isinstance(self.vocoder_config, dict):
            self.vocoder_config = FastSpeech2ConformerHifiGanConfig(**self.vocoder_config)

        super().__post_init__(**kwargs)


__all__ = ["FastSpeech2ConformerConfig", "FastSpeech2ConformerHifiGanConfig", "FastSpeech2ConformerWithHifiGanConfig"]
