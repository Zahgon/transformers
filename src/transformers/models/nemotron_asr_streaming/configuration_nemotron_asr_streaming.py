
import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrStreamingEncoderConfig(PreTrainedConfig):

    model_type = "nemotron_asr_streaming_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    intermediate_size: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = True
    convolution_bias: bool = True
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 80
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    dropout: float | int = 0.1
    dropout_positions: float | int = 0.0
    layerdrop: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    max_position_embeddings: int = 5000
    scale_input: bool = True
    initializer_range: float = 0.02

    sliding_window: int = 71
    default_num_lookahead_tokens: int = 13

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)

    @property
    def subsampling_out_hidden_size(self) -> int:
        pass


@auto_docstring(checkpoint="nvidia/nemotron-speech-streaming-en-0.6b")
@strict
class NemotronAsrStreamingConfig(PreTrainedConfig):

    model_type = "nemotron_asr_streaming"
    sub_configs = {"encoder_config": NemotronAsrStreamingEncoderConfig}
    vocab_size: int = 1025
    decoder_hidden_size: int = 640
    num_decoder_layers: int = 2
    hidden_act: str = "relu"
    max_symbols_per_step: int = 10
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int = 0
    blank_token_id: int = 1024
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = NemotronAsrStreamingEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = NemotronAsrStreamingEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)


__all__ = ["NemotronAsrStreamingConfig", "NemotronAsrStreamingEncoderConfig"]
