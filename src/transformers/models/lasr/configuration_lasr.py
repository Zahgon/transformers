
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/medasr")
@strict
class LasrEncoderConfig(PreTrainedConfig):

    model_type = "lasr_encoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    hidden_size: int = 512
    num_hidden_layers: int = 17
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_act: str = "silu"
    attention_bias: bool = False
    convolution_bias: bool = False
    conv_kernel_size: int = 32
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 128
    subsampling_conv_kernel_size: int = 5
    subsampling_conv_stride: int = 2
    dropout: float | int = 0.1
    dropout_positions: float | int = 0.0
    layerdrop: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    max_position_embeddings: int = 10000
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    feed_forward_residual_weights: list[float] | tuple[float, ...] = (1.5, 0.5)
    conv_residual_weights: list[float] | tuple[float, ...] = (2.0, 1.0)
    batch_norm_momentum: float = 0.01
    rope_parameters: dict | None = None

    def __post_init__(self, **kwargs):
        self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="google/medasr")
@strict
class LasrCTCConfig(PreTrainedConfig):

    model_type = "lasr_ctc"
    sub_configs = {"encoder_config": LasrEncoderConfig}

    vocab_size: int = 512
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    encoder_config: dict | PreTrainedConfig | None = None
    pad_token_id: int = 0

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder_config, dict):
            self.encoder_config = LasrEncoderConfig(**self.encoder_config)
        elif self.encoder_config is None:
            self.encoder_config = LasrEncoderConfig()
        self.initializer_range = self.encoder_config.initializer_range
        super().__post_init__(**kwargs)

    @property
    def inputs_to_logits_ratio(self):
        pass


__all__ = ["LasrEncoderConfig", "LasrCTCConfig"]
