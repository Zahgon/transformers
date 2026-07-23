
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/mms-tts-eng")
@strict
class VitsConfig(PreTrainedConfig):

    model_type = "vits"

    vocab_size: int = 38
    hidden_size: int = 192
    num_hidden_layers: int = 6
    num_attention_heads: int = 2
    window_size: int = 4
    use_bias: bool = True
    ffn_dim: int = 768
    layerdrop: float | int = 0.1
    ffn_kernel_size: int = 3
    flow_size: int = 192
    spectrogram_bins: int = 513
    hidden_act: str = "relu"
    hidden_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_stochastic_duration_prediction: bool = True
    num_speakers: int = 1
    speaker_embedding_size: int = 0
    upsample_initial_channel: int = 512
    upsample_rates: list[int] | tuple[int, ...] = (8, 8, 2, 2)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (16, 16, 4, 4)
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    leaky_relu_slope: float = 0.1
    depth_separable_channels: int = 2
    depth_separable_num_layers: int = 3
    duration_predictor_flow_bins: int = 10
    duration_predictor_tail_bound: float = 5.0
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout: float | int = 0.5
    duration_predictor_num_flows: int = 4
    duration_predictor_filter_channels: int = 256
    prior_encoder_num_flows: int = 4
    prior_encoder_num_wavenet_layers: int = 4
    posterior_encoder_num_wavenet_layers: int = 16
    wavenet_kernel_size: int = 5
    wavenet_dilation_rate: int = 1
    wavenet_dropout: float | int = 0.0
    speaking_rate: float | int = 1.0
    noise_scale: float = 0.667
    noise_scale_duration: float = 0.8
    sampling_rate: int = 16_000
    pad_token_id: int | None = None

    def validate_architecture(self):
        pass


__all__ = ["VitsConfig"]
