
import functools
import operator

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/speecht5_asr")
@strict
class SpeechT5Config(PreTrainedConfig):

    model_type = "speecht5"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "num_hidden_layers": "encoder_layers"}

    vocab_size: int = 81
    hidden_size: int = 768
    encoder_layers: int = 12
    encoder_attention_heads: int = 12
    encoder_ffn_dim: int = 3072
    encoder_layerdrop: float | int = 0.1
    decoder_layers: int = 6
    decoder_ffn_dim: int = 3072
    decoder_attention_heads: int = 12
    decoder_layerdrop: float | int = 0.1
    hidden_act: str = "gelu"
    positional_dropout: float | int = 0.1
    hidden_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    scale_embedding: bool = False
    feat_extract_norm: str = "group"
    feat_proj_dropout: float | int = 0.0
    feat_extract_activation: str = "gelu"
    conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    apply_spec_augment: bool = True
    mask_time_prob: float | int = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float | int = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    decoder_start_token_id: int | None = 2
    num_mel_bins: int = 80
    speech_decoder_prenet_layers: int = 2
    speech_decoder_prenet_units: int = 256
    speech_decoder_prenet_dropout: float | int = 0.5
    speaker_embedding_dim: int = 512
    speech_decoder_postnet_layers: int = 5
    speech_decoder_postnet_units: int = 256
    speech_decoder_postnet_kernel: int = 5
    speech_decoder_postnet_dropout: float | int = 0.5
    reduction_factor: int = 2
    max_speech_positions: int = 4000
    max_text_positions: int = 450
    encoder_max_relative_position: int = 160
    use_guided_attention_loss: bool = True
    guided_attention_loss_num_heads: int = 2
    guided_attention_loss_sigma: float = 0.4
    guided_attention_loss_scale: float = 10.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_feat_extract_layers = len(self.conv_dim)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    def inputs_to_logits_ratio(self):
        pass


@auto_docstring(checkpoint="microsoft/speecht5_asr")
@strict
class SpeechT5HifiGanConfig(PreTrainedConfig):

    model_type = "speecht5_hifigan"

    model_in_dim: int = 80
    sampling_rate: int = 16000
    upsample_initial_channel: int = 512
    upsample_rates: list[int] | tuple[int, ...] = (4, 4, 4, 4)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (8, 8, 8, 8)
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    initializer_range: float = 0.01
    leaky_relu_slope: float = 0.1
    normalize_before: bool = True


__all__ = ["SpeechT5Config", "SpeechT5HifiGanConfig"]
