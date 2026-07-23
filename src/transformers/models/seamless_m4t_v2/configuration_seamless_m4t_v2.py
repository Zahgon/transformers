
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/hf-seamless-m4t-medium")
@strict
class SeamlessM4Tv2Config(PreTrainedConfig):

    model_type = "seamless_m4t_v2"
    attribute_map = {"num_hidden_layers": "decoder_layers", "num_attention_heads": "decoder_attention_heads"}

    vocab_size: int = 256102
    t2u_vocab_size: int = 10082
    char_vocab_size: int = 10943
    hidden_size: int = 1024
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    max_position_embeddings: int = 4096
    is_encoder_decoder: bool = True
    encoder_layerdrop: float | int = 0.05
    decoder_layerdrop: float | int = 0.05
    activation_function: str = "relu"
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    scale_embedding: bool = True
    encoder_layers: int = 24
    encoder_ffn_dim: int = 8192
    encoder_attention_heads: int = 16
    decoder_layers: int = 24
    decoder_ffn_dim: int = 8192
    decoder_attention_heads: int = 16
    decoder_start_token_id: int = 3
    max_new_tokens: int | None = 256
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 3
    speech_encoder_layers: int = 24
    speech_encoder_attention_heads: int = 16
    speech_encoder_intermediate_size: int = 4096
    speech_encoder_hidden_act: str = "swish"
    speech_encoder_dropout: float | int = 0.0
    add_adapter: bool = True
    speech_encoder_layerdrop: float | int = 0.1
    feature_projection_input_dim: int = 160
    adaptor_kernel_size: int = 8
    adaptor_stride: int = 8
    adaptor_dropout: float | int = 0.1
    num_adapter_layers: int = 1
    position_embeddings_type: str = "relative_key"
    conv_depthwise_kernel_size: int = 31
    left_max_position_embeddings: int = 64
    right_max_position_embeddings: int = 8
    speech_encoder_chunk_size: int = 20000
    speech_encoder_left_chunk_num: int = 128
    t2u_bos_token_id: int | None = 0
    t2u_pad_token_id: int | None = 1
    t2u_eos_token_id: int | list[int] | None = 2
    t2u_encoder_layers: int = 6
    t2u_encoder_ffn_dim: int = 8192
    t2u_encoder_attention_heads: int = 16
    t2u_decoder_layers: int = 6
    t2u_decoder_ffn_dim: int = 8192
    t2u_decoder_attention_heads: int = 16
    t2u_max_position_embeddings: int = 4096
    t2u_variance_predictor_embed_dim: int = 1024
    t2u_variance_predictor_hidden_dim: int = 256
    t2u_variance_predictor_kernel_size: int = 3
    t2u_variance_pred_dropout: float | int = 0.5
    sampling_rate: int = 16000
    upsample_initial_channel: int = 512
    upsample_rates: list[int] | tuple[int, ...] = (5, 4, 4, 2, 2)
    upsample_kernel_sizes: list[int] | tuple[int, ...] = (11, 8, 8, 4, 4)
    resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11)
    resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    leaky_relu_slope: float = 0.1
    unit_hifi_gan_vocab_size: int = 10000
    unit_embed_dim: int = 1280
    lang_embed_dim: int = 256
    spkr_embed_dim: int = 256
    vocoder_num_langs: int = 36
    vocoder_num_spkrs: int = 200
    variance_predictor_kernel_size: int = 3
    var_pred_dropout: float | int = 0.5
    vocoder_offset: int = 4
    tie_word_embeddings: bool = True


__all__ = ["SeamlessM4Tv2Config"]
