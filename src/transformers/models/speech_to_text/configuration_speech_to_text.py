
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/s2t-small-librispeech-asr")
@strict
class Speech2TextConfig(PreTrainedConfig):

    model_type = "speech_to_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 10000
    encoder_layers: int = 12
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 4
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 4
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "relu"
    d_model: int = 256
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int = 2
    scale_embedding: bool = True
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    max_source_positions: int = 6000
    max_target_positions: int = 1024
    num_conv_layers: int = 2
    conv_kernel_sizes: list[int] | tuple[int, ...] = (5, 5)
    conv_channels: int = 1024
    input_feat_per_channel: int = 80
    input_channels: int = 1
    tie_word_embeddings: bool = True

    def validate_architecture(self):
        pass


__all__ = ["Speech2TextConfig"]
