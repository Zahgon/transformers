
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/pegasus-x-large")
@strict
class PegasusXConfig(PreTrainedConfig):

    model_type = "pegasus_x"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 96103
    max_position_embeddings: int = 16384
    encoder_layers: int = 16
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 16
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.0
    decoder_layerdrop: float | int = 0.0
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int | None = 0
    scale_embedding: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    forced_eos_token_id: int | list[int] | None = 1
    num_global_tokens: int = 32
    block_size: int = 512
    stagger_local_blocks: bool = True
    tie_word_embeddings: bool = True


__all__ = ["PegasusXConfig"]
