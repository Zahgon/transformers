
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/bigbird-pegasus-large-arxiv")
@strict
class BigBirdPegasusConfig(PreTrainedConfig):

    model_type = "bigbird_pegasus"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "attention_probs_dropout_prob": "attention_dropout",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 96103
    max_position_embeddings: int = 4096
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
    activation_function: str = "gelu_new"
    d_model: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int = 2
    classifier_dropout: float | int = 0.0
    scale_embedding: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 2
    eos_token_id: int | list[int] | None = 1
    attention_type: str = "block_sparse"  # only for encoder
    block_size: int = 64
    num_random_blocks: int = 3
    use_bias: bool = False
    is_decoder: bool = False
    tie_word_embeddings: bool = True


__all__ = ["BigBirdPegasusConfig"]
