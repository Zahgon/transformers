
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/layoutlmv3-base")
@strict
class LayoutLMv3Config(PreTrainedConfig):

    model_type = "layoutlmv3"

    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    max_2d_position_embeddings: int = 1024
    coordinate_size: int = 128
    shape_size: int = 128
    has_relative_attention_bias: bool = True
    rel_pos_bins: int = 32
    max_rel_pos: int = 128
    rel_2d_pos_bins: int = 64
    max_rel_2d_pos: int = 256
    has_spatial_attention_bias: bool = True
    text_embed: bool = True
    visual_embed: bool = True
    input_size: int = 224
    num_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    classifier_dropout: float | int | None = None


__all__ = ["LayoutLMv3Config"]
