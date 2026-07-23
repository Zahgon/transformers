
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="dandelin/vilt-b32-mlm")
@strict
class ViltConfig(PreTrainedConfig):

    model_type = "vilt"

    vocab_size: int = 30522
    type_vocab_size: int = 2
    modality_type_vocab_size: int = 2
    max_position_embeddings: int = 40
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 384
    patch_size: int | list[int] | tuple[int, int] = 32
    num_channels: int = 3
    qkv_bias: bool = True
    max_image_length: int = -1
    tie_word_embeddings: bool = True
    num_images: int = -1
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        kwargs.pop("tie_word_embeddings", None)
        self.tie_word_embeddings = True  # force it
        super().__post_init__(**kwargs)


__all__ = ["ViltConfig"]
