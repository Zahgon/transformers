
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/ijepa_vith14_1k")
@strict
class IJepaConfig(PreTrainedConfig):

    model_type = "ijepa"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    pooler_output_size: int | None = None
    pooler_act: str = "tanh"

    def __post_init__(self, **kwargs):
        self.pooler_output_size = self.pooler_output_size if self.pooler_output_size else self.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["IJepaConfig"]
