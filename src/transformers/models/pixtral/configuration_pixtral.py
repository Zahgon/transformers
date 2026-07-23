
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="mistral-labs/pixtral-12b")
@strict
class PixtralVisionConfig(PreTrainedConfig):

    model_type = "pixtral"

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.head_dim = self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["PixtralVisionConfig"]
