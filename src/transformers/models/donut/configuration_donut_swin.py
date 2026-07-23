
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="naver-clova-ix/donut-base")
@strict
class DonutSwinConfig(PreTrainedConfig):

    model_type = "donut-swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 4
    num_channels: int = 3
    embed_dim: int = 96
    depths: list[int] | tuple[int, ...] = (2, 2, 6, 2)
    num_heads: list[int] | tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.1
    hidden_act: str = "gelu"
    use_absolute_embeddings: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5

    def __post_init__(self, **kwargs):
        self.num_layers = len(self.depths)
        self.hidden_size = int(self.embed_dim * 2 ** (len(self.depths) - 1))
        super().__post_init__(**kwargs)


__all__ = ["DonutSwinConfig"]
