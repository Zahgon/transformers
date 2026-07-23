
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="caidas/swin2sr-classicalsr-x2-64")
@strict
class Swin2SRConfig(PreTrainedConfig):

    model_type = "swin2sr"

    attribute_map = {
        "hidden_size": "embed_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    image_size: int | list[int] | tuple[int, int] = 64
    patch_size: int | list[int] | tuple[int, int] = 1
    num_channels: int = 3
    num_channels_out: int | None = None
    embed_dim: int = 180
    depths: list[int] | tuple[int, ...] = (6, 6, 6, 6, 6, 6)
    num_heads: list[int] | tuple[int, ...] = (6, 6, 6, 6, 6, 6)
    window_size: int = 8
    mlp_ratio: float = 2.0
    qkv_bias: bool = True
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    drop_path_rate: float | int = 0.1
    hidden_act: str = "gelu"
    use_absolute_embeddings: bool = False
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    upscale: int = 2
    img_range: float = 1.0
    resi_connection: str = "1conv"
    upsampler: str = "pixelshuffle"

    def __post_init__(self, **kwargs):
        self.num_channels_out = self.num_channels if self.num_channels_out is None else self.num_channels_out
        self.num_layers = len(self.depths)
        super().__post_init__(**kwargs)


__all__ = ["Swin2SRConfig"]
