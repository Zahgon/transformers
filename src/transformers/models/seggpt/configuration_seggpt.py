
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="BAAI/seggpt-vit-large")
@strict
class SegGptConfig(PreTrainedConfig):

    model_type = "seggpt"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, ...] = (896, 448)
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    mlp_dim: int | None = None
    drop_path_rate: float | int = 0.1
    pretrain_image_size: int | list[int] | tuple[int, int] = 224
    decoder_hidden_size: int = 64
    use_relative_position_embeddings: bool = True
    merge_index: int = 2
    intermediate_hidden_state_indices: list[int] | tuple[int, ...] = (5, 11, 17, 23)
    beta: float = 0.01

    def __post_init__(self, **kwargs):
        self.mlp_dim = int(self.hidden_size * 4) if self.mlp_dim is None else self.mlp_dim
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["SegGptConfig"]
