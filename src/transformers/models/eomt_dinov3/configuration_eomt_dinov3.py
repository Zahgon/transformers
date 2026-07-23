from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="tue-mps/coco_panoptic_eomt_large_640_dinov3")
@strict
class EomtDinov3Config(PreTrainedConfig):

    model_type = "eomt_dinov3"

    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 640
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    num_upscale_blocks: int = 2
    attention_dropout: float | int = 0.0
    num_blocks: int = 4
    no_object_weight: float = 0.1
    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    train_num_points: int = 12544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75
    num_queries: int = 200
    num_register_tokens: int = 4
    default_theta = 100.0
    intermediate_size: int = 4096
    rope_parameters: RopeParameters | dict | None = None
    query_bias: bool = True
    key_bias: bool = False
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    use_gated_mlp: bool = False
    pos_embed_shift: float | None = None
    pos_embed_jitter: float | None = None
    pos_embed_rescale: float | None = 2.0


__all__ = ["EomtDinov3Config"]
