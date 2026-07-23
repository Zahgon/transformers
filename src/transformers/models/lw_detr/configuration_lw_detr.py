import math

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin, consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="AnnaZhang/lwdetr_small_60e_coco")
@strict
class LwDetrViTConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "lw_detr_vit"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6

    image_size: int | list[int] | tuple[int, int] = 256
    pretrain_image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    qkv_bias: bool = True
    window_block_indices: list[int] | tuple[int, ...] = ()
    use_absolute_position_embeddings: bool = True
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    cae_init_values: float = 0.1
    num_windows: int = 16

    def __post_init__(self, **kwargs):
        self.num_windows_side = int(math.sqrt(self.num_windows))
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="AnnaZhang/lwdetr_small_60e_coco")
@strict
class LwDetrConfig(PreTrainedConfig):

    model_type = "lw_detr"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    projector_scale_factors: list[float] | tuple[float, ...] = ()
    hidden_expansion: float = 0.5
    c2f_num_blocks: int = 3
    activation_function: str = "silu"
    batch_norm_eps: float = 1e-5
    dropout: float | int = 0.0
    decoder_ffn_dim: int = 2048
    decoder_n_points: int = 4
    decoder_layers: int = 3
    decoder_self_attention_heads: int = 8
    decoder_cross_attention_heads: int = 16
    decoder_activation_function: str = "relu"
    num_queries: int = 300
    attention_bias: bool = True
    attention_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    group_detr: int = 13
    init_std: float = 0.02
    disable_custom_kernels: bool = True
    class_cost: int | float = 2
    bbox_cost: int | float = 5
    giou_cost: int | float = 2
    class_loss_coefficient: int | float = 1
    dice_loss_coefficient: int | float = 1
    bbox_loss_coefficient: int | float = 5
    giou_loss_coefficient: int | float = 2
    eos_coefficient: float = 0.1
    focal_alpha: float = 0.25
    auxiliary_loss: bool = True
    d_model: int = 256

    def __post_init__(self, **kwargs):
        if "mask_loss_coefficient" in kwargs:
            logger.warning_once(
                "The parameter `mask_loss_coefficient` was renamed to `class_loss_coefficient` in LW-DETR. "
                "Please use `class_loss_coefficient` instead. `mask_loss_coefficient` will be removed in a future version."
            )
            self.class_loss_coefficient = kwargs.pop("mask_loss_coefficient")

        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="lw_detr_vit",
            default_config_kwargs={
                "image_size": 1024,
                "hidden_size": 192,
                "num_hidden_layers": 10,
                "window_block_indices": [0, 1, 3, 6, 7, 9],
                "out_indices": [2, 4, 5, 9],
            },
            **kwargs,
        )

        self.projector_in_channels = [self.d_model] * len(self.projector_scale_factors)
        self.projector_out_channels = self.d_model
        self.num_feature_levels = len(self.projector_scale_factors)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["LwDetrConfig", "LwDetrViTConfig"]
