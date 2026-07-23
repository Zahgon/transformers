from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin, consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="Roboflow/rf-detr-base")
@strict
class RfDetrDinov2Config(BackboneConfigMixin, PreTrainedConfig):

    model_type = "rf_detr_dinov2"

    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    mlp_ratio: int = 4
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-6
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 14
    num_channels: int = 3
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    use_swiglu_ffn: bool = False
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    apply_layernorm: bool = True
    reshape_hidden_states: bool = True
    use_mask_token: bool = True

    num_windows: int = 4

    def __post_init__(self, **kwargs):
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        window_block_indexes = set(range(self._out_indices[-1] + 1))
        window_block_indexes.difference_update(self._out_indices)
        self.window_block_indexes = list(window_block_indexes)
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="Roboflow/rf-detr-base")
@strict
class RfDetrConfig(PreTrainedConfig):

    model_type = "rf_detr"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    hidden_expansion: float = 0.5
    c2f_num_blocks: int = 3
    activation_function: str = "silu"
    dropout: float = 0.1
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

    layer_norm_eps: float = 1e-5
    num_feature_levels: int = 1
    mask_loss_coefficient: int | float = 1
    mask_point_sample_ratio: int = 16
    mask_downsample_ratio: int = 4
    mask_class_loss_coefficient: int | float = 5.0
    mask_dice_loss_coefficient: int | float = 5.0
    segmentation_head_activation_function: str = "gelu"
    intermediate_size: int = 1024

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="rf_detr_dinov2",
            default_config_kwargs={
                "num_attention_heads": 6,
                "out_features": ["stage2", "stage5", "stage8", "stage11"],
                "hidden_size": 384,
                "num_register_tokens": 0,
                "image_size": 518,
            },
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["RfDetrConfig", "RfDetrDinov2Config"]
