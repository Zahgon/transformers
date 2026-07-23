from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="PekingU/rtdetr_r18vd")
@strict
class RTDetrV2Config(PreTrainedConfig):

    model_type = "rt_detr_v2"
    sub_configs = {"backbone_config": AutoConfig}
    layer_types = ["basic", "bottleneck"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    initializer_range: float = 0.01
    initializer_bias_prior_prob: float | None = None
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    backbone_config: dict | PreTrainedConfig | None = None
    freeze_backbone_batch_norms: bool = True
    encoder_hidden_dim: int = 256
    encoder_in_channels: list[int] | tuple[int, ...] = (512, 1024, 2048)
    feat_strides: list[int] | tuple[int, ...] = (8, 16, 32)
    encoder_layers: int = 1
    encoder_ffn_dim: int = 1024
    encoder_attention_heads: int = 8
    dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    encode_proj_layers: list[int] | tuple[int, ...] = (2,)
    positional_encoding_temperature: int = 10000
    encoder_activation_function: str = "gelu"
    activation_function: str = "silu"
    eval_size: int | None = None
    normalize_before: bool = False
    hidden_expansion: float = 1.0
    d_model: int = 256
    num_queries: int = 300
    decoder_in_channels: list[int] | tuple[int, ...] = (256, 256, 256)
    decoder_ffn_dim: int = 1024
    num_feature_levels: int = 3
    decoder_n_points: int = 4
    decoder_layers: int = 6
    decoder_attention_heads: int = 8
    decoder_activation_function: str = "relu"
    attention_dropout: float | int = 0.0
    num_denoising: int = 100
    label_noise_ratio: float = 0.5
    box_noise_scale: float = 1.0
    learn_initial_query: bool = False
    anchor_image_size: int | list[int] | None = None
    with_box_refine: bool = True
    is_encoder_decoder: bool = True
    matcher_alpha: float = 0.25
    matcher_gamma: float = 2.0
    matcher_class_cost: float = 2.0
    matcher_bbox_cost: float = 5.0
    matcher_giou_cost: float = 2.0
    use_focal_loss: bool = True
    auxiliary_loss: bool = True
    focal_loss_alpha: float = 0.75
    focal_loss_gamma: float = 2.0
    weight_loss_vfl: float = 1.0
    weight_loss_bbox: float = 5.0
    weight_loss_giou: float = 2.0
    eos_coefficient: float = 1e-4
    decoder_n_levels: int = 3
    decoder_offset_scale: float = 0.5
    decoder_method: str = "default"
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="rt_detr_resnet",
            default_config_kwargs={"out_indices": [2, 3, 4]},
            **kwargs,
        )
        super().__post_init__(**kwargs)


__all__ = ["RTDetrV2Config"]
