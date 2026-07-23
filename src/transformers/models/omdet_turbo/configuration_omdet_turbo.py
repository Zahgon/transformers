
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="omlab/omdet-turbo-swin-tiny-hf")
@strict
class OmDetTurboConfig(PreTrainedConfig):

    model_type = "omdet-turbo"
    sub_configs = {"backbone_config": AutoConfig, "text_config": AutoConfig}
    attribute_map = {
        "encoder_hidden_dim": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }

    text_config: dict | PreTrainedConfig | None = None
    backbone_config: dict | PreTrainedConfig | None = None
    apply_layernorm_after_vision_backbone: bool = True
    image_size: int | list[int] | tuple[int, int] = 640
    disable_custom_kernels: bool = False
    layer_norm_eps: float = 1e-5
    batch_norm_eps: float = 1e-5
    init_std: float = 0.02
    text_projection_in_dim: int = 512
    text_projection_out_dim: int = 512
    task_encoder_hidden_dim: int = 1024
    class_embed_dim: int = 512
    class_distance_type: Literal["cosine", "dot"] = "cosine"
    num_queries: int = 900
    csp_activation: str = "silu"
    conv_norm_activation: str = "gelu"
    encoder_feedforward_activation: str = "relu"
    encoder_feedforward_dropout: float | int = 0.0
    encoder_dropout: float | int = 0.0
    hidden_expansion: int = 1
    encoder_hidden_dim: int = 256
    vision_features_channels: list[int] | tuple[int, ...] = (256, 256, 256)
    encoder_in_channels: list[int] | tuple[int, ...] = (192, 384, 768)
    encoder_projection_indices: list[int] | tuple[int, ...] = (2,)
    encoder_attention_heads: int = 8
    encoder_dim_feedforward: int = 2048
    encoder_layers: int = 1
    positional_encoding_temperature: int = 10000
    num_feature_levels: int = 3
    decoder_hidden_dim: int = 256
    decoder_num_heads: int = 8
    decoder_num_layers: int = 6
    decoder_activation: str = "relu"
    decoder_dim_feedforward: int = 2048
    decoder_num_points: int = 4
    decoder_dropout: float | int = 0.0
    eval_size: int | None = None
    learn_initial_query: bool = False
    cache_size: int = 100
    is_encoder_decoder: bool = True

    def __post_init__(self, **kwargs):
        timm_default_kwargs = {
            "out_indices": [1, 2, 3],
            "img_size": self.image_size,
            "always_partition": True,
        }
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_backbone="swin_tiny_patch4_window7_224",
            default_config_type="swin",
            default_config_kwargs={"image_size": self.image_size, "out_indices": [2, 3, 4]},
            timm_default_kwargs=timm_default_kwargs,
            **kwargs,
        )

        self.timm_kwargs = {}
        if getattr(self.backbone_config, "model_type", None) == "timm_backbone":
            for attr in ("img_size", "always_partition"):
                if hasattr(self.backbone_config, attr):
                    self.timm_kwargs[attr] = getattr(self.backbone_config, attr)

        if self.text_config is None:
            logger.info("`text_config` is `None`. Initializing the config with the default `clip_text_model`")
            self.text_config = CONFIG_MAPPING["clip_text_model"]()
        elif isinstance(self.text_config, dict):
            text_model_type = self.text_config.get("model_type")
            self.text_config = CONFIG_MAPPING[text_model_type](**self.text_config)

        super().__post_init__(**kwargs)

    def to_dict(self):
        output = super().to_dict()
        output.pop("timm_kwargs", None)
        return output


__all__ = ["OmDetTurboConfig"]
