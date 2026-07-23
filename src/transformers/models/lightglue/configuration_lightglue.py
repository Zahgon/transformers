
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig
from ..superpoint import SuperPointConfig


@auto_docstring(checkpoint="ETH-CVG/lightglue_superpoint")
@strict
class LightGlueConfig(PreTrainedConfig):

    model_type = "lightglue"
    sub_configs = {"keypoint_detector_config": AutoConfig}

    keypoint_detector_config: dict | SuperPointConfig | None = None
    descriptor_dim: int = 256
    num_hidden_layers: int = 9
    num_attention_heads: int = 4
    num_key_value_heads: int | None = None
    depth_confidence: float = 0.95
    width_confidence: float = 0.99
    filter_threshold: float = 0.1
    initializer_range: float = 0.02
    hidden_act: str = "gelu"
    attention_dropout: float | int = 0.0
    attention_bias: bool = True

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if isinstance(self.keypoint_detector_config, dict):
            self.keypoint_detector_config["model_type"] = self.keypoint_detector_config.get("model_type", "superpoint")
            self.keypoint_detector_config = CONFIG_MAPPING[self.keypoint_detector_config["model_type"]](
                **self.keypoint_detector_config, attn_implementation="eager"
            )
        elif self.keypoint_detector_config is None:
            self.keypoint_detector_config = CONFIG_MAPPING["superpoint"](attn_implementation="eager")

        self.intermediate_size = self.descriptor_dim * 2
        self.hidden_size = self.descriptor_dim
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["LightGlueConfig"]
