
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="magic-leap-community/superglue_indoor")
@strict
class SuperGlueConfig(PreTrainedConfig):

    model_type = "superglue"
    sub_configs = {"keypoint_detector_config": AutoConfig}

    keypoint_detector_config: dict | PreTrainedConfig | None = None
    hidden_size: int = 256
    keypoint_encoder_sizes: list[int] | None = None
    gnn_layers_types: list[str] | None = None
    num_attention_heads: int = 4
    sinkhorn_iterations: int = 100
    matching_threshold: float = 0.0
    initializer_range: float = 0.02
    is_decoder: bool = False
    attention_probs_dropout_prob: int | float = 0.0

    def __post_init__(self, **kwargs):
        self.gnn_layers_types = self.gnn_layers_types if self.gnn_layers_types is not None else ["self", "cross"] * 9
        self.keypoint_encoder_sizes = (
            self.keypoint_encoder_sizes if self.keypoint_encoder_sizes is not None else [32, 64, 128, 256]
        )

        if isinstance(self.keypoint_detector_config, dict):
            self.keypoint_detector_config["model_type"] = self.keypoint_detector_config.get("model_type", "superpoint")
            self.keypoint_detector_config = CONFIG_MAPPING[self.keypoint_detector_config["model_type"]](
                **self.keypoint_detector_config
            )
        elif self.keypoint_detector_config is None:
            self.keypoint_detector_config = CONFIG_MAPPING["superpoint"]()

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["SuperGlueConfig"]
