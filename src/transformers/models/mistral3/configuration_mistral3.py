

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="mistralai/Mistral-Small-3.1-24B-Instruct-2503")
@strict
class Mistral3Config(PreTrainedConfig):

    model_type = "mistral3"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    is_composition = True

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 10
    projector_hidden_act: str = "gelu"
    vision_feature_layer: int | list[int] = -1
    multimodal_projector_bias: bool = False
    spatial_merge_size: int = 2
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "pixtral")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["pixtral"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=1540,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                head_dim=64,
                hidden_act="gelu",
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "mistral")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["mistral"](
                attention_dropout=0.0,
                head_dim=128,
                hidden_act="silu",
                hidden_size=5120,
                initializer_range=0.02,
                intermediate_size=32768,
                max_position_embeddings=131072,
                model_type="mistral",
                num_attention_heads=32,
                num_hidden_layers=40,
                num_key_value_heads=8,
                rms_norm_eps=1e-05,
                rope_theta=1000000000.0,
                sliding_window=None,
                use_cache=True,
                vocab_size=131072,
            )

        super().__post_init__(**kwargs)


__all__ = ["Mistral3Config"]
