
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="lightonai/LightOnOCR-1B-1025")
@strict
class LightOnOcrConfig(PreTrainedConfig):

    model_type = "lighton_ocr"
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    spatial_merge_size: int = 2
    image_token_id: int = 151655
    tie_word_embeddings: bool = True
    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["pixtral"](
                attention_dropout=0.0,
                head_dim=64,
                hidden_act="silu",
                hidden_size=1024,
                image_size=1540,
                initializer_range=0.02,
                intermediate_size=4096,
                model_type="pixtral",
                num_attention_heads=16,
                num_channels=3,
                num_hidden_layers=24,
                patch_size=14,
                rope_theta=10000,
            )
        elif isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "pixtral")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)

        if self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen3"](
                attention_dropout=0.0,
                head_dim=128,
                hidden_act="silu",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=3072,
                max_position_embeddings=40960,
                num_attention_heads=16,
                num_hidden_layers=28,
                num_key_value_heads=8,
                rms_norm_eps=1e-6,
                rope_theta=1000000,
                sliding_window=None,
                use_cache=True,
                vocab_size=151936,
            )
        elif isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen3")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)

        super().__post_init__(**kwargs)


__all__ = ["LightOnOcrConfig"]
