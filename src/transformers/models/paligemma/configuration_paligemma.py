
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="google/paligemma-3b-pt-224")
@strict
class PaliGemmaConfig(PreTrainedConfig):

    model_type = "paligemma"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 256000
    vocab_size: int = 257152
    projection_dim: int = 2048
    hidden_size: int = 2048
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "siglip_vision_model")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["siglip_vision_model"](
                intermediate_size=4096,
                hidden_size=1152,
                patch_size=14,
                image_size=224,
                num_hidden_layers=27,
                num_attention_heads=16,
                vocab_size=257152,
                vision_use_head=False,
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "gemma")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["gemma"](
                hidden_size=2048,
                num_hidden_layers=18,
                intermediate_size=16384,
                num_attention_heads=8,
                num_key_value_heads=1,
                is_encoder_decoder=False,
                vocab_size=self.vocab_size,
            )

        if self.text_config.use_bidirectional_attention is None:
            self.text_config.use_bidirectional_attention = True

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = self.projection_dim
        super().__post_init__(**kwargs)


__all__ = ["PaliGemmaConfig"]
