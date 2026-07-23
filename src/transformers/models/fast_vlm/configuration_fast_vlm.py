

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="KamilaMila/FastVLM-7B")
@strict
class FastVlmConfig(PreTrainedConfig):

    model_type = "fast_vlm"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 151646
    image_seq_length: int = 576
    projector_hidden_act: str = "gelu"
    vision_feature_select_strategy: str = "full"
    vision_feature_layer: int | list[int] = -1
    multimodal_projector_bias: bool = True
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config["model_type"] = self.vision_config.get("model_type", "timm_wrapper")
            self.vision_config = CONFIG_MAPPING[self.vision_config["model_type"]](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = CONFIG_MAPPING["timm_wrapper"](
                architecture="fastvit_mci3",
                do_pooling=True,
                global_pool="avg",
                hidden_size=3072,
                initializer_range=0.02,
                model_args={"inference_mode": True},
            )

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"](
                hidden_size=3584,
                vocab_size=152128,
                intermediate_size=18944,
                num_attention_heads=28,
                num_key_value_heads=4,
                num_hidden_layers=28,
            )
        if not self.tie_word_embeddings and self.text_config.tie_word_embeddings:
            self.tie_word_embeddings = self.text_config.tie_word_embeddings

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["FastVlmConfig"]
