
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="facebook/sam-vit-huge")
@strict
class PPChart2TableVisionConfig(PreTrainedConfig):

    base_config_key = "vision_config"
    hidden_size: int = 768
    output_channels: int = 256
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 1024
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-06
    attention_dropout: float | int = 0.0
    initializer_range: float = 1e-10
    qkv_bias: bool = True
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    window_size: int = 14
    global_attn_indexes: list[int] | tuple[int, ...] = (2, 5, 8, 11)
    mlp_dim: int = 3072


@auto_docstring(checkpoint="PaddlePaddle/PP-Chart2Table_safetensors")
@strict
class PPChart2TableConfig(PreTrainedConfig):

    model_type = "pp_chart2table"
    attribute_map = {
        "image_token_id": "image_token_index",
    }
    sub_configs = {"text_config": AutoConfig, "vision_config": PPChart2TableVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_index: int = 151859
    image_seq_length: int = 576
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = PPChart2TableVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = PPChart2TableVisionConfig(**self.vision_config)

        if isinstance(self.text_config, dict):
            self.text_config["model_type"] = self.text_config.get("model_type", "qwen2")
            self.text_config = CONFIG_MAPPING[self.text_config["model_type"]](**self.text_config)
        elif self.text_config is None:
            self.text_config = CONFIG_MAPPING["qwen2"](
                vocab_size=151860,
                hidden_size=1024,
                intermediate_size=2816,
                num_hidden_layers=24,
                num_attention_heads=16,
                num_key_value_heads=16,
                hidden_act="silu",
                max_position_embeddings=32768,
                initializer_range=0.02,
                rms_norm_eps=1e-6,
                use_cache=True,
                tie_word_embeddings=self.tie_word_embeddings,
                rope_theta=1000000.0,
                rope_parameters=None,
                use_sliding_window=False,
                sliding_window=4096,
                max_window_layers=21,
                attention_dropout=0.0,
            )

        super().__post_init__(**kwargs)


__all__ = ["PPChart2TableConfig"]
