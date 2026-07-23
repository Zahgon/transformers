from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import CONFIG_MAPPING, AutoConfig


@auto_docstring(checkpoint="lerobot/pi0_base")
@strict
class PI0Config(PreTrainedConfig):

    model_type = "pi0"
    sub_configs = {"vlm_config": AutoConfig, "dit_config": AutoConfig}

    vlm_config: dict | PreTrainedConfig | None = None
    dit_config: dict | PreTrainedConfig | None = None
    chunk_size: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32
    num_inference_steps: int = 10
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0
    loss_reduction: str = "mean"

    def __post_init__(self, **kwargs):
        if isinstance(self.vlm_config, dict):
            vlm_model_type = self.vlm_config.get("model_type", "paligemma")
            self.vlm_config = CONFIG_MAPPING[vlm_model_type](**self.vlm_config)
        elif self.vlm_config is None:
            self.vlm_config = CONFIG_MAPPING["paligemma"](
                text_config={
                    "model_type": "gemma",
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "intermediate_size": 16384,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 1,
                    "vocab_size": 257152,
                },
                vision_config={
                    "model_type": "siglip_vision_model",
                    "intermediate_size": 4304,
                    "hidden_size": 1152,
                    "patch_size": 14,
                    "image_size": 224,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "vocab_size": 257152,
                    "vision_use_head": False,
                },
                projection_dim=2048,
                image_token_id=257152,
            )

        if isinstance(self.dit_config, dict):
            dit_model_type = self.dit_config.get("model_type", "gemma")
            self.dit_config = CONFIG_MAPPING[dit_model_type](**self.dit_config)
        elif self.dit_config is None:
            self.dit_config = CONFIG_MAPPING["gemma"](
                hidden_size=1024,
                num_hidden_layers=18,
                intermediate_size=4096,
                num_attention_heads=8,
                num_key_value_heads=1,
                head_dim=256,
                vocab_size=self.vlm_config.text_config.vocab_size,
            )

        self.dit_config.is_causal = True
        self.dit_config.use_bidirectional_attention = True
        self.vlm_config.text_config.use_bidirectional_attention = True
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["PI0Config"]
