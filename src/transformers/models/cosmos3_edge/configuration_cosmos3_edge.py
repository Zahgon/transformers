from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeTextConfig(PreTrainedConfig):

    model_type = "cosmos3_edge_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.fc1": "colwise",
        "layers.*.mlp.fc2": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    vocab_size: int = 131072
    hidden_size: int = 2048
    intermediate_size: int = 9216
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int | None = 8
    hidden_act: str = "relu2"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 11
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    mlp_bias: bool = False
    head_dim: int = 128
    base_config_key = "text_config"
    default_theta = 100_000_000.0
    ignore_keys_at_rope_validation = {"mrope_section"}

    def __post_init__(self, **kwargs):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": self.default_theta,
                "mrope_section": [24, 20, 20],
            }
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeVisionConfig(PreTrainedConfig):

    model_type = "cosmos3_edge_vision"
    base_config_key = "vision_config"
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_hidden_layers: int = 27
    num_attention_heads: int = 16
    num_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 16
    hidden_act: str = "gelu_pytorch_tanh"
    layer_norm_eps: float = 1e-6
    attention_dropout: float | int = 0.0
    num_patches: int = 256
    spatial_merge_size: int = 2


@auto_docstring(checkpoint="nvidia/Cosmos3-Edge-Reasoner")
@strict
class Cosmos3EdgeConfig(PreTrainedConfig):

    model_type = "cosmos3_edge"
    sub_configs = {
        "text_config": Cosmos3EdgeTextConfig,
        "vision_config": Cosmos3EdgeVisionConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: Cosmos3EdgeTextConfig | dict | None = None
    vision_config: Cosmos3EdgeVisionConfig | dict | None = None
    projector_hidden_size: int = 11520
    image_token_id: int = 19
    video_token_id: int = 18
    vision_start_token_id: int = 20
    vision_end_token_id: int = 21
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = Cosmos3EdgeTextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = Cosmos3EdgeTextConfig(**self.text_config)

        if self.vision_config is None:
            self.vision_config = Cosmos3EdgeVisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = Cosmos3EdgeVisionConfig(**self.vision_config)

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["Cosmos3EdgeConfig", "Cosmos3EdgeTextConfig", "Cosmos3EdgeVisionConfig"]
