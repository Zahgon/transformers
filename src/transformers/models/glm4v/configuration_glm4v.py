from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="zai-org/GLM-4.1V-9B-Thinking")
@strict
class Glm4vVisionConfig(PreTrainedConfig):

    model_type = "glm4v_vision"
    base_config_key = "vision_config"

    depth: int = 24
    hidden_size: int = 1536
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    num_heads: int = 12
    in_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 336
    patch_size: int | list[int] | tuple[int, int] = 14
    rms_norm_eps: float = 1e-05
    spatial_merge_size: int = 2
    temporal_patch_size: int | list[int] | tuple[int, int] = 2
    out_hidden_size: int = 4096
    intermediate_size: int = 13696
    initializer_range: float = 0.02


@auto_docstring(checkpoint="zai-org/GLM-4.1V-9B-Thinking")
@strict
class Glm4vTextConfig(PreTrainedConfig):

    model_type = "glm4v_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_up_proj": "colwise_gather_output",  # we need to replicate here due to the `chunk` operation
        "layers.*.mlp.down_proj": "rowwise_split_input",  # input is replicated due to the `chunk` operation
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    ignore_keys_at_rope_validation = {"mrope_section"}

    vocab_size: int = 151552
    hidden_size: int = 4096
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="zai-org/GLM-4.1V-9B-Thinking")
@strict
class Glm4vConfig(PreTrainedConfig):

    model_type = "glm4v"
    sub_configs = {"vision_config": Glm4vVisionConfig, "text_config": Glm4vTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 151343
    video_token_id: int = 151344
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151341
    video_end_token_id: int = 151342
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"](**kwargs)

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        super().__post_init__(**kwargs)


__all__ = ["Glm4vConfig", "Glm4vTextConfig", "Glm4vVisionConfig"]
