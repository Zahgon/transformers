
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="zai-org/GLM-Image")
@strict
class GlmImageVQVAEConfig(PreTrainedConfig):
    model_type = "glm_image_vqmodel"
    base_config_key = "vq_config"

    embed_dim: int = 2048
    num_embeddings: int = 16384
    latent_channels: int = 1536
    in_channels: int = 3
    initializer_range: float = 0.02


@auto_docstring(checkpoint="zai-org/GLM-Image")
@strict
class GlmImageVisionConfig(PreTrainedConfig):

    model_type = "glm_image_vision"
    base_config_key = "vision_config"

    depth: int = 40
    hidden_size: int = 1536
    hidden_act: str = "gelu"
    attention_bias: bool = True
    attention_dropout: float | int = 0.0
    num_heads: int = 16
    in_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 2048
    patch_size: int | list[int] | tuple[int, int] = 16
    spatial_merge_size: int = 1
    intermediate_size: int = 6144
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-06


@auto_docstring(checkpoint="zai-org/GLM-Image")
@strict
class GlmImageTextConfig(PreTrainedConfig):

    model_type = "glm_image_text"
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

    vocab_size: int = 168064
    hidden_size: int = 4096
    intermediate_size: int = 13696
    num_hidden_layers: int = 40
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 2
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-05
    use_cache: bool = True
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None
    pad_token_id: int = 167841
    vision_vocab_size: int = 16512
    attention_bias: bool = True
    eos_token_id: int | list[int] | None = 16385

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="zai-org/GLM-Image")
@strict
class GlmImageConfig(PreTrainedConfig):

    model_type = "glm_image"
    sub_configs = {
        "vision_config": GlmImageVisionConfig,
        "text_config": GlmImageTextConfig,
        "vq_config": GlmImageVQVAEConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    vq_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 167855
    image_start_token_id: int = 16384
    image_end_token_id: int = 16385
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"](**kwargs)

        if isinstance(self.vq_config, dict):
            self.vq_config = self.sub_configs["vq_config"](**self.vq_config)
        elif self.vq_config is None:
            self.vq_config = self.sub_configs["vq_config"](**kwargs)

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        super().__post_init__(**kwargs)


__all__ = ["GlmImageVQVAEConfig", "GlmImageVisionConfig", "GlmImageTextConfig", "GlmImageConfig"]
