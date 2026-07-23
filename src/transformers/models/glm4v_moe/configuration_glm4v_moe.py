from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="zai-org/GLM-4.5V")
@strict
class Glm4vMoeTextConfig(PreTrainedConfig):

    model_type = "glm4v_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    vocab_size: int = 151424
    hidden_size: int = 4096
    intermediate_size: int = 10944
    num_hidden_layers: int = 46
    num_attention_heads: int = 96
    num_key_value_heads: int = 8
    hidden_act: str = "silu"
    max_position_embeddings: int = 65536
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = True
    attention_dropout: float | int = 0.0
    moe_intermediate_size: int = 1408
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    n_routed_experts: int = 128
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    first_k_dense_replace: int = 1
    norm_topk_prob: bool = True
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section"}
    router_aux_loss_coef: float = 0.0001

    def __post_init__(self, **kwargs):
        kwargs.setdefault("partial_rotary_factor", 0.5)  # assign default for BC
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="zai-org/GLM-4.1V-9B-Thinking")
@strict
class Glm4vMoeVisionConfig(PreTrainedConfig):

    model_type = "glm4v_moe_vision"
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


@auto_docstring(checkpoint="zai-org/GLM-4.5V")
@strict
class Glm4vMoeConfig(PreTrainedConfig):

    model_type = "glm4v_moe"
    sub_configs = {"vision_config": Glm4vMoeVisionConfig, "text_config": Glm4vMoeTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None

    image_token_id: int = 151363
    video_token_id: int = 151364
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


__all__ = ["Glm4vMoeConfig", "Glm4vMoeVisionConfig", "Glm4vMoeTextConfig"]
