from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="baidu/ERNIE-4.5-VL-28B-A3B-PT")
@strict
class Ernie4_5_VLMoeVisionConfig(PreTrainedConfig):

    model_type = "ernie4_5_vl_moe_vision"
    base_config_key = "vision_config"

    depth: int = 32

    hidden_size: int = 1280
    hidden_act: str = "quick_gelu"
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int | list[int] | tuple[int, int] = 14
    spatial_merge_size: int = 2
    initializer_range: float = 0.02

    base_model_tp_plan = {
        "blocks.*.attn.qkv": "colwise",
        "blocks.*.attn.proj": "rowwise",
        "blocks.*.mlp.fc1": "colwise",
        "blocks.*.mlp.fc2": "rowwise",
    }
    intermediate_size: int = 4 * 1280
    temporal_merge_size: int = 2
    rms_norm_eps: float = 1e-6


@auto_docstring(checkpoint="baidu/ERNIE-4.5-VL-28B-A3B-PT")
@strict
class Ernie4_5_VLMoeTextConfig(PreTrainedConfig):

    model_type = "ernie4_5_vl_moe_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_experts": "moe_num_experts", "num_experts_per_tok": "moe_k"}
    default_theta = 500000.0

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.shared_experts.gate_proj": "colwise",
        "layers.*.mlp.shared_experts.up_proj": "colwise",
        "layers.*.mlp.shared_experts.down_proj": "rowwise",
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

    vocab_size: int = 103424
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    hidden_size: int = 2560
    intermediate_size: int = 12288
    num_hidden_layers: int = 28
    num_attention_heads: int = 20
    num_key_value_heads: int | None = 4
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    use_bias: bool | None = False
    moe_intermediate_size: list[int] | None = None
    moe_k: int | None = 6
    moe_num_experts: int | None = 64
    moe_num_shared_experts: int | None = 2
    moe_norm_min: float | None = 1e-12
    output_router_logits: bool | None = False
    router_aux_loss_coef: float | None = 0.001
    base_config_key = "text_config"
    ignore_keys_at_rope_validation = {"mrope_section"}

    mlp_layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.mlp_layer_types is None:
            self.mlp_layer_types = ["dense"] + ["sparse"] * (self.num_hidden_layers - 1)

        if self.moe_intermediate_size is None:
            self.moe_intermediate_size = [1536, 512]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="baidu/ERNIE-4.5-VL-28B-A3B-PT")
@strict
class Ernie4_5_VLMoeConfig(PreTrainedConfig):

    model_type = "ernie4_5_vl_moe"
    sub_configs = {"vision_config": Ernie4_5_VLMoeVisionConfig, "text_config": Ernie4_5_VLMoeTextConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    text_config: dict | PreTrainedConfig | None = None
    vision_config: dict | PreTrainedConfig | None = None
    image_start_token_id: int = 101304
    image_end_token_id: int = 101305
    image_token_id: int = 100295
    video_start_token_id: int = 101306
    video_end_token_id: int = 101307
    video_token_id: int = 103367
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(self.text_config, dict):
            self.text_config = self.sub_configs["text_config"](**self.text_config)
        elif self.text_config is None:
            self.text_config = self.sub_configs["text_config"](**kwargs)

        super().__post_init__(**kwargs)


class Ernie4_5_VL_MoeConfig(Ernie4_5_VLMoeConfig):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeConfig` is deprecated; please use `Ernie4_5_VLMoeConfig` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeTextConfig(Ernie4_5_VLMoeTextConfig):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeTextConfig` is deprecated; please use `Ernie4_5_VLMoeTextConfig` instead.",
        )
        super().__init__(*args, **kwargs)


class Ernie4_5_VL_MoeVisionConfig(Ernie4_5_VLMoeVisionConfig):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "`Ernie4_5_VL_MoeVisionConfig` is deprecated; please use `Ernie4_5_VLMoeVisionConfig` instead.",
        )
        super().__init__(*args, **kwargs)


__all__ = [
    "Ernie4_5_VL_MoeConfig",
    "Ernie4_5_VL_MoeTextConfig",
    "Ernie4_5_VL_MoeVisionConfig",
    "Ernie4_5_VLMoeConfig",
    "Ernie4_5_VLMoeTextConfig",
    "Ernie4_5_VLMoeVisionConfig",
]
