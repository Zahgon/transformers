
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="deepseek-community/DeepSeek-OCR-2")
@strict
class DeepseekOcr2SamVisionConfig(PreTrainedConfig):

    base_config_key = "sam_config"
    model_type = "deepseek_ocr2_sam_vision_model"

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
    mlp_ratio: float = 4.0
    use_abs_pos: bool = True
    use_rel_pos: bool = True
    window_size: int = 14
    global_attn_indexes: list[int] | tuple[int, ...] = (2, 5, 8, 11)
    mlp_dim: int | None = None

    downsample_channels: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.downsample_channels is None:
            self.downsample_channels = [512, 896]
        self.mlp_dim = int(self.hidden_size * self.mlp_ratio) if self.mlp_dim is None else self.mlp_dim
        self.scale = self.hidden_size // 2
        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="deepseek-community/DeepSeek-OCR-2")
@strict
class DeepseekOcr2VisionEncoderConfig(PreTrainedConfig):

    model_type = "deepseek_ocr2_encoder"
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

    vocab_size: int = 151936
    hidden_size: int = 4096
    intermediate_size: int = 22016
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = 32
    hidden_act: str = "silu"
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    use_sliding_window: bool = False
    sliding_window: int | None = 4096
    max_window_layers: int = 28
    layer_types: list[str] | None = None
    attention_dropout: float | int = 0.0
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    base_config_key = "encoder_config"

    def __post_init__(self, **kwargs):
        self.sliding_window = self.sliding_window if self.use_sliding_window else None
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="deepseek-community/DeepSeek-OCR-2")
@strict
class DeepseekOcr2VisionConfig(PreTrainedConfig):

    model_type = "deepseek_ocr2_vision"
    base_config_key = "vision_config"
    sub_configs = {
        "sam_config": DeepseekOcr2SamVisionConfig,
        "encoder_config": DeepseekOcr2VisionEncoderConfig,
    }

    sam_config: dict | PreTrainedConfig | None = None
    encoder_config: dict | PreTrainedConfig | None = None

    def __post_init__(self, **kwargs):
        if self.sam_config is None:
            self.sam_config = DeepseekOcr2SamVisionConfig()
        elif isinstance(self.sam_config, dict):
            self.sam_config = DeepseekOcr2SamVisionConfig(**self.sam_config)

        if self.encoder_config is None:
            self.encoder_config = DeepseekOcr2VisionEncoderConfig()
        elif isinstance(self.encoder_config, dict):
            self.encoder_config = DeepseekOcr2VisionEncoderConfig(**self.encoder_config)

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="deepseek-community/DeepSeek-OCR-2")
@strict
class DeepseekOcr2TextConfig(PreTrainedConfig):

    model_type = "deepseek_ocr2_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.experts.gate_up_proj": "packed_colwise",
        "layers.*.mlp.experts.down_proj": "rowwise",
        "layers.*.mlp.experts": "moe_tp_experts",
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

    vocab_size: int = 32000
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int | None = None
    hidden_act: str = "silu"
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    pretraining_tp: int | None = 1
    tie_word_embeddings: bool = False
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: float | None = 0.0
    mlp_bias: bool = False
    head_dim: int | None = None
    base_model_ep_plan = {
        "layers.*.mlp.gate": "ep_router",
        "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
        "layers.*.mlp.experts.down_proj": "grouped_gemm",
        "layers.*.mlp.experts": "moe_tp_experts",
    }
    attribute_map = {
        "num_experts": "n_routed_experts",
    }
    n_group: int | None = None
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    routed_scaling_factor: float = 1.0
    topk_group: int | None = None
    topk_method: str | None = "greedy"
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int = 1407

    base_config_key = "text_config"
    mlp_layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="deepseek-community/DeepSeek-OCR-2")
@strict
class DeepseekOcr2Config(PreTrainedConfig):

    model_type = "deepseek_ocr2"
    sub_configs = {
        "vision_config": DeepseekOcr2VisionConfig,
        "text_config": DeepseekOcr2TextConfig,
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 128815
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = DeepseekOcr2VisionConfig()
        elif isinstance(self.vision_config, dict):
            self.vision_config = DeepseekOcr2VisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = DeepseekOcr2TextConfig()
        elif isinstance(self.text_config, dict):
            self.text_config = DeepseekOcr2TextConfig(**self.text_config)

        super().__post_init__(**kwargs)


__all__ = [
    "DeepseekOcr2Config",
    "DeepseekOcr2TextConfig",
    "DeepseekOcr2VisionConfig",
    "DeepseekOcr2VisionEncoderConfig",
    "DeepseekOcr2SamVisionConfig",
]
