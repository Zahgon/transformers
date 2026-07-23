

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="meta-llama/Llama-4-Scout-17B-16E")
@strict
class Llama4VisionConfig(PreTrainedConfig):

    base_model_tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "vision_adapter.mlp.fc1": "colwise",
        "vision_adapter.mlp.fc2": "rowwise",
        "patch_embedding.linear": "colwise_gather_output",
    }
    model_type = "llama4_vision_model"
    base_config_key = "vision_config"

    hidden_size: int = 768
    hidden_act: str = "gelu"
    num_hidden_layers: int = 34
    num_attention_heads: int = 16
    num_channels: int = 3
    intermediate_size: int = 5632
    vision_output_dim: int = 7680
    image_size: int | list[int] | tuple[int, int] = 448
    patch_size: int | list[int] | tuple[int, int] = 14
    norm_eps: float = 1e-5
    vision_feature_select_strategy: str = "default"
    initializer_range: float = 0.02
    pixel_shuffle_ratio: float = 0.5
    projector_input_dim: int = 4096
    projector_output_dim: int = 4096
    multi_modal_projector_bias: bool = False
    projector_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    rope_parameters: RopeParameters | dict | None = None


@auto_docstring(checkpoint="meta-llama/Llama-4-Scout-17B-16E")
@strict
class Llama4TextConfig(PreTrainedConfig):

    model_type = "llama4_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = 500000.0
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.shared_expert.gate_proj": "colwise",
        "layers.*.feed_forward.shared_expert.up_proj": "colwise",
        "layers.*.feed_forward.shared_expert.down_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "packed_rowwise",  # row because not linear
        "layers.*.feed_forward.experts.down_proj": "colwise",  # col because not linear
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
    }
    base_model_ep_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.feed_forward.experts.gate_up_proj": "grouped_gemm",  # row because not linear
        "layers.*.feed_forward.experts.down_proj": "grouped_gemm",  # col because not linear
        "layers.*.feed_forward.gate_proj": "colwise",
        "layers.*.feed_forward.up_proj": "colwise",
        "layers.*.feed_forward.down_proj": "rowwise",
        "layers.*.feed_forward.router": "ep_router",
    }

    vocab_size: int = 202048
    hidden_size: int = 5120
    intermediate_size: int = 8192
    intermediate_size_mlp: int = 16384
    num_hidden_layers: int = 48
    num_attention_heads: int = 40
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 4096 * 32
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = False
    attention_dropout: float | int = 0.0
    num_experts_per_tok: int = 1
    num_local_experts: int = 16
    moe_layers: list[int] | None = None
    interleave_moe_layer_step: int = 1
    use_qk_norm: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    rope_parameters: RopeParameters | dict | None = None
    no_rope_layers: list[int] | None = None
    no_rope_layer_interval: int = 4
    attention_chunk_size: int | None = 8192
    layer_types: list[str] | None = None
    attn_temperature_tuning: bool = True
    floor_scale: int = 8192
    attn_scale: float = 0.1
    attention_bias: bool = False

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        default_no_rope_layers = [
            int((layer_idx + 1) % self.no_rope_layer_interval != 0) for layer_idx in range(self.num_hidden_layers)
        ]
        self.no_rope_layers = self.no_rope_layers if self.no_rope_layers else default_no_rope_layers
        self.head_dim = self.head_dim if self.head_dim is not None else self.hidden_size // self.num_attention_heads

        self.moe_layers = (
            self.moe_layers
            if self.moe_layers is not None
            else list(
                range(
                    self.interleave_moe_layer_step - 1,
                    self.num_hidden_layers,
                    self.interleave_moe_layer_step,
                )
            )
        )

        if self.layer_types is None:
            self.layer_types = [
                "chunked_attention" if no_rope else "full_attention" for no_rope in self.no_rope_layers
            ]

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="meta-llama/Llama-4-Scout-17B-16E")
@strict
class Llama4Config(PreTrainedConfig):

    model_type = "llama4"
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }
    sub_configs = {"text_config": Llama4TextConfig, "vision_config": Llama4VisionConfig}
    base_model_tp_plan = {
        "multi_modal_projector.linear_1": "colwise_rep",
    }

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    boi_token_index: int = 200080
    eoi_token_index: int = 200081
    image_token_index: int = 200092
    tie_word_embeddings: bool = False

    def __post_init__(self, **kwargs):
        if self.vision_config is None:
            self.vision_config = Llama4VisionConfig()
            logger.info("vision_config is None, using default llama4 vision config")
        elif isinstance(self.vision_config, dict):
            self.vision_config = Llama4VisionConfig(**self.vision_config)

        if self.text_config is None:
            self.text_config = Llama4TextConfig()
            logger.info("text_config is None, using default llama4 text config")
        elif isinstance(self.text_config, dict):
            self.text_config = Llama4TextConfig(**self.text_config)
        super().__post_init__(**kwargs)


__all__ = ["Llama4Config", "Llama4TextConfig", "Llama4VisionConfig"]
