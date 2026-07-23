from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..siglip import SiglipVisionConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="google/t5gemma-2-270m-270m")
@strict
class T5Gemma2TextConfig(PreTrainedConfig):

    model_type = "t5gemma2_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
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

    vocab_size: int = 262_208
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    query_pre_attn_scalar: int = 256
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __post_init__(self, **kwargs):
        _sliding_window_pattern = kwargs.pop("sliding_window_pattern", 6)
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % _sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)

        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        self.standardize_rope_params()
        return kwargs


@auto_docstring(checkpoint="google/t5gemma-2-270m-270m")
@strict
class T5Gemma2EncoderConfig(PreTrainedConfig):

    model_type = "t5gemma2_encoder"
    attribute_map = {
        "image_token_id": "image_token_index",
        "boi_token_id": "boi_token_index",
        "eoi_token_id": "eoi_token_index",
    }

    sub_configs = {
        "text_config": T5Gemma2TextConfig,
        "vision_config": SiglipVisionConfig,
    }

    text_config: T5Gemma2TextConfig | dict[str, Any] | None = None
    vision_config: SiglipVisionConfig | dict[str, Any] | None = None
    mm_tokens_per_image: int | None = 256
    boi_token_index: int | None = 255_999
    eoi_token_index: int | None = 256_000
    image_token_index: int | None = 262_144
    initializer_range: float | None = 0.02
    tie_word_embeddings: bool | None = True

    def __post_init__(self, **kwargs):
        if self.text_config is None:
            self.text_config = T5Gemma2TextConfig()
            logger.info("text_config is None, using default T5Gemma2EncoderTextConfig text config.")
        elif isinstance(self.text_config, dict):
            self.text_config = T5Gemma2TextConfig(**self.text_config)

        if isinstance(self.vision_config, dict):
            self.vision_config = SiglipVisionConfig(**self.vision_config)
        elif self.vision_config is None:
            self.vision_config = SiglipVisionConfig()
            logger.info("vision_config is None, using default SiglipVisionConfig vision config.")

        super().__post_init__(**kwargs)


@auto_docstring(checkpoint="google/t5gemma-2-270m-270m")
@strict
class T5Gemma2DecoderConfig(PreTrainedConfig):

    model_type = "t5gemma2_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
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

    vocab_size: int = 262_208
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    query_pre_attn_scalar: int = 256
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = None
    attn_logit_softcapping: float | None = None
    default_theta = {"global": 1_000_000.0, "local": 10_000.0}

    def __post_init__(self, **kwargs):
        _sliding_window_pattern = kwargs.pop("sliding_window_pattern", 6)
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % _sliding_window_pattern) else "full_attention"
                for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)

        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_theta", self.default_theta["global"])
        )
        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("rope_local_base_freq", self.default_theta["local"])
        )

        self.standardize_rope_params()
        return kwargs


@auto_docstring(checkpoint="google/t5gemma-2-270m-270m")
@strict
class T5Gemma2Config(PreTrainedConfig):

    model_type = "t5gemma2"
    keys_to_ignore_at_inference = ["past_key_values"]

    sub_configs = {
        "encoder": T5Gemma2EncoderConfig,
        "decoder": T5Gemma2DecoderConfig,
    }

    attribute_map = {
        "image_token_id": "image_token_index",
        "eoi_token_id": "eoi_token_index",
    }

    encoder: T5Gemma2EncoderConfig | dict[str, Any] | None = None
    decoder: T5Gemma2DecoderConfig | dict[str, Any] | None = None
    is_encoder_decoder: bool = True
    dropout_rate: float | int = 0.0
    attention_dropout: float | int = 0.0
    classifier_dropout_rate: float | int = 0.0
    initializer_range: float = 0.02
    image_token_index: int = 256_001
    eoi_token_index: int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder, dict):
            self.encoder = T5Gemma2EncoderConfig(**self.encoder)
        elif self.encoder is None:
            self.encoder = T5Gemma2EncoderConfig()
            logger.info("encoder is None, using default T5Gemma2EncoderConfig encoder config.")

        if isinstance(self.decoder, dict):
            self.decoder = T5Gemma2DecoderConfig(**self.decoder)
        elif self.decoder is None:
            self.decoder = T5Gemma2DecoderConfig()
            logger.info("decoder is None, using default T5Gemma2DecoderConfig decoder config.")

        self.encoder.text_config.dropout_rate = self.dropout_rate
        self.encoder.text_config.attention_dropout = self.attention_dropout
        self.encoder.vision_config.attention_dropout = self.attention_dropout
        self.encoder.image_token_index = self.image_token_index

        self.decoder.dropout_rate = self.dropout_rate
        self.decoder.attention_dropout = self.attention_dropout
        self.eoi_token_index = self.encoder.eoi_token_index

        for special_token_key in ["bos_token_id", "pad_token_id", "eos_token_id", "vocab_size"]:
            if special_token_key not in kwargs:
                kwargs[special_token_key] = getattr(self.decoder, special_token_key)

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["T5Gemma2Config", "T5Gemma2TextConfig", "T5Gemma2EncoderConfig", "T5Gemma2DecoderConfig"]
