from typing import Any

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/t5_gemma_module-7b")
@strict
class T5GemmaModuleConfig(PreTrainedConfig):

    model_type = "t5_gemma_module"
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

    vocab_size: int = 256000
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = 2
    tie_word_embeddings: bool = True
    rope_parameters: RopeParameters | dict | None = None
    attention_bias: bool = False
    attention_dropout: int | float | None = 0.0
    query_pre_attn_scalar: int = 256
    sliding_window: int | None = 4096
    layer_types: list[str] | None = None
    final_logit_softcapping: float | None = 30.0
    attn_logit_softcapping: float | None = 50.0

    is_decoder: bool = False

    def __post_init__(self, **kwargs):
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if bool((i + 1) % 2) else "full_attention" for i in range(self.num_hidden_layers)
            ]

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


@auto_docstring(checkpoint="google/t5_gemma_module-7b")
@strict
class T5GemmaConfig(PreTrainedConfig):

    model_type = "t5gemma"
    keys_to_ignore_at_inference = ["past_key_values"]
    sub_configs = {"encoder": T5GemmaModuleConfig, "decoder": T5GemmaModuleConfig}

    encoder: T5GemmaModuleConfig | dict[Any, Any] | None = None
    decoder: T5GemmaModuleConfig | dict[Any, Any] | None = None
    is_encoder_decoder: bool = True
    dropout_rate: int | float = 0.0
    classifier_dropout_rate: int | float = 0.0
    attention_dropout: float | int = 0.0
    tie_word_embeddings: bool = True
    vocab_size: int = 256000

    def __post_init__(self, **kwargs):
        if isinstance(self.encoder, dict):
            self.encoder = T5GemmaModuleConfig(**self.encoder)
        elif self.encoder is None:
            self.encoder = T5GemmaModuleConfig()

        if isinstance(self.decoder, dict):
            self.decoder = T5GemmaModuleConfig(**self.decoder)
        elif self.decoder is None:
            self.decoder = T5GemmaModuleConfig()

        self.encoder.is_decoder = False
        self.encoder.dropout_rate = self.dropout_rate
        self.encoder.attention_dropout = self.attention_dropout

        self.decoder.is_decoder = True
        self.decoder.use_cache = True
        self.decoder.dropout_rate = self.dropout_rate
        self.decoder.attention_dropout = self.attention_dropout
        self.decoder.cross_attention_hidden_size = self.encoder.hidden_size

        self.initializer_range = kwargs.pop("initializer_range", self.decoder.initializer_range)

        for special_token_key in ["bos_token_id", "pad_token_id", "eos_token_id"]:
            if special_token_key not in kwargs:
                kwargs[special_token_key] = getattr(self.decoder, special_token_key)

        super().__post_init__(**kwargs)


__all__ = ["T5GemmaConfig", "T5GemmaModuleConfig"]
