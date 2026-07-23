from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="blab-jhu/test-32m-dec")
@strict
class ModernBertDecoderConfig(PreTrainedConfig):

    model_type = "modernbert-decoder"
    keys_to_ignore_at_inference = ["past_key_values"]
    default_theta = {"global": 160_000.0, "local": 10_000.0}

    vocab_size: int = 50368
    hidden_size: int = 768
    intermediate_size: int = 1152
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    hidden_activation: str = "gelu"
    max_position_embeddings: int = 8192
    initializer_range: float = 0.02
    initializer_cutoff_factor: float = 2.0
    norm_eps: float = 1e-5
    norm_bias: bool = False
    pad_token_id: int = 50283
    eos_token_id: int | list[int] | None = 50282
    bos_token_id: int = 50281
    cls_token_id: int = 50281
    sep_token_id: int = 50282
    attention_bias: bool = False
    attention_dropout: float | int = 0.0
    embedding_dropout: float | int = 0.0
    mlp_bias: bool = False
    mlp_dropout: float | int = 0.0
    decoder_bias: bool = True
    classifier_dropout: float | int = 0.0
    classifier_bias: bool = False
    classifier_activation: str = "gelu"
    use_cache: bool = True
    local_attention: int | None = 128
    layer_types: list[str] | None = None
    tie_word_embeddings: bool = True
    rope_parameters: dict[Literal["full_attention", "sliding_attention"], dict] | None = None

    def __post_init__(self, **kwargs):
        global_attn_every_n_layers = kwargs.get("global_attn_every_n_layers", 3)
        if self.layer_types is None:
            self.layer_types = []
            for layer_id in range(self.num_hidden_layers):
                if layer_id % global_attn_every_n_layers != 0:
                    self.layer_types.append("sliding_attention")
                else:
                    self.layer_types.append("full_attention")

        self.sliding_window = self.local_attention // 2 if self.local_attention else -1
        super().__post_init__(**kwargs)

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)

        default_rope_params = {
            "sliding_attention": {"rope_type": "default"},
            "full_attention": {"rope_type": "default"},
        }
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else default_rope_params
        if rope_scaling is not None:
            self.rope_parameters["full_attention"].update(rope_scaling)
            self.rope_parameters["sliding_attention"].update(rope_scaling)

        if self.rope_parameters.get("full_attention") is None:
            self.rope_parameters["full_attention"] = {"rope_type": "default"}
        self.rope_parameters["full_attention"].setdefault(
            "rope_theta", kwargs.pop("global_rope_theta", self.default_theta["global"])
        )
        if self.rope_parameters.get("sliding_attention") is None:
            self.rope_parameters["sliding_attention"] = {"rope_type": "default"}
        self.rope_parameters["sliding_attention"].setdefault(
            "rope_theta", kwargs.pop("local_rope_theta", self.default_theta["local"])
        )

        self.standardize_rope_params()
        return kwargs


__all__ = ["ModernBertDecoderConfig"]
