
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/switch-base-8")
@strict
class SwitchTransformersConfig(PreTrainedConfig):

    model_type = "switch_transformers"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    vocab_size: int = 32128
    d_model: int = 768
    d_kv: int = 64
    d_ff: int = 2048
    expert_capacity: int = 64
    num_layers: int = 12
    num_sparse_encoder_layers: int = 3
    num_decoder_layers: int | None = 12
    num_sparse_decoder_layers: int = 3
    num_heads: int = 12
    num_experts: int = 8
    router_bias: bool = False
    router_jitter_noise: int | float = 0.01
    router_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    router_ignore_padding_tokens: bool = False
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    dropout_rate: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001
    initializer_factor: float = 1.0
    dense_act_fn: str = "relu"
    is_encoder_decoder: bool = True
    add_router_probs: bool = False
    use_cache: bool = True
    pad_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    bos_token_id: int | None = None
    tie_word_embeddings: bool = True
    is_decoder: bool = False
    add_cross_attention: bool = False

    def __post_init__(self, **kwargs):
        self.num_decoder_layers = (
            self.num_decoder_layers if self.num_decoder_layers is not None else self.num_layers
        )  # default = symmetry

        if self.num_sparse_encoder_layers > 0:
            self.encoder_sparse_step = self.num_layers // self.num_sparse_encoder_layers
        else:
            self.encoder_sparse_step = self.num_layers  # HACK: this will create 0 sparse layers

        if self.num_sparse_decoder_layers > 0:
            self.decoder_sparse_step = self.num_decoder_layers // self.num_sparse_decoder_layers
        else:
            self.decoder_sparse_step = self.num_decoder_layers  # HACK: this will create 0 sparse layers

        super().__post_init__(**kwargs)


__all__ = ["SwitchTransformersConfig"]
