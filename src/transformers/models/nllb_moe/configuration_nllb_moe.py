
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/nllb-moe-54b")
@strict
class NllbMoeConfig(PreTrainedConfig):

    model_type = "nllb-moe"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_attention_heads": "encoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "encoder_layers",
    }

    vocab_size: int = 128112
    max_position_embeddings: int = 1024
    encoder_layers: int = 12
    encoder_ffn_dim: int = 4096
    encoder_attention_heads: int = 16
    decoder_layers: int = 12
    decoder_ffn_dim: int = 4096
    decoder_attention_heads: int = 16
    encoder_layerdrop: float | int = 0.05
    decoder_layerdrop: float | int = 0.05
    use_cache: bool = True
    is_encoder_decoder: bool = True
    activation_function: str = "relu"
    d_model: int = 1024
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.0
    init_std: float = 0.02
    decoder_start_token_id: int | None = 2
    scale_embedding: bool = True
    router_bias: bool = False
    router_dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    router_ignore_padding_tokens: bool = False
    num_experts: int = 128
    expert_capacity: int = 64
    encoder_sparse_step: int = 4
    decoder_sparse_step: int = 4
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001
    second_expert_policy: str = "all"
    normalize_router_prob_before_dropping: bool = False
    batch_prioritized_routing: bool = False
    moe_eval_capacity_token_fraction: float = 1.0
    moe_token_dropout: float | int = 0.2
    pad_token_id: int | None = 1
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 2
    tie_word_embeddings: bool = True
    output_router_logits: bool = False


__all__ = ["NllbMoeConfig"]
