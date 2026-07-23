
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="")
@strict
class DecisionTransformerConfig(PreTrainedConfig):

    model_type = "decision_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    state_dim: int = 17
    act_dim: int = 4
    hidden_size: int = 128
    max_ep_len: int = 4096
    action_tanh: bool = True
    vocab_size: int = 1
    n_positions: int = 1024
    n_layer: int = 3
    n_head: int = 1
    n_inner: int | None = None
    activation_function: str = "relu"
    resid_pdrop: float | int = 0.1
    embd_pdrop: float | int = 0.1
    attn_pdrop: float | int = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False
    add_cross_attention: bool = False


__all__ = ["DecisionTransformerConfig"]
