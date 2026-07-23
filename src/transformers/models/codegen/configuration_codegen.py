
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="Salesforce/codegen-2B-mono")
@strict
class CodeGenConfig(PreTrainedConfig):

    model_type = "codegen"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 50400
    n_positions: int = 2048
    n_ctx: int = 2048
    n_embd: int = 4096
    n_layer: int = 28
    n_head: int = 16
    rotary_dim: int = 64
    n_inner: int | None = None
    activation_function: str = "gelu_new"
    resid_pdrop: float | int = 0.0
    embd_pdrop: float | int = 0.0
    attn_pdrop: float | int = 0.0
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    bos_token_id: int | None = 50256
    eos_token_id: int | list[int] | None = 50256
    tie_word_embeddings: bool = False


__all__ = ["CodeGenConfig"]
