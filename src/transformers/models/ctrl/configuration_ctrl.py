
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="Salesforce/ctrl")
@strict
class CTRLConfig(PreTrainedConfig):

    model_type = "ctrl"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    vocab_size: int = 246534
    n_positions: int = 256
    n_embd: int = 1280
    dff: int = 8192
    n_layer: int = 48
    n_head: int = 16
    resid_pdrop: float | int = 0.1
    embd_pdrop: float | int = 0.1
    layer_norm_epsilon: float = 1e-6
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True


__all__ = ["CTRLConfig"]
