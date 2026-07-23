
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="flaubert/flaubert_base_uncased")
@strict
class FlaubertConfig(PreTrainedConfig):

    model_type = "flaubert"
    attribute_map = {
        "hidden_size": "emb_dim",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layers",
        "n_words": "vocab_size",  # For backward compatibility
        "bos_index": "bos_token_id",
        "eos_index": "eos_token_id",
        "pad_index": "pad_token_id",
    }

    pre_norm: bool = False
    layerdrop: float | int = 0.0
    vocab_size: int = 30145
    emb_dim: int = 2048
    n_layers: int = 12
    n_heads: int = 16
    dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    gelu_activation: bool = True
    sinusoidal_embeddings: bool = False
    causal: bool = False
    asm: bool = False
    n_langs: int = 1
    use_lang_emb: bool = True
    max_position_embeddings: int = 512
    embed_init_std: float = 2048**-0.5
    layer_norm_eps: float = 1e-12
    init_std: float = 0.02
    bos_index: int = 0
    eos_index: int = 1
    pad_index: int = 2
    unk_index: int = 3
    mask_index: int = 5
    is_encoder: bool = True
    summary_type: str = "first"
    summary_use_proj: bool = True
    summary_activation: str | None = None
    summary_proj_to_labels: bool = True
    summary_first_dropout: float | int = 0.1
    start_n_top: int = 5
    end_n_top: int = 5
    mask_token_id: int = 0
    lang_id: int = 0
    pad_token_id: int | None = 2
    bos_token_id: int | None = 0
    eos_token_id: int | list[int] | None = 1
    tie_word_embeddings: bool = True


__all__ = ["FlaubertConfig"]
