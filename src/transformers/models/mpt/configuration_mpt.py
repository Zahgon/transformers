
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="mosaicml/mpt-7b")
@strict
class MptAttentionConfig(PreTrainedConfig):

    base_config_key = "attn_config"

    attn_type: Literal["multihead_attention", "multiquery_attention"] = "multihead_attention"
    attn_pdrop: int = 0
    attn_impl: str = "torch"
    clip_qkv: float | None = None
    softmax_scale: float | None = None
    prefix_lm: bool = False
    qk_ln: bool = False
    attn_uses_sequence_id: bool = False
    alibi: bool = True
    alibi_bias_max: int = 8


@auto_docstring(checkpoint="mosaicml/mpt-7b")
@strict
class MptConfig(PreTrainedConfig):

    model_type = "mpt"
    sub_configs = {"attn_config": MptAttentionConfig}
    attribute_map = {
        "num_attention_heads": "n_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
    }

    d_model: int = 2048
    n_heads: int = 16
    n_layers: int = 24
    expansion_ratio: int = 4
    max_seq_len: int = 2048
    vocab_size: int = 50368
    resid_pdrop: float | int = 0.0
    layer_norm_epsilon: float = 1e-5
    emb_pdrop: float | int = 0.0
    learned_pos_emb: bool = True
    attn_config: dict | MptAttentionConfig | None = None
    init_device: str = "cpu"
    logit_scale: float | str | None = None
    no_bias: bool = True
    embedding_fraction: float = 1.0
    norm_type: str = "low_precision_layernorm"
    use_cache: bool = False
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.attn_config is None:
            self.attn_config = MptAttentionConfig()
        elif isinstance(self.attn_config, dict):
            self.attn_config = MptAttentionConfig(**self.attn_config)
        super().__post_init__(**kwargs)


__all__ = ["MptConfig"]
