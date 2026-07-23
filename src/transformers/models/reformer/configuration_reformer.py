
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/reformer-crime-and-punishment")
@strict
class ReformerConfig(PreTrainedConfig):

    model_type = "reformer"
    keys_to_ignore_at_inference = ["past_buckets_states"]
    attribute_map = {}

    attention_head_size: int = 64
    attn_layers: list[str] | tuple[str, ...] = ("local", "lsh", "local", "lsh", "local", "lsh")
    axial_norm_std: float = 1.0
    axial_pos_embds: bool = True
    axial_pos_shape: list[int] | tuple[int, ...] = (64, 64)
    axial_pos_embds_dim: list[int] | tuple[int, ...] = (64, 192)
    chunk_size_lm_head: int = 0
    eos_token_id: int | list[int] | None = 2
    feed_forward_size: int = 512
    hash_seed: int | None = None
    hidden_act: str = "relu"
    hidden_dropout_prob: float | int = 0.05
    hidden_size: int = 256
    initializer_range: float = 0.02
    is_decoder: bool = False
    layer_norm_eps: float = 1e-12
    local_num_chunks_before: int = 1
    local_num_chunks_after: int = 0
    local_attention_probs_dropout_prob: float | int = 0.05
    local_attn_chunk_length: int = 64
    lsh_attn_chunk_length: int | None = 64
    lsh_attention_probs_dropout_prob: float | None = 0.0
    lsh_num_chunks_before: int | None = 1
    lsh_num_chunks_after: int | None = 0
    max_position_embeddings: int = 4096
    num_attention_heads: int = 12
    num_buckets: int | list[int] | None = None
    num_hashes: int = 1
    vocab_size: int = 320
    tie_word_embeddings: bool = False
    use_cache: bool = True
    classifier_dropout: float | int | None = None
    bos_token_id: int | None = None
    pad_token_id: int | None = 0

    def __post_init__(self, **kwargs):
        self.num_hidden_layers = len(self.attn_layers)
        self.axial_pos_shape = tuple(self.axial_pos_shape)
        self.axial_pos_embds_dim = tuple(self.axial_pos_embds_dim)
        super().__post_init__(**kwargs)


__all__ = ["ReformerConfig"]
