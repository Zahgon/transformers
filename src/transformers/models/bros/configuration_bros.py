
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="jinho8345/bros-base-uncased")
@strict
class BrosConfig(PreTrainedConfig):

    model_type = "bros"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    dim_bbox: int = 8
    bbox_scale: float = 100.0
    n_relations: int = 1
    classifier_dropout_prob: float | int = 0.1
    is_decoder: bool = False
    add_cross_attention: bool = False

    def __post_init__(self, **kwargs):
        self.dim_bbox_sinusoid_emb_2d = self.hidden_size // 4
        self.dim_bbox_sinusoid_emb_1d = self.dim_bbox_sinusoid_emb_2d // self.dim_bbox
        self.dim_bbox_projection = self.hidden_size // self.num_attention_heads
        super().__post_init__(**kwargs)


__all__ = ["BrosConfig"]
