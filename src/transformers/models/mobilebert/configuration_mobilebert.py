
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/mobilebert-uncased")
@strict
class MobileBertConfig(PreTrainedConfig):

    model_type = "mobilebert"

    vocab_size: int = 30522
    hidden_size: int = 512
    num_hidden_layers: int = 24
    num_attention_heads: int = 4
    intermediate_size: int = 512
    hidden_act: str = "relu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    embedding_size: int = 128
    trigram_input: bool = True
    use_bottleneck: bool = True
    intra_bottleneck_size: int = 128
    use_bottleneck_attention: bool = False
    key_query_shared_bottleneck: bool = True
    num_feedforward_networks: int = 4
    normalization_type: str = "no_norm"
    classifier_activation: bool = True
    classifier_dropout: float | int | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if self.use_bottleneck:
            self.true_hidden_size = self.intra_bottleneck_size
        else:
            self.true_hidden_size = self.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["MobileBertConfig"]
