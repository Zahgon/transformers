
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="huggingface/informer-tourism-monthly")
@strict
class InformerConfig(PreTrainedConfig):

    model_type = "informer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
        "initializer_range": "init_std",
    }

    prediction_length: int | None = None
    context_length: int | None = None
    distribution_output: str = "student_t"
    loss: str = "nll"
    input_size: int = 1
    lags_sequence: list[int] | None = None
    scaling: str | bool | None = "mean"
    num_dynamic_real_features: int = 0
    num_static_real_features: int = 0
    num_static_categorical_features: int = 0
    num_time_features: int = 0
    cardinality: list[int] | None = None
    embedding_dimension: list[int] | None = None
    d_model: int = 64
    encoder_ffn_dim: int = 32
    decoder_ffn_dim: int = 32
    encoder_attention_heads: int = 2
    decoder_attention_heads: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 2
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    dropout: float | int = 0.05
    encoder_layerdrop: float | int = 0.1
    decoder_layerdrop: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    num_parallel_samples: int = 100
    init_std: float = 0.02
    use_cache: bool = True
    attention_type: str = "prob"
    sampling_factor: int = 5
    distil: bool = True

    def __post_init__(self, **kwargs):
        self.context_length = self.context_length or self.prediction_length
        self.lags_sequence = self.lags_sequence if self.lags_sequence is not None else [1, 2, 3, 4, 5, 6, 7]

        if not (self.cardinality and self.num_static_categorical_features > 0):
            self.cardinality = [0]

        if not (self.embedding_dimension and self.num_static_categorical_features > 0):
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]

        self.feature_size = self.input_size * len(self.lags_sequence) + self._number_of_features
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def _number_of_features(self) -> int:
        pass


__all__ = ["InformerConfig"]
