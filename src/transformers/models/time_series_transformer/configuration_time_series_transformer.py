
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="huggingface/time-series-transformer-tourism-monthly")
@strict
class TimeSeriesTransformerConfig(PreTrainedConfig):

    model_type = "time_series_transformer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
        "num_hidden_layers": "encoder_layers",
    }

    prediction_length: int | None = None
    context_length: int | None = None
    distribution_output: str = "student_t"
    loss: str = "nll"
    input_size: int = 1
    lags_sequence: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7)
    scaling: str | bool | None = "mean"
    num_dynamic_real_features: int = 0
    num_static_categorical_features: int = 0
    num_static_real_features: int = 0
    num_time_features: int = 0
    cardinality: list[int] | None = None
    embedding_dimension: list[int] | None = None
    encoder_ffn_dim: int = 32
    decoder_ffn_dim: int = 32
    encoder_attention_heads: int = 2
    decoder_attention_heads: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 2
    is_encoder_decoder: bool = True
    activation_function: str = "gelu"
    d_model: int = 64
    dropout: float | int = 0.1
    encoder_layerdrop: float | int = 0.1
    decoder_layerdrop: float | int = 0.1
    attention_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    num_parallel_samples: int = 100
    init_std: float = 0.02
    use_cache: bool = True

    def __post_init__(self, **kwargs):
        if not (self.cardinality and self.num_static_categorical_features > 0):
            self.cardinality = [0]

        if not (self.embedding_dimension and self.num_static_categorical_features > 0):
            self.embedding_dimension = [min(50, (cat + 1) // 2) for cat in self.cardinality]

        self.context_length = self.context_length or self.prediction_length
        self.feature_size = self.input_size * len(self.lags_sequence) + self._number_of_features
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def _number_of_features(self) -> int:
        pass


__all__ = ["TimeSeriesTransformerConfig"]
