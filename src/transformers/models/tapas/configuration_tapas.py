
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/tapas-base-finetuned-sqa")
@strict
class TapasConfig(PreTrainedConfig):

    model_type = "tapas"

    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 1024
    type_vocab_sizes: list[int] | tuple[int, ...] = (3, 256, 256, 2, 256, 256, 10)
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int | None = 0
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    positive_label_weight: float = 10.0
    num_aggregation_labels: int = 0
    aggregation_loss_weight: float = 1.0
    use_answer_as_supervision: bool | None = None
    answer_loss_importance: float = 1.0
    use_normalized_answer_loss: bool = False
    huber_loss_delta: float | None = None
    temperature: float = 1.0
    aggregation_temperature: float = 1.0
    use_gumbel_for_cells: bool = False
    use_gumbel_for_aggregation: bool = False
    average_approximation_function: str = "ratio"
    cell_selection_preference: float | None = None
    answer_loss_cutoff: float | int | None = None
    max_num_rows: int = 64
    max_num_columns: int = 32
    average_logits_per_cell: bool = False
    select_one_column: bool = True
    allow_empty_column_selection: bool = False
    init_cell_selection_weights_to_zero: bool = False
    reset_position_index_per_cell: bool = True
    disable_per_token_loss: bool = False
    aggregation_labels: dict | None = None
    no_aggregation_label_index: int | None = None
    is_decoder: bool = False
    add_cross_attention: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        if isinstance(self.aggregation_labels, dict):
            self.aggregation_labels = {int(k): v for k, v in self.aggregation_labels.items()}
        super().__post_init__(**kwargs)


__all__ = ["TapasConfig"]
