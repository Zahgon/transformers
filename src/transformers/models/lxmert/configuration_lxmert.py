
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="unc-nlp/lxmert-base-uncased")
@strict
class LxmertConfig(PreTrainedConfig):

    model_type = "lxmert"
    attribute_map = {}

    vocab_size: int = 30522
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_qa_labels: int = 9500
    num_object_labels: int = 1600
    num_attr_labels: int = 400
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.1
    attention_probs_dropout_prob: float | int = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    l_layers: int = 9
    x_layers: int = 5
    r_layers: int = 5
    visual_feat_dim: int = 2048
    visual_pos_dim: int = 4
    visual_loss_normalizer: float = 6.67
    task_matched: bool = True
    task_mask_lm: bool = True
    task_obj_predict: bool = True
    task_qa: bool = True
    visual_obj_loss: bool = True
    visual_attr_loss: bool = True
    visual_feat_loss: bool = True
    pad_token_id: int | None = None
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs):
        self.num_hidden_layers = {"vision": self.r_layers, "cross_encoder": self.x_layers, "language": self.l_layers}
        super().__post_init__(**kwargs)


__all__ = ["LxmertConfig"]
