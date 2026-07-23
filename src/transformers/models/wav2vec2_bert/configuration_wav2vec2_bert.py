
from typing import Literal

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/wav2vec2-bert-rel-pos-large")
@strict
class Wav2Vec2BertConfig(PreTrainedConfig):

    model_type = "wav2vec2-bert"

    vocab_size: int | None = None
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    feature_projection_input_dim: int = 160
    hidden_act: str = "swish"
    hidden_dropout: float | int = 0.0
    activation_dropout: float | int = 0.0
    attention_dropout: float | int = 0.0
    feat_proj_dropout: float | int = 0.0
    final_dropout: float | int = 0.1
    layerdrop: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    apply_spec_augment: bool = True
    mask_time_prob: float | int = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float | int = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    ctc_loss_reduction: str = "sum"
    ctc_zero_infinity: bool = False
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 768
    tdnn_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 1500)
    tdnn_kernel: list[int] | tuple[int, ...] = (5, 3, 3, 1, 1)
    tdnn_dilation: list[int] | tuple[int, ...] = (1, 2, 3, 1, 1)
    xvector_output_dim: int = 512
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2
    add_adapter: bool = False
    adapter_kernel_size: int = 3
    adapter_stride: int = 2
    num_adapter_layers: int = 1
    adapter_act: str = "relu"
    use_intermediate_ffn_before_adapter: bool = False
    output_hidden_size: int | None = None
    position_embeddings_type: Literal["rotary", "relative", "relative_key"] | None = "relative_key"
    rotary_embedding_base: int = 10000
    max_source_positions: int = 5000
    left_max_position_embeddings: int = 64
    right_max_position_embeddings: int = 8
    conv_depthwise_kernel_size: int = 31
    conformer_conv_dropout: float | int = 0.1

    def __post_init__(self, **kwargs):
        self.output_hidden_size = self.output_hidden_size or self.hidden_size
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def inputs_to_logits_ratio(self):
        pass


__all__ = ["Wav2Vec2BertConfig"]
