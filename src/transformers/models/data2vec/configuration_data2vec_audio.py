
import math

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/data2vec-audio-base-960h")
@strict
class Data2VecAudioConfig(PreTrainedConfig):

    model_type = "data2vec-audio"

    vocab_size: int = 32
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    feat_proj_dropout: float | int = 0.0
    final_dropout: float | int = 0.1
    layerdrop: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    feat_extract_activation: str = "gelu"
    conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_bias: bool = False
    num_conv_pos_embedding_groups: int = 16
    conv_pos_kernel_size: int = 19
    num_conv_pos_embeddings: int = 5
    mask_time_prob: float | int = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float | int = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    ctc_loss_reduction: str = "sum"
    ctc_zero_infinity: bool = False
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
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
    num_adapter_layers: int = 3
    output_hidden_size: int | None = None

    def __post_init__(self, **kwargs):
        self.output_hidden_size = self.output_hidden_size or self.hidden_size
        self.num_feat_extract_layers = len(self.conv_dim)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def inputs_to_logits_ratio(self):
        pass


__all__ = ["Data2VecAudioConfig"]
