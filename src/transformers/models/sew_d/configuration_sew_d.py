
import functools
import operator

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="BAAI/seggpt-vit-large")
@strict
class SEWDConfig(PreTrainedConfig):

    model_type = "sew-d"
    vocab_size: int = 32
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    squeeze_factor: int = 2
    max_position_embeddings: int = 512
    position_buckets: int = 256
    share_att_key: bool = True
    relative_attention: bool = True
    pos_att_type: list[str] | tuple[str, ...] = ("p2c", "c2p")
    norm_rel_ebd: str = "layer_norm"
    hidden_act: str = "gelu_python"
    hidden_dropout: float | int = 0.1
    activation_dropout: float | int = 0.1
    attention_dropout: float | int = 0.1
    feat_proj_dropout: float | int = 0.0
    final_dropout: float | int = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-7
    feature_layer_norm_eps: float = 1e-5
    feat_extract_norm: str = "group"
    feat_extract_activation: str = "gelu"
    conv_dim: list[int] | tuple[int, ...] = (64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512)
    conv_stride: list[int] | tuple[int, ...] = (5, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1)
    conv_kernel: list[int] | tuple[int, ...] = (10, 3, 1, 3, 1, 3, 1, 3, 1, 2, 1, 2, 1)
    conv_bias: bool = False
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    apply_spec_augment: bool = True
    mask_time_prob: float | int = 0.05
    mask_time_length: int = 10
    mask_time_min_masks: int = 2
    mask_feature_prob: float | int = 0.0
    mask_feature_length: int = 10
    mask_feature_min_masks: int = 0
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = False
    use_weighted_layer_sum: bool = False
    classifier_proj_size: int = 256
    pad_token_id: int | None = 0
    bos_token_id: int | None = 1
    eos_token_id: int | list[int] | None = 2

    def __post_init__(self, **kwargs):
        self.num_feat_extract_layers = len(self.conv_dim)
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass

    @property
    def inputs_to_logits_ratio(self):
        pass


__all__ = ["SEWDConfig"]
