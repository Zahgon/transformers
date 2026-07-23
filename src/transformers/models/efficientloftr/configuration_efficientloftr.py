

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="zju-community/efficientloftr")
@strict
class EfficientLoFTRConfig(PreTrainedConfig):

    model_type = "efficientloftr"

    stage_num_blocks: list[int] | None = None
    out_features: list[int] | None = None
    stage_stride: list[int] | None = None
    hidden_size: int = 256
    activation_function: str = "relu"
    q_aggregation_kernel_size: int = 4
    kv_aggregation_kernel_size: int = 4
    q_aggregation_stride: int = 4
    kv_aggregation_stride: int = 4
    num_attention_layers: int = 4
    num_attention_heads: int = 8
    attention_dropout: float | int = 0.0
    attention_bias: bool = False
    mlp_activation_function: str = "leaky_relu"
    coarse_matching_skip_softmax: bool = False
    coarse_matching_threshold: float = 0.2
    coarse_matching_temperature: float = 0.1
    coarse_matching_border_removal: int = 2
    fine_kernel_size: int = 8
    batch_norm_eps: float = 1e-5
    rope_parameters: dict | None = None
    fine_matching_slice_dim: int = 8
    fine_matching_regress_temperature: float = 10.0
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        self.stage_num_blocks = self.stage_num_blocks if self.stage_num_blocks is not None else [1, 2, 4, 14]
        self.stage_stride = self.stage_stride if self.stage_stride is not None else [2, 1, 2, 2]
        self.out_features = self.out_features if self.out_features is not None else [64, 64, 128, 256]
        self.stage_in_channels = [1] + self.out_features[:-1]

        self.stage_block_stride = [
            [stride] + [1] * (num_blocks - 1) for stride, num_blocks in zip(self.stage_stride, self.stage_num_blocks)
        ]
        self.stage_block_out_channels = [
            [self.out_features[stage_idx]] * num_blocks for stage_idx, num_blocks in enumerate(self.stage_num_blocks)
        ]
        self.stage_block_in_channels = [
            [self.stage_in_channels[stage_idx]] + self.stage_block_out_channels[stage_idx][:-1]
            for stage_idx in range(len(self.stage_num_blocks))
        ]

        self.num_key_value_heads = self.num_attention_heads
        self.fine_fusion_dims = list(reversed(self.out_features))[:-1]
        self.intermediate_size = self.hidden_size * 2
        kwargs.setdefault("partial_rotary_factor", 4.0)  # assign default for BC
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["EfficientLoFTRConfig"]
