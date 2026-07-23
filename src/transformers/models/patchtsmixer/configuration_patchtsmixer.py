
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="ibm/patchtsmixer-etth1-pretrain")
@strict
class PatchTSMixerConfig(PreTrainedConfig):

    model_type = "patchtsmixer"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_layers",
    }

    context_length: int = 32
    patch_length: int = 8
    num_input_channels: int = 1
    patch_stride: int = 8
    num_parallel_samples: int = 100
    d_model: int = 8
    expansion_factor: int = 2
    num_layers: int = 3
    dropout: float | int = 0.2
    mode: str = "common_channel"
    gated_attn: bool = True
    norm_mlp: str = "LayerNorm"
    self_attn: bool = False
    self_attn_heads: int = 1
    use_positional_encoding: bool = False
    positional_encoding_type: str = "sincos"
    scaling: str | bool | None = "std"
    loss: str = "mse"
    init_std: float = 0.02
    norm_eps: float = 1e-5
    mask_type: str = "random"
    random_mask_ratio: float = 0.5
    num_forecast_mask_patches: list[int] | tuple[int, ...] | int | None = (2,)
    mask_value: int = 0
    masked_loss: bool = True
    channel_consistent_masking: bool = True
    unmasked_channel_indices: list[int] | None = None
    head_dropout: float | int = 0.2
    distribution_output: str = "student_t"
    prediction_length: int = 16
    prediction_channel_indices: list | None = None
    num_targets: int = 3
    output_range: list | None = None
    head_aggregation: str | None = "max_pool"

    def __post_init__(self, **kwargs):
        self.num_patches = (max(self.context_length, self.patch_length) - self.patch_length) // self.patch_stride + 1
        self.patch_last = True
        super().__post_init__(**kwargs)


__all__ = ["PatchTSMixerConfig"]
