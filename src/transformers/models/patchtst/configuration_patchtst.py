
from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import PreTrainedConfig
from transformers.utils import auto_docstring


@auto_docstring(checkpoint="ibm-granite/granite-timeseries-patchtst")
@strict
class PatchTSTConfig(PreTrainedConfig):

    model_type = "patchtst"
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
    }

    num_input_channels: int = 1
    context_length: int = 32
    distribution_output: str = "student_t"
    loss: str | None = "mse"
    patch_length: int = 1
    patch_stride: int = 1
    num_hidden_layers: int = 3
    d_model: int = 128
    num_attention_heads: int = 4
    share_embedding: bool = True
    channel_attention: bool = False
    ffn_dim: int = 512
    norm_type: str = "batchnorm"
    norm_eps: float = 1e-05
    attention_dropout: float | int = 0.0
    positional_dropout: float | int = 0.0
    path_dropout: float | int = 0.0
    ff_dropout: float | int = 0.0
    bias: bool = True
    activation_function: str = "gelu"
    pre_norm: bool = True
    positional_encoding_type: str = "sincos"
    use_cls_token: bool = False
    init_std: float = 0.02
    share_projection: bool = True
    scaling: str | bool | None = "std"
    do_mask_input: bool | None = None
    mask_type: str = "random"
    random_mask_ratio: float = 0.5
    num_forecast_mask_patches: list[int] | tuple[int, ...] | int | None = (2,)
    channel_consistent_masking: bool | None = False
    unmasked_channel_indices: list[int] | None = None
    mask_value: int = 0
    pooling_type: str | None = "mean"
    head_dropout: float | int = 0.0
    prediction_length: int = 24
    num_targets: int = 1
    output_range: list | None = None
    num_parallel_samples: int = 100


__all__ = ["PatchTSTConfig"]
