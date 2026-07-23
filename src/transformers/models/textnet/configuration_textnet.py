
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="czczup/textnet-base")
@strict
class TextNetConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "textnet"

    stem_kernel_size: int = 3
    stem_stride: int = 2
    stem_num_channels: int = 3
    stem_out_channels: int = 64
    stem_act_func: str = "relu"
    image_size: list[int] | tuple[int, int] | int = (640, 640)
    conv_layer_kernel_sizes: list | None = None
    conv_layer_strides: list | None = None
    hidden_sizes: list[int] | tuple[int, ...] = (64, 64, 128, 256, 512)
    batch_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.conv_layer_kernel_sizes is None:
            self.conv_layer_kernel_sizes = [
                [[3, 3], [3, 3], [3, 3]],
                [[3, 3], [1, 3], [3, 3], [3, 1]],
                [[3, 3], [3, 3], [3, 1], [1, 3]],
                [[3, 3], [3, 1], [1, 3], [3, 3]],
            ]
        if self.conv_layer_strides is None:
            self.conv_layer_strides = [[1, 2, 1], [2, 1, 1, 1], [2, 1, 1, 1], [2, 1, 1, 1]]

        self.depths = [len(layer) for layer in self.conv_layer_kernel_sizes]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, 5)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["TextNetConfig"]
