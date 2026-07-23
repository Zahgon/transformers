

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="PaddlePaddle/Not_yet_released")
@strict
class PPLCNetV3Config(BackboneConfigMixin, PreTrainedConfig):

    model_type = "pp_lcnet_v3"

    scale: float | int = 1.0
    block_configs: list | None = None
    stem_channels: int = 16
    stem_stride: int = 2
    reduction: int = 4
    divisor: int = 8
    hidden_act: str = "hardswish"
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    conv_symmetric_num: int = 4

    def __post_init__(self, **kwargs):
        self.block_configs = (
            [
                [[3, 16, 32, 1, False]],
                [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
                [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
                [
                    [3, 128, 256, 2, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                    [5, 256, 256, 1, False],
                ],
                [[5, 256, 512, 2, True], [5, 512, 512, 1, True], [5, 512, 512, 1, False], [5, 512, 512, 1, False]],
            ]
            if self.block_configs is None
            else self.block_configs
        )
        self.depths = [len(blocks) for blocks in self.block_configs]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.block_configs) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["PPLCNetV3Config"]
