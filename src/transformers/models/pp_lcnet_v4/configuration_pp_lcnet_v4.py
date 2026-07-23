

from collections.abc import Sequence

from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="PaddlePaddle/Not_yet_released")
@strict
class PPLCNetV4Config(BackboneConfigMixin, PreTrainedConfig):

    model_type = "pp_lcnet_v4"

    scale: float | int = 1.0
    block_configs: list | None = None
    stem_channels: list[int] | tuple[int, ...] = (3, 48, 96)
    reduction: int = 4
    hidden_act: str = "relu"
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    num_channels: int = 3
    stem_strides: Sequence[int | list[int] | tuple[int, ...]] = (2, 1, 1, 2, 1)
    stem_type: str = "large"
    use_learnable_affine_block: bool = False

    def __post_init__(self, **kwargs):
        self.block_configs = (
            [
                [[3, 96, 96, 1, True]],
                [[3, 96, 96, 1, False], [3, 96, 96, 1, False]],
                [
                    [3, 96, 192, [2, 1], False],
                    [3, 192, 192, 1, True],
                    [3, 192, 192, 1, False],
                    [3, 192, 192, 1, True],
                    [3, 192, 192, 1, False],
                    [3, 192, 192, 1, True],
                    [3, 192, 192, 1, False],
                ],
                [
                    [3, 192, 384, [2, 1], False],
                    [3, 384, 384, 1, True],
                    [3, 384, 384, 1, False],
                ],
            ]
            if self.block_configs is None
            else self.block_configs
        )

        self.depths = [len(blocks) for blocks in self.block_configs]
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.block_configs) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        self.stage_out_channels = [blocks[-1][2] for blocks in self.block_configs]
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["PPLCNetV4Config"]
