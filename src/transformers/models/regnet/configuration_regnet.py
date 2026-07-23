
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/regnet-y-040")
@strict
class RegNetConfig(PreTrainedConfig):

    model_type = "regnet"
    layer_types = ["x", "y"]

    num_channels: int = 3
    embedding_size: int = 32
    hidden_sizes: list[int] | tuple[int, ...] = (128, 192, 512, 1088)
    depths: list[int] | tuple[int, ...] = (2, 6, 12, 2)
    groups_width: int = 64
    layer_type: str = "y"
    hidden_act: str = "relu"
    downsample_in_first_stage: bool = True

    def validate_architecture(self):
        pass


__all__ = ["RegNetConfig"]
