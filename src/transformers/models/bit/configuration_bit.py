
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/bit-50")
@strict
class BitConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "bit"
    layer_types = ["preactivation", "bottleneck"]
    supported_padding = [None, "SAME", "VALID"]

    num_channels: int = 3
    embedding_size: int = 64
    hidden_sizes: list[int] | tuple[int, ...] = (256, 512, 1024, 2048)
    depths: list[int] | tuple[int, ...] = (3, 4, 6, 3)
    layer_type: str = "preactivation"
    hidden_act: str = "relu"
    global_padding: str | None = None
    num_groups: int = 32
    drop_path_rate: float | int = 0.0
    embedding_dynamic_padding: bool = False
    output_stride: int = 32
    width_factor: int = 1
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None

    def __post_init__(self, **kwargs):
        self.hidden_sizes = list(self.hidden_sizes)
        self.depths = list(self.depths)

        if self.global_padding is not None:
            self.global_padding = self.global_padding.upper()

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )

        super().__post_init__(**kwargs)

    def validate_architecture(self):
        pass


__all__ = ["BitConfig"]
