
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/mobilenet_v1_1.0_224")
@strict
class MobileNetV1Config(PreTrainedConfig):

    model_type = "mobilenet_v1"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    depth_multiplier: float | int = 1.0
    min_depth: int = 8
    hidden_act: str = "relu6"
    tf_padding: bool = True
    classifier_dropout_prob: float | int = 0.999
    initializer_range: float = 0.02
    layer_norm_eps: float = 0.001

    def validate_architecture(self):
        pass


__all__ = ["MobileNetV1Config"]
