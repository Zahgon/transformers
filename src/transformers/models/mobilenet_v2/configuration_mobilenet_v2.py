
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="google/mobilenet_v2_1.0_224")
@strict
class MobileNetV2Config(PreTrainedConfig):

    model_type = "mobilenet_v2"

    num_channels: int = 3
    image_size: int | list[int] | tuple[int, int] = 224
    depth_multiplier: float | int = 1.0
    depth_divisible_by: int = 8
    min_depth: int = 8
    expand_ratio: float | int = 6.0
    output_stride: int = 32
    first_layer_is_expansion: bool = True
    finegrained_output: bool = True
    hidden_act: str = "relu6"
    tf_padding: bool = True
    classifier_dropout_prob: float | int = 0.8
    initializer_range: float = 0.02
    layer_norm_eps: float = 0.001
    semantic_loss_ignore_index: int = 255

    def validate_architecture(self):
        pass


__all__ = ["MobileNetV2Config"]
