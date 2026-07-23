

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="magic-leap-community/superpoint")
@strict
class SuperPointConfig(PreTrainedConfig):

    model_type = "superpoint"

    encoder_hidden_sizes: list[int] | tuple[int, ...] = (64, 64, 128, 128)
    decoder_hidden_size: int = 256
    keypoint_decoder_dim: int = 65
    descriptor_decoder_dim: int = 256
    keypoint_threshold: float = 0.005
    max_keypoints: int = -1
    nms_radius: int = 4
    border_removal_distance: int = 4
    initializer_range: float = 0.02


__all__ = ["SuperPointConfig"]
