
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from ...utils import auto_docstring


__all__ = ["RadioConfig"]


@auto_docstring(checkpoint="nvidia/C-RADIOv4-H")
@strict
class RadioConfig(PreTrainedConfig):

    model_type = "radio"

    hidden_size: int = 1280
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    mlp_ratio: float = 4.0
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-6
    attention_probs_dropout_prob: float = 0.0
    hidden_dropout_prob: float = 0.0
    drop_path_rate: float = 0.0
    use_swiglu_ffn: bool = False
    qkv_bias: bool = True
    layerscale_value: float = 1.0
    num_channels: int = 3
    patch_size: int = 16
    image_size: int = 224
    max_img_size: int = 2048
    num_cls_tokens: int = 3
    num_registers: int = 7
    summary_idxs: list[int] | None = None
    norm_mean: list[float] | tuple[float, float, float] = tuple(OPENAI_CLIP_MEAN)
    norm_std: list[float] | tuple[float, float, float] = tuple(OPENAI_CLIP_STD)
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.summary_idxs is None:
            self.summary_idxs = [0, 1]
        self.norm_mean = list(self.norm_mean)
        self.norm_std = list(self.norm_std)
        super().__post_init__(**kwargs)

    @property
    def num_summary_tokens(self) -> int:
        pass
