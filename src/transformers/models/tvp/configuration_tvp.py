
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto import AutoConfig


@auto_docstring(checkpoint="Intel/tvp-base")
@strict
class TvpConfig(PreTrainedConfig):

    model_type = "tvp"
    sub_configs = {"backbone_config": AutoConfig}

    backbone_config: dict | PreTrainedConfig | None = None
    distance_loss_weight: float = 1.0
    duration_loss_weight: float = 0.1
    visual_prompter_type: str = "framepad"
    visual_prompter_apply: str = "replace"
    visual_prompt_size: int = 96
    max_img_size: int = 448
    num_frames: int = 48
    vocab_size: int = 30522
    type_vocab_size: int = 2
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    max_grid_col_position_embeddings: int = 100
    max_grid_row_position_embeddings: int = 100
    hidden_dropout_prob: float | int = 0.1
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    attention_probs_dropout_prob: float | int = 0.1
    pad_token_id: int | None = None

    def __post_init__(self, **kwargs):
        self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
            backbone_config=self.backbone_config,
            default_config_type="resnet",
            default_config_kwargs={"out_features": ["stage4"]},
            **kwargs,
        )

        super().__post_init__(**kwargs)


__all__ = ["TvpConfig"]
