
from huggingface_hub.dataclasses import strict

from ...backbone_utils import consolidate_backbone_kwargs_to_config
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="Intel/dpt-large")
@strict
class DPTConfig(PreTrainedConfig):

    model_type = "dpt"
    sub_configs = {"backbone_config": AutoConfig}

    hidden_size: int = 768
    num_hidden_layers: None | int = 12
    num_attention_heads: int | None = 12
    intermediate_size: int | None = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int | None = 0.0
    attention_probs_dropout_prob: float | int | None = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float | None = 1e-12
    image_size: int | list[int] | tuple[int, int] | None = 384
    patch_size: int | list[int] | tuple[int, int] | None = 16
    num_channels: int | None = 3
    is_hybrid: bool = False
    qkv_bias: bool | None = True
    backbone_out_indices: list[int] | tuple[int, ...] | None = (2, 5, 8, 11)
    readout_type: str = "project"
    reassemble_factors: list[int | float] | tuple[int | float, ...] = (4, 2, 1, 0.5)
    neck_hidden_sizes: list[int] | tuple[int, ...] = (96, 192, 384, 768)
    fusion_hidden_size: int = 256
    head_in_index: int = -1
    use_batch_norm_in_fusion_residual: bool | None = False
    use_bias_in_fusion_residual: bool | None = None
    add_projection: bool = False
    use_auxiliary_head: bool | None = True
    auxiliary_loss_weight: float = 0.4
    semantic_loss_ignore_index: int = 255
    semantic_classifier_dropout: float | int = 0.1
    backbone_featmap_shape: list[int] | tuple[int, ...] | None = (1, 1024, 24, 24)
    neck_ignore_stages: list[int] | tuple[int, ...] = (0, 1)
    backbone_config: dict | PreTrainedConfig | None = None
    pooler_output_size: int | None = None
    pooler_act: str = "tanh"

    def __post_init__(self, **kwargs):
        if self.readout_type not in ["ignore", "add", "project"]:
            raise ValueError("Readout_type must be one of ['ignore', 'add', 'project']")

        if self.is_hybrid:
            if isinstance(self.backbone_config, dict):
                self.backbone_config.setdefault("model_type", "bit")

            self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
                backbone_config=self.backbone_config,
                default_config_type="bit",
                default_config_kwargs={
                    "global_padding": "same",
                    "layer_type": "bottleneck",
                    "depths": [3, 4, 9],
                    "out_features": ["stage1", "stage2", "stage3"],
                    "embedding_dynamic_padding": True,
                },
                **kwargs,
            )
            if self.readout_type != "project":
                raise ValueError("Readout type must be 'project' when using `DPT-hybrid` mode.")
        elif kwargs.get("backbone") is not None or self.backbone_config is not None:
            self.backbone_config, kwargs = consolidate_backbone_kwargs_to_config(
                backbone_config=self.backbone_config,
                **kwargs,
            )
            self.backbone_out_indices = None

        self.backbone_featmap_shape = self.backbone_featmap_shape if self.is_hybrid else None
        self.neck_ignore_stages = self.neck_ignore_stages if self.is_hybrid else []
        self.pooler_output_size = self.pooler_output_size if self.pooler_output_size else self.hidden_size
        super().__post_init__(**kwargs)


__all__ = ["DPTConfig"]
