
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="microsoft/beit-base-patch16-224-pt22k")
@strict
class BeitConfig(BackboneConfigMixin, PreTrainedConfig):

    model_type = "beit"

    vocab_size: int = 8192
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float | int = 0.0
    attention_probs_dropout_prob: float | int = 0.0
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    image_size: int | list[int] | tuple[int, int] = 224
    patch_size: int | list[int] | tuple[int, int] = 16
    num_channels: int = 3
    use_mask_token: bool = False
    use_absolute_position_embeddings: bool = False
    use_relative_position_bias: bool = False
    use_shared_relative_position_bias: bool = False
    layer_scale_init_value: float = 0.1
    drop_path_rate: float | int = 0.1
    use_mean_pooling: bool = True
    pool_scales: list[int] | tuple[int, ...] = (1, 2, 3, 6)
    use_auxiliary_head: bool = True
    auxiliary_loss_weight: float = 0.4
    auxiliary_channels: int = 256
    auxiliary_num_convs: int = 1
    auxiliary_concat_input: bool = False
    semantic_loss_ignore_index: int = 255
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    add_fpn: bool = False
    reshape_hidden_states: bool = True

    def __post_init__(self, **kwargs):
        if "segmentation_indices" in kwargs and kwargs.get("out_indices") is None:
            kwargs["out_indices"] = kwargs.pop("segmentation_indices")

        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )

        if self.add_fpn and (self._out_indices is None or len(self._out_indices) != 4):
            raise ValueError(
                "BeitConfig requires `out_indices` to be a list of exactly 4 integers when `add_fpn=True`, "
                "specifying which features to use from the backbone. One can use `out_indices=[3, 5, 7, 11]` "
                "for a base-sized architecture."
            )

        super().__post_init__(**kwargs)


__all__ = ["BeitConfig"]
