
from huggingface_hub.dataclasses import strict

from ...backbone_utils import BackboneConfigMixin
from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring


@auto_docstring(checkpoint="facebook/sapiens2-seg-0.4b")
@strict
class Sapiens2HeadConfig(PreTrainedConfig):

    model_type = "sapiens2_head"
    base_config_key = "head_config"

    upsample_out_channels: list[int] | None = None
    upsample_kernel_sizes: list[int] | None = None
    upsample_kernel_size: int = 4
    use_pixel_shuffle: bool | None = None
    conv_out_channels: list[int] | None = None
    conv_kernel_sizes: list[int] | None = None
    conv_kernel_size: int = 1
    scale_conv_out_channels: list[int] | None = None
    scale_conv_kernel_sizes: list[int] | None = None
    scale_conv_kernel_size: int = 1
    scale_final_input_size: int | None = None
    scale_final_hidden_sizes: list[int] | None = None

    def __post_init__(self, **kwargs):
        if self.upsample_out_channels is not None and self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [self.upsample_kernel_size] * len(self.upsample_out_channels)
        if self.conv_out_channels is not None and self.conv_kernel_sizes is None:
            self.conv_kernel_sizes = [self.conv_kernel_size] * len(self.conv_out_channels)
        if self.scale_conv_out_channels is not None and self.scale_conv_kernel_sizes is None:
            self.scale_conv_kernel_sizes = [self.scale_conv_kernel_size] * len(self.scale_conv_out_channels)
        super().__post_init__(**kwargs)

    def _init_scale_final_input_size(
        self, image_size: int | list[int] | tuple[int, int], patch_size: int | list[int] | tuple[int, int]
    ) -> None:
        if (
            self.scale_final_input_size is not None
            or self.scale_conv_out_channels is None
            or self.scale_conv_kernel_sizes is None
        ):
            return
        image_height, image_width = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        patch_height = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_width = patch_size if isinstance(patch_size, int) else patch_size[1]
        features_height = image_height // patch_height
        features_width = image_width // patch_width
        for kernel_size in self.scale_conv_kernel_sizes:
            padding = (kernel_size - 1) // 2
            features_height = (features_height + 2 * padding - kernel_size) // 2 + 1
            features_width = (features_width + 2 * padding - kernel_size) // 2 + 1
        self.scale_final_input_size = features_height * features_width * self.scale_conv_out_channels[-1]


@auto_docstring(checkpoint="facebook/sapiens2-pretrain-0.4b")
@strict
class Sapiens2Config(BackboneConfigMixin, PreTrainedConfig):

    model_type = "sapiens2"

    patch_size: int | list[int] | tuple[int, int] = 16

    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    hidden_act: str = "silu"
    attention_dropout: float | int = 0.0
    initializer_range: float = 0.02
    rope_theta: float = 100.0
    image_size: int | list[int] | tuple[int, int] = 224
    num_channels: int = 3
    query_bias: bool = True
    key_bias: bool = True
    value_bias: bool = True
    proj_bias: bool = True
    mlp_bias: bool = True
    layerscale_value: float = 1.0
    drop_path_rate: float | int = 0.0
    use_gated_mlp: bool = True
    num_register_tokens: int = 8
    pos_embed_shift: float | None = None
    pos_embed_jitter: float | None = None
    pos_embed_rescale: float | None = 2.0
    _out_features: list[str] | None = None
    _out_indices: list[int] | None = None
    reshape_hidden_states: bool = True
    sub_configs = {"head_config": Sapiens2HeadConfig}
    use_mask_token: bool = False
    rms_norm_eps: float = 1e-6
    normalize_backbone_outputs: bool = True
    use_qk_norm: bool = True
    num_key_value_heads_per_layer: list[int] | None = None
    num_key_value_attention_heads: int = 8
    num_first_full_attention_layers: int = 8
    num_last_full_attention_layers: int = 8
    semantic_loss_ignore_index: int = 255
    flip_pairs: list[list[int]] | None = None
    head_config: Sapiens2HeadConfig | dict | None = None

    def __post_init__(self, **kwargs):
        if self.num_key_value_heads_per_layer is None:
            self.num_key_value_heads_per_layer = [
                self.num_attention_heads
                if (
                    layer_index < self.num_first_full_attention_layers
                    or layer_index >= self.num_hidden_layers - self.num_last_full_attention_layers
                )
                else self.num_key_value_attention_heads
                for layer_index in range(self.num_hidden_layers)
            ]
        if isinstance(self.head_config, dict):
            self.head_config = Sapiens2HeadConfig(**self.head_config)
        if self.head_config is not None:
            self.head_config._init_scale_final_input_size(image_size=self.image_size, patch_size=self.patch_size)
        self.stage_names = ["stem"] + [f"stage{i}" for i in range(1, self.num_hidden_layers + 1)]
        self.set_output_features_output_indices(
            out_indices=kwargs.pop("out_indices", None), out_features=kwargs.pop("out_features", None)
        )
        super().__post_init__(**kwargs)


__all__ = ["Sapiens2Config", "Sapiens2HeadConfig"]
