
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ... import initialization as init
from ...backbone_utils import load_backbone
from ...modeling_outputs import ModelOutput
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import OutputRecorder, capture_outputs
from ..auto import AutoConfig
from ..d_fine.configuration_d_fine import DFineConfig
from ..d_fine.modeling_d_fine import (
    DFineAIFILayer,
    DFineConvNormLayer,
    DFineDecoder,
    DFineDecoderLayer,
    DFineDecoderOutput,
    DFineEncoderLayer,
    DFineForObjectDetection,
    DFineGate,
    DFineHybridEncoder,
    DFineIntegral,
    DFineLQE,
    DFineMLP,
    DFineModel,
    DFineModelOutput,
    DFineMultiscaleDeformableAttention,
    DFinePreTrainedModel,
    DFineRepVggBlock,
    DFineSCDown,
    get_contrastive_denoising_training_group,
    replace_batch_norm,
)
from ..llama.modeling_llama import LlamaMLP, LlamaRMSNorm


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="Intellindust/DEIMv2_HGNetv2_N_COCO")
@strict
class Deimv2Config(DFineConfig):

    model_type = "deimv2"
    sub_configs = {"backbone_config": AutoConfig}

    eval_size: list[int] | tuple[int, int] | None = None
    weight_loss_mal: float = 1.0
    use_dense_one_to_one: bool = True
    mal_alpha: float | None = None
    encoder_fuse_op: str = "sum"
    spatial_tuning_adapter_inplanes: int = 16
    encoder_type: str = "hybrid"
    use_gateway: bool = True
    share_bbox_head: bool = False
    encoder_has_trailing_conv: bool = True


class Deimv2DecoderOutput(DFineDecoderOutput):
    pass


class Deimv2ModelOutput(DFineModelOutput):
    pass


@auto_docstring(
    custom_intro="""
    Output type for DEIMv2 encoder modules (HybridEncoder and LiteEncoder).
    Attentions are only available for HybridEncoder variants with AIFI layers.
    """
)
@dataclass
class Deimv2EncoderOutput(ModelOutput):

    feature_maps: list[torch.FloatTensor] = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class Deimv2RMSNorm(LlamaRMSNorm):
    pass


class Deimv2SwiGLUFFN(LlamaMLP):
    def __init__(self, config: Deimv2Config):
        nn.Module.__init__(self)
        hidden_features = config.decoder_ffn_dim // 2
        self.gate_proj = nn.Linear(config.d_model, hidden_features, bias=True)
        self.up_proj = nn.Linear(config.d_model, hidden_features, bias=True)
        self.down_proj = nn.Linear(hidden_features, config.d_model, bias=True)
        self.act_fn = nn.SiLU()


class Deimv2Gate(DFineGate):
    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.norm = Deimv2RMSNorm(d_model)


class Deimv2MLP(DFineMLP):
    pass


class Deimv2MultiscaleDeformableAttention(DFineMultiscaleDeformableAttention):
    pass


class Deimv2ConvNormLayer(DFineConvNormLayer):
    pass


class Deimv2RepVggBlock(DFineRepVggBlock):
    pass


class Deimv2CSPRepLayer(nn.Module):

    def __init__(
        self, config: Deimv2Config, in_channels: int, out_channels: int, num_blocks: int, expansion: float = 1.0
    ):
        super().__init__()
        activation = config.activation_function
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Deimv2ConvNormLayer(config, in_channels, hidden_channels * 2, 1, 1, activation=activation)
        self.bottlenecks = nn.ModuleList(
            [Deimv2RepVggBlock(config, hidden_channels, hidden_channels) for _ in range(num_blocks)]
        )
        self.conv2 = (
            Deimv2ConvNormLayer(config, hidden_channels, out_channels, 3, 1, activation=activation)
            if config.encoder_has_trailing_conv
            else nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual, hidden_states = self.conv1(hidden_states).chunk(2, dim=1)
        for bottleneck in self.bottlenecks:
            hidden_states = bottleneck(hidden_states)
        return self.conv2(residual + hidden_states)


class Deimv2RepNCSPELAN5(nn.Module):

    def __init__(self, config: Deimv2Config, numb_blocks: int = 3):
        super().__init__()
        activation = config.activation_function
        in_channels = config.encoder_hidden_dim
        out_channels = config.encoder_hidden_dim
        split_channels = config.encoder_hidden_dim * 2
        csp_channels = round(config.hidden_expansion * config.encoder_hidden_dim // 2)
        self.conv1 = Deimv2ConvNormLayer(config, in_channels, split_channels, 1, 1, activation=activation)
        self.csp_rep1 = Deimv2CSPRepLayer(config, split_channels // 2, csp_channels, num_blocks=numb_blocks)
        self.csp_rep2 = Deimv2CSPRepLayer(config, csp_channels, csp_channels, num_blocks=numb_blocks)
        self.conv2 = Deimv2ConvNormLayer(
            config, split_channels + (2 * csp_channels), out_channels, 1, 1, activation=activation
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_1, hidden_states_2 = self.conv1(hidden_states).chunk(2, dim=1)
        hidden_states_3 = self.csp_rep1(hidden_states_2)
        hidden_states_4 = self.csp_rep2(hidden_states_3)
        merged_hidden_states = torch.cat([hidden_states_1, hidden_states_2, hidden_states_3, hidden_states_4], dim=1)
        return self.conv2(merged_hidden_states)


class Deimv2SCDown(DFineSCDown):
    pass


class Deimv2EncoderLayer(DFineEncoderLayer):
    pass


class Deimv2AIFILayer(DFineAIFILayer):
    pass


class Deimv2SpatialTuningAdapter(nn.Module):
    def __init__(self, config: Deimv2Config):
        super().__init__()
        inplanes = config.spatial_tuning_adapter_inplanes
        self.stem_conv = Deimv2ConvNormLayer(config, 3, inplanes, 3, 2, activation="gelu")
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = Deimv2ConvNormLayer(config, inplanes, 2 * inplanes, 3, 2)
        self.conv3 = Deimv2ConvNormLayer(config, 2 * inplanes, 4 * inplanes, 3, 2)
        self.conv4 = Deimv2ConvNormLayer(config, 4 * inplanes, 4 * inplanes, 3, 2)
        self.act_fn = nn.GELU()

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states_1 = self.stem_pool(self.stem_conv(pixel_values))
        hidden_states_2 = self.conv2(hidden_states_1)
        hidden_states_3 = self.conv3(self.act_fn(hidden_states_2))
        hidden_states_4 = self.conv4(self.act_fn(hidden_states_3))
        return hidden_states_2, hidden_states_3, hidden_states_4


def fuse_feature_maps(feature_map_1: torch.Tensor, feature_map_2: torch.Tensor, fuse_op: str = "sum") -> torch.Tensor:
    """Fuses two feature maps via element-wise sum or channel-wise concatenation."""
    if fuse_op == "sum":
        return feature_map_1 + feature_map_2
    return torch.cat([feature_map_1, feature_map_2], dim=1)


class Deimv2Integral(DFineIntegral):
    pass


class Deimv2LQE(DFineLQE):
    pass


class Deimv2DecoderLayer(DFineDecoderLayer):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)
        self.encoder_attn = Deimv2MultiscaleDeformableAttention(config=config)
        self.self_attn_layer_norm = Deimv2RMSNorm(config.d_model)
        self.final_layer_norm = Deimv2RMSNorm(config.d_model)
        self.mlp = Deimv2SwiGLUFFN(config)
        self.use_gateway = config.use_gateway
        self.gateway = Deimv2Gate(config.d_model) if config.use_gateway else None
        self.encoder_attn_layer_norm = None if config.use_gateway else Deimv2RMSNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        reference_points: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states

        hidden_states = hidden_states if position_embeddings is None else hidden_states + position_embeddings
        hidden_states, _ = self.encoder_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        if self.gateway is not None:
            hidden_states = self.gateway(residual, hidden_states)
        else:
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class Deimv2PreTrainedModel(DFinePreTrainedModel):
    _no_split_modules = [r"Deimv2HybridEncoder", r"Deimv2LiteEncoder", r"Deimv2DecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)

        if isinstance(module, Deimv2SwiGLUFFN):
            init.xavier_uniform_(module.gate_proj.weight)
            init.constant_(module.gate_proj.bias, 0)
            init.xavier_uniform_(module.up_proj.weight)
            init.constant_(module.up_proj.bias, 0)
            init.xavier_uniform_(module.down_proj.weight)
            init.constant_(module.down_proj.bias, 0)


class Deimv2ConvEncoder(Deimv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        backbone = load_backbone(config)

        if config.freeze_backbone_batch_norms:
            with torch.no_grad():
                replace_batch_norm(backbone)
        self.model = backbone
        self.intermediate_channel_sizes = self.model.channels
        self.encoder_input_proj = nn.ModuleList(
            [
                Deimv2ConvNormLayer(config, in_channel, config.encoder_hidden_dim, 1, 1)
                if config.encoder_type != "lite"
                else nn.Identity()
                for in_channel in self.intermediate_channel_sizes
            ]
        )

        self.post_init()

    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> list[torch.Tensor]:
        features = self.model(pixel_values, **kwargs).feature_maps
        return [proj(feat) for proj, feat in zip(self.encoder_input_proj, features)]


class Deimv2DINOv3ConvEncoder(Deimv2PreTrainedModel):
    def __init__(self, config: Deimv2Config):
        super().__init__(config)
        self.backbone = load_backbone(config)

        self.spatial_tuning_adapter = Deimv2SpatialTuningAdapter(config)

        embed_dim = config.backbone_config.hidden_size
        hidden_dim = config.encoder_hidden_dim
        spatial_tuning_adapter_channels = config.spatial_tuning_adapter_inplanes
        self.fusion_proj = nn.ModuleList(
            [
                Deimv2ConvNormLayer(config, embed_dim + spatial_tuning_adapter_channels * 2, hidden_dim, 1, 1),
                Deimv2ConvNormLayer(config, embed_dim + spatial_tuning_adapter_channels * 4, hidden_dim, 1, 1),
                Deimv2ConvNormLayer(config, embed_dim + spatial_tuning_adapter_channels * 4, hidden_dim, 1, 1),
            ]
        )

        self.post_init()

    def forward(self, pixel_values: torch.Tensor, **kwargs: Unpack[TransformersKwargs]) -> list[torch.Tensor]:
        backbone_output = self.backbone(pixel_values, **kwargs)
        feature_maps = backbone_output.feature_maps

        patch_size = self.backbone.config.patch_size
        height_patches = pixel_values.shape[2] // patch_size
        width_patches = pixel_values.shape[3] // patch_size

        semantic_features = []
        num_scales = len(feature_maps)
        for i, feat in enumerate(feature_maps):
            resize_height = int(height_patches * 2 ** (num_scales - 2 - i))
            resize_width = int(width_patches * 2 ** (num_scales - 2 - i))
            spatial = F.interpolate(feat, size=[resize_height, resize_width], mode="bilinear", align_corners=False)
            semantic_features.append(spatial)

        detail_features = self.spatial_tuning_adapter(pixel_values)

        outputs = []
        for i, (semantic_feature, detail_feature) in enumerate(zip(semantic_features, detail_features)):
            fused = torch.cat([semantic_feature.to(detail_feature.device), detail_feature], dim=1)
            outputs.append(self.fusion_proj[i](fused))

        return outputs


class Deimv2LiteEncoder(Deimv2PreTrainedModel):
    _can_record_outputs = {
        "hidden_states": [
            OutputRecorder(Deimv2ConvNormLayer, layer_name="input_proj"),
            OutputRecorder(Deimv2ConvNormLayer, layer_name="bi_fusion_conv"),
        ],
    }

    def __init__(self, config: Deimv2Config):
        super().__init__(config)
        hidden_dim = config.encoder_hidden_dim
        activation = config.activation_function

        self.input_proj = nn.ModuleList(
            [Deimv2ConvNormLayer(config, in_channel, hidden_dim, 1, 1) for in_channel in config.encoder_in_channels]
        )

        self.down_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv1 = Deimv2ConvNormLayer(config, hidden_dim, hidden_dim, 1, 1, activation=activation)
        self.down_pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.down_conv2 = Deimv2ConvNormLayer(config, hidden_dim, hidden_dim, 1, 1, activation=activation)

        self.bi_fusion_conv = Deimv2ConvNormLayer(config, hidden_dim, hidden_dim, 1, 1, activation=activation)

        num_blocks = round(3 * config.depth_mult)
        self.fpn_block = Deimv2RepNCSPELAN5(config, numb_blocks=num_blocks)
        self.pan_block = Deimv2RepNCSPELAN5(config, numb_blocks=num_blocks)

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(self, inputs_embeds: list[torch.Tensor], **kwargs: Unpack[TransformersKwargs]) -> Deimv2EncoderOutput:
        projected_features = [self.input_proj[i](feature) for i, feature in enumerate(inputs_embeds)]
        projected_features.append(self.down_conv1(self.down_pool1(projected_features[-1])))

        projected_features[-1] = self.bi_fusion_conv(
            projected_features[-1] + F.adaptive_avg_pool2d(projected_features[-1], 1)
        )

        outputs = []
        fused_feature = projected_features[0] + F.interpolate(projected_features[1], scale_factor=2.0, mode="nearest")
        outputs.append(self.fpn_block(fused_feature))

        fused_feature = projected_features[1] + self.down_conv2(self.down_pool2(outputs[-1]))
        outputs.append(self.pan_block(fused_feature))

        return Deimv2EncoderOutput(feature_maps=outputs)


class Deimv2HybridEncoder(DFineHybridEncoder):

    def __init__(self, config: Deimv2Config):
        Deimv2PreTrainedModel.__init__(self, config)
        self.config = config
        self.in_channels = config.encoder_in_channels
        self.num_fpn_stages = len(self.in_channels) - 1
        self.feat_strides = config.feat_strides
        self.encoder_hidden_dim = config.encoder_hidden_dim
        self.encode_proj_layers = config.encode_proj_layers
        self.positional_encoding_temperature = config.positional_encoding_temperature
        self.eval_size = config.eval_size
        self.out_channels = [self.encoder_hidden_dim for _ in self.in_channels]
        self.out_strides = self.feat_strides
        self.fuse_op = config.encoder_fuse_op

        self.aifi = nn.ModuleList([Deimv2AIFILayer(config) for _ in range(len(self.encode_proj_layers))])

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1, 0, -1):
            self.lateral_convs.append(
                Deimv2ConvNormLayer(config, self.encoder_hidden_dim, self.encoder_hidden_dim, 1, 1)
            )
            num_blocks = round(3 * config.depth_mult)
            self.fpn_blocks.append(Deimv2RepNCSPELAN5(config, numb_blocks=num_blocks))

        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(self.in_channels) - 1):
            self.downsample_convs.append(Deimv2SCDown(config, 3, 2))
            num_blocks = round(3 * config.depth_mult)
            self.pan_blocks.append(Deimv2RepNCSPELAN5(config, numb_blocks=num_blocks))

        self.post_init()

    def forward(
        self,
        inputs_embeds: list[torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Deimv2EncoderOutput:
        r"""
        Args:
            inputs_embeds (`list[torch.FloatTensor]`):
                Multi-scale feature maps from the backbone (one tensor per feature level) passed to the encoder.
        """
        feature_maps = inputs_embeds

        if self.config.encoder_layers > 0:
            for i, enc_ind in enumerate(self.encode_proj_layers):
                feature_maps[enc_ind] = self.aifi[i](feature_maps[enc_ind], **kwargs)

        fpn_feature_maps = [feature_maps[-1]]
        for idx, (lateral_conv, fpn_block) in enumerate(zip(self.lateral_convs, self.fpn_blocks)):
            backbone_feature_map = feature_maps[self.num_fpn_stages - idx - 1]
            top_fpn_feature_map = fpn_feature_maps[-1]
            top_fpn_feature_map = lateral_conv(top_fpn_feature_map)
            fpn_feature_maps[-1] = top_fpn_feature_map
            top_fpn_feature_map = F.interpolate(top_fpn_feature_map, scale_factor=2.0, mode="nearest")
            fused_feature_map = fuse_feature_maps(top_fpn_feature_map, backbone_feature_map, self.fuse_op)
            new_fpn_feature_map = fpn_block(fused_feature_map)
            fpn_feature_maps.append(new_fpn_feature_map)

        fpn_feature_maps.reverse()

        pan_feature_maps = [fpn_feature_maps[0]]
        for idx, (downsample_conv, pan_block) in enumerate(zip(self.downsample_convs, self.pan_blocks)):
            top_pan_feature_map = pan_feature_maps[-1]
            fpn_feature_map = fpn_feature_maps[idx + 1]
            downsampled_feature_map = downsample_conv(top_pan_feature_map)
            fused_feature_map = fuse_feature_maps(downsampled_feature_map, fpn_feature_map, self.fuse_op)
            new_pan_feature_map = pan_block(fused_feature_map)
            pan_feature_maps.append(new_pan_feature_map)

        return Deimv2EncoderOutput(feature_maps=pan_feature_maps)


class Deimv2Decoder(DFineDecoder):
    def __init__(self, config: Deimv2Config):
        super().__init__(config=config)
        self.query_pos_head = Deimv2MLP(4, config.d_model, config.d_model, 3, config.decoder_activation_function)


class Deimv2Model(DFineModel):
    def __init__(self, config: Deimv2Config):
        Deimv2PreTrainedModel.__init__(self, config)

        is_dinov3 = getattr(config.backbone_config, "model_type", None) == "dinov3_vit"
        self.conv_encoder = Deimv2DINOv3ConvEncoder(config) if is_dinov3 else Deimv2ConvEncoder(config)
        self.encoder = (
            Deimv2LiteEncoder(config) if config.encoder_type == "lite" else Deimv2HybridEncoder(config=config)
        )

        if config.num_denoising > 0:
            self.denoising_class_embed = nn.Embedding(
                config.num_labels + 1, config.d_model, padding_idx=config.num_labels
            )

        if config.learn_initial_query:
            self.weight_embedding = nn.Embedding(config.num_queries, config.d_model)

        self.enc_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps),
        )
        self.enc_score_head = nn.Linear(config.d_model, config.num_labels)
        self.enc_bbox_head = Deimv2MLP(config.d_model, config.d_model, 4, 3)

        if config.anchor_image_size:
            self.anchors, self.valid_mask = self.generate_anchors(dtype=self.dtype)

        num_backbone_outs = len(config.decoder_in_channels)
        decoder_input_proj = []
        in_channels = config.decoder_in_channels[-1]
        for _ in range(num_backbone_outs):
            decoder_input_proj.append(
                nn.Identity()
                if config.hidden_size == config.decoder_in_channels[-1]
                else Deimv2ConvNormLayer(config, in_channels, config.d_model, 1, 1)
            )
        for _ in range(config.num_feature_levels - num_backbone_outs):
            decoder_input_proj.append(
                nn.Identity()
                if config.hidden_size == config.decoder_in_channels[-1]
                else Deimv2ConvNormLayer(config, in_channels, config.d_model, 3, 2)
            )
        self.decoder_input_proj = nn.ModuleList(decoder_input_proj)
        self.decoder = Deimv2Decoder(config)

        self.post_init()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_mask: torch.LongTensor | None = None,
        encoder_outputs: torch.FloatTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: list[dict] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if pixel_values is None and inputs_embeds is None:
            raise ValueError("You have to specify either pixel_values or inputs_embeds")

        if inputs_embeds is None:
            batch_size, num_channels, height, width = pixel_values.shape
            device = pixel_values.device
            if pixel_mask is None:
                pixel_mask = torch.ones(((batch_size, height, width)), device=device)

            proj_feats = self.conv_encoder(pixel_values)
        else:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
            proj_feats = inputs_embeds

        encoder_outputs = self.encoder(
            proj_feats,
            **kwargs,
        )

        sources = []
        for level, source in enumerate(encoder_outputs.feature_maps):
            sources.append(self.decoder_input_proj[level](source))

        if self.config.num_feature_levels > len(sources):
            sources.append(self.decoder_input_proj[len(sources)](encoder_outputs.feature_maps[-1]))
            for i in range(len(sources), self.config.num_feature_levels):
                sources.append(self.decoder_input_proj[i](encoder_outputs.feature_maps[-1]))

        source_flatten = []
        spatial_shapes_list = []
        spatial_shapes = torch.empty((len(sources), 2), device=device, dtype=torch.long)
        for level, source in enumerate(sources):
            height, width = source.shape[-2:]
            spatial_shapes[level, 0] = height
            spatial_shapes[level, 1] = width
            spatial_shapes_list.append((height, width))
            source = source.flatten(2).transpose(1, 2)
            source_flatten.append(source)
        source_flatten = torch.cat(source_flatten, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        if self.training and self.config.num_denoising > 0 and labels is not None:
            (
                denoising_class,
                denoising_bbox_unact,
                attention_mask,
                denoising_meta_values,
            ) = get_contrastive_denoising_training_group(
                targets=labels,
                num_classes=self.config.num_labels,
                num_queries=self.config.num_queries,
                class_embed=self.denoising_class_embed,
                num_denoising_queries=self.config.num_denoising,
                label_noise_ratio=self.config.label_noise_ratio,
                box_noise_scale=self.config.box_noise_scale,
            )
        else:
            denoising_class, denoising_bbox_unact, attention_mask, denoising_meta_values = None, None, None, None

        batch_size = len(source_flatten)
        device = source_flatten.device
        dtype = source_flatten.dtype

        if self.training or self.config.anchor_image_size is None:
            spatial_shapes_tuple = tuple(spatial_shapes_list)
            anchors, valid_mask = self.generate_anchors(spatial_shapes_tuple, device=device, dtype=dtype)
        else:
            anchors, valid_mask = self.anchors, self.valid_mask
            anchors, valid_mask = anchors.to(device, dtype), valid_mask.to(device, dtype)

        memory = valid_mask.to(source_flatten.dtype) * source_flatten

        output_memory = self.enc_output(memory)

        enc_outputs_class = self.enc_score_head(output_memory)
        enc_outputs_coord_logits = self.enc_bbox_head(output_memory) + anchors

        _, topk_ind = torch.topk(enc_outputs_class.max(-1).values, self.config.num_queries, dim=1)

        reference_points_unact = enc_outputs_coord_logits.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_coord_logits.shape[-1])
        )

        enc_topk_bboxes = F.sigmoid(reference_points_unact)
        if denoising_bbox_unact is not None:
            reference_points_unact = torch.concat([denoising_bbox_unact, reference_points_unact], 1)

        enc_topk_logits = enc_outputs_class.gather(
            dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, enc_outputs_class.shape[-1])
        )

        if self.config.learn_initial_query:
            target = self.weight_embedding.tile([batch_size, 1, 1])
        else:
            target = output_memory.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1]))
            target = target.detach()

        if denoising_class is not None:
            target = torch.concat([denoising_class, target], 1)

        init_reference_points = reference_points_unact.detach()

        decoder_outputs = self.decoder(
            inputs_embeds=target,
            encoder_hidden_states=source_flatten,
            encoder_attention_mask=attention_mask,
            reference_points=init_reference_points,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            **kwargs,
        )

        return Deimv2ModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_logits=decoder_outputs.intermediate_logits,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            intermediate_predicted_corners=decoder_outputs.intermediate_predicted_corners,
            initial_reference_points=decoder_outputs.initial_reference_points,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.feature_maps,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            init_reference_points=init_reference_points,
            enc_topk_logits=enc_topk_logits,
            enc_topk_bboxes=enc_topk_bboxes,
            enc_outputs_class=enc_outputs_class,
            enc_outputs_coord_logits=enc_outputs_coord_logits,
            denoising_meta_values=denoising_meta_values,
        )


class Deimv2ForObjectDetection(DFineForObjectDetection):
    _no_split_modules = None  # Restrictions are collected from self.model during post_init.

    @property
    def _tied_weights_keys(self):
        pass

    def __init__(self, config: Deimv2Config):
        Deimv2PreTrainedModel.__init__(self, config)

        self.eval_idx = config.eval_idx if config.eval_idx >= 0 else config.decoder_layers + config.eval_idx
        self.model = Deimv2Model(config)
        scaled_dim = round(config.layer_scale * config.hidden_size)
        num_pred = config.decoder_layers
        self.class_embed = nn.ModuleList([nn.Linear(config.d_model, config.num_labels) for _ in range(num_pred)])
        if config.share_bbox_head:
            shared_bbox = Deimv2MLP(config.hidden_size, config.hidden_size, 4 * (config.max_num_bins + 1), 3)
            self.bbox_embed = nn.ModuleList([shared_bbox] * num_pred)
        else:
            self.bbox_embed = nn.ModuleList(
                [
                    Deimv2MLP(config.hidden_size, config.hidden_size, 4 * (config.max_num_bins + 1), 3)
                    for _ in range(self.eval_idx + 1)
                ]
                + [
                    Deimv2MLP(scaled_dim, scaled_dim, 4 * (config.max_num_bins + 1), 3)
                    for _ in range(config.decoder_layers - self.eval_idx - 1)
                ]
            )

        self.model.decoder.class_embed = self.class_embed
        self.model.decoder.bbox_embed = self.bbox_embed
        self.post_init()

    def forward(**super_kwargs):
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers.image_utils import load_image
        >>> from transformers import AutoImageProcessor, Deimv2ForObjectDetection

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = load_image(url)

        >>> image_processor = AutoImageProcessor.from_pretrained("harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers")
        >>> model = Deimv2ForObjectDetection.from_pretrained("harshaljanjani/DEIMv2_HGNetv2_N_COCO_Transformers")

        >>> # prepare image for the model
        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**inputs)

        >>> logits = outputs.logits
        >>> list(logits.shape)
        [1, 300, 80]

        >>> boxes = outputs.pred_boxes
        >>> list(boxes.shape)
        [1, 300, 4]

        >>> # convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
        >>> target_sizes = torch.tensor([image.size[::-1]])
        >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)
        >>> result = results[0]  # first image in batch

        >>> for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        ...     box = [round(i, 2) for i in box.tolist()]
        ...     print(
        ...         f"Detected {model.config.id2label[label.item()]} with confidence "
        ...         f"{round(score.item(), 3)} at location {box}"
        ...     )
        ```
        """
        super().forward(**super_kwargs)


__all__ = [
    "Deimv2Config",
    "Deimv2Model",
    "Deimv2PreTrainedModel",
    "Deimv2ForObjectDetection",
]
