
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from typing_extensions import Unpack

from ... import initialization as init
from ...activations import ACT2FN
from ...backbone_utils import load_backbone
from ...modeling_outputs import DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple
from .configuration_tipsv2_dpt import Tipsv2DptConfig


@dataclass
class Tipsv2DptDensePredictorOutput(ModelOutput):

    predicted_depth: torch.FloatTensor | None = None
    normals: torch.FloatTensor | None = None
    segmentation_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@auto_docstring(
    custom_intro="""
    Class for outputs of normal estimation models.
    """
)
@dataclass
class Tipsv2DptNormalEstimatorOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    normals: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class Tipsv2DptReadoutProjectLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_dim, out_dim), activation])

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


def _get_backbone_hidden_size(config):
    if config.backbone_config is not None and hasattr(config.backbone_config, "hidden_size"):
        return config.backbone_config.hidden_size
    else:
        return config.hidden_size


class Tipsv2DptReassembleLayer(nn.Module):
    def __init__(self, config: Tipsv2DptConfig, channels: int, factor: int):
        super().__init__()
        hidden_size = _get_backbone_hidden_size(config)
        self.projection = nn.Conv2d(in_channels=hidden_size, out_channels=channels, kernel_size=1)

        if factor > 1:
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            self.resize = nn.Identity()
        elif factor < 1:
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

    def forward(self, hidden_state):
        hidden_state = self.projection(hidden_state)
        hidden_state = self.resize(hidden_state)
        return hidden_state


class Tipsv2DptReassembleStage(nn.Module):

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__()
        self.num_register_tokens = config.backbone_config.num_register_tokens
        self.readout_projects = nn.ModuleList(
            [
                Tipsv2DptReadoutProjectLayer(
                    in_dim=2 * config.backbone_config.hidden_size,
                    out_dim=config.backbone_config.hidden_size,
                    activation=ACT2FN[config.readout_activation],
                )
                for _ in config.neck_hidden_sizes
            ]
        )
        self.layers = nn.ModuleList(
            [
                Tipsv2DptReassembleLayer(config, channels=channels, factor=factor)
                for channels, factor in zip(config.neck_hidden_sizes, config.reassemble_factors)
            ]
        )

    def forward(
        self,
        hidden_states: list[torch.Tensor],
        patch_height: int,
        patch_width: int,
    ) -> list[torch.Tensor]:
        """
        Args:
            hidden_states (`list[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []
        for stage_idx, hidden_state in enumerate(hidden_states):
            cls_token = hidden_state[:, 0]
            patch_tokens = hidden_state[:, 1 + self.num_register_tokens :]
            batch_size, num_patches, hidden_size = patch_tokens.shape

            readout = cls_token.unsqueeze(1).expand(-1, num_patches, -1)
            patch_tokens = self.readout_projects[stage_idx](torch.cat([patch_tokens, readout], dim=-1))

            patch_tokens = patch_tokens.reshape(batch_size, patch_height, patch_width, hidden_size)
            patch_tokens = patch_tokens.permute(0, 3, 1, 2).contiguous()

            patch_tokens = self.layers[stage_idx](patch_tokens)
            out.append(patch_tokens)

        return out


class Tipsv2DptPreActResidualLayer(nn.Module):

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__()

        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)

        return hidden_state + residual


class Tipsv2DptFeatureFusionLayer(nn.Module):

    def __init__(self, config: Tipsv2DptConfig, align_corners: bool = True, has_residual: bool = True):
        super().__init__()

        self.align_corners = align_corners

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)
        self.residual_layer1 = Tipsv2DptPreActResidualLayer(config) if has_residual else nn.Identity()
        self.residual_layer2 = Tipsv2DptPreActResidualLayer(config)

    def forward(self, hidden_state: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state


class Tipsv2DptFeatureFusionStage(nn.Module):
    def __init__(self, config: Tipsv2DptConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                Tipsv2DptFeatureFusionLayer(config, has_residual=(idx > 0))
                for idx in range(len(config.neck_hidden_sizes))
            ]
        )

    def forward(self, hidden_states):
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        fused_hidden_state = None
        for hidden_state, layer in zip(hidden_states, self.layers):
            if fused_hidden_state is None:
                fused_hidden_state = layer(hidden_state)
            else:
                fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class Tipsv2DptNeck(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.reassemble_stage = Tipsv2DptReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        self.fusion_stage = Tipsv2DptFeatureFusionStage(config)

    def forward(self, hidden_states: list[torch.Tensor], patch_height=None, patch_width=None) -> list[torch.Tensor]:
        """
        Args:
            hidden_states (`list[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise TypeError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        output = self.fusion_stage(features)

        return output


class Tipsv2DptDecoder(nn.Module):
    def __init__(self, config: Tipsv2DptConfig, out_channels: int, activation: str | None = None):
        super().__init__()
        self.project = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, padding=1)
        self.activation = ACT2FN[activation] if activation is not None else nn.Identity()
        self.head = nn.Linear(config.fusion_hidden_size, out_channels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.project(hidden_state)
        hidden_state = self.activation(hidden_state)
        hidden_state = hidden_state.permute(0, 2, 3, 1)
        hidden_state = self.head(hidden_state)
        hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
        return hidden_state


class Tipsv2DptFeaturesToDepth(nn.Module):

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__()
        self.min_depth = config.min_depth
        self.max_depth = config.max_depth
        self.activation = nn.ReLU()
        bin_centers = torch.linspace(config.min_depth, config.max_depth, config.num_depth_bins)
        self.register_buffer("bin_centers", bin_centers, persistent=False)

    def forward(self, depth_logits: torch.Tensor) -> torch.Tensor:
        probs = self.activation(depth_logits) + self.min_depth
        probs = probs / probs.sum(dim=1, keepdim=True)
        bin_centers = self.bin_centers.to(dtype=depth_logits.dtype)
        return probs.permute(0, 2, 3, 1) @ bin_centers


@auto_docstring
class Tipsv2DptPreTrainedModel(PreTrainedModel):
    config: Tipsv2DptConfig
    base_model_prefix = "backbone"
    main_input_name = "pixel_values"
    input_modalities = ["image"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module) -> None:
        super()._init_weights(module)
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, Tipsv2DptFeaturesToDepth):
            bin_centers = torch.linspace(module.min_depth, module.max_depth, module.bin_centers.shape[0])
            init.copy_(module.bin_centers, bin_centers)


@auto_docstring(
    custom_intro="""
    TIPSv2-DPT Model with three independent heads for depth estimation, surface normal estimation,
    and semantic segmentation — running a single shared backbone forward pass.
    """
)
class Tipsv2DptForDensePrediction(Tipsv2DptPreTrainedModel):
    def __init__(self, config: Tipsv2DptConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.depth_neck = Tipsv2DptNeck(config)
        self.depth_decoder = Tipsv2DptDecoder(
            config, out_channels=config.num_depth_bins, activation=config.depth_decoder_activation
        )
        self.depth_bin_regressor = Tipsv2DptFeaturesToDepth(config)
        self.normals_neck = Tipsv2DptNeck(config)
        self.normals_decoder = Tipsv2DptDecoder(config, out_channels=3)
        self.segmentation_neck = Tipsv2DptNeck(config)
        self.segmentation_decoder = Tipsv2DptDecoder(config, out_channels=config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tipsv2DptDensePredictorOutput:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import Tipsv2DptForDensePrediction, AutoImageProcessor
        >>> from transformers.image_utils import load_image

        >>> model_id = "google/tipsv2-b14-dpt"
        >>> model = Tipsv2DptForDensePrediction.from_pretrained(model_id, device_map="auto")
        >>> image_processor = AutoImageProcessor.from_pretrained(model_id)

        >>> image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/room.jpg")
        >>> inputs = image_processor(images=image, return_tensors="pt").to(model.device)

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> # outputs.predicted_depth: (batch_size, height, width) tensor with predicted depth in meters
        >>> # outputs.normals: (batch_size, 3, height, width) tensor with normals in XYZ format (unnormalized)
        >>> # outputs.segmentation_logits: (batch_size, config.num_labels, height, width) tensor with segmentation logits
        >>> depth_results = image_processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        >>> normal_results = image_processor.post_process_normal_estimation(outputs, target_sizes=[(image.height, image.width)])
        >>> segmentation_results = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])

        >>> predicted_depth = depth_results[0]["predicted_depth"]  # (height, width) tensor with predicted depth in meters
        >>> normals = normal_results[0]["normals"]  # (3, height, width) tensor with normals in XYZ format (L2-normalized)
        >>> segmentation = segmentation_results[0]  # (height, width) tensor with class ids
        ```"""
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, **kwargs)
        feature_maps = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.backbone_config.patch_size
        patch_size_height = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_width = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_height
        patch_width = width // patch_size_width

        depth_fused = self.depth_neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        depth_logits = self.depth_decoder(depth_fused[-1])
        predicted_depth = self.depth_bin_regressor(depth_logits)

        normals_fused = self.normals_neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        normals = self.normals_decoder(normals_fused[-1])

        segmentation_feature_maps_fused = self.segmentation_neck(
            feature_maps, patch_height=patch_height, patch_width=patch_width
        )
        segmentation_logits = self.segmentation_decoder(segmentation_feature_maps_fused[-1])

        return Tipsv2DptDensePredictorOutput(
            predicted_depth=predicted_depth,
            normals=normals,
            segmentation_logits=segmentation_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    TIPSv2-DPT Model with a monocular depth estimation head.
    """
)
class Tipsv2DptForDepthEstimation(Tipsv2DptPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = {"normals_head", "segmentation_head"}

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = Tipsv2DptNeck(config)
        self.decoder = Tipsv2DptDecoder(
            config, out_channels=config.num_depth_bins, activation=config.depth_decoder_activation
        )
        self.bin_regressor = Tipsv2DptFeaturesToDepth(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> DepthEstimatorOutput:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoModelForDepthEstimation, AutoImageProcessor
        >>> from transformers.image_utils import load_image

        >>> model_id = "google/tipsv2-b14-dpt"
        >>> model = AutoModelForDepthEstimation.from_pretrained(model_id, device_map="auto")
        >>> image_processor = AutoImageProcessor.from_pretrained(model_id)

        >>> image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/room.jpg")
        >>> inputs = image_processor(images=image, return_tensors="pt").to(model.device)

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> results = image_processor.post_process_depth_estimation(outputs, target_sizes=[(image.height, image.width)])
        >>> predicted_depth = results[0]["predicted_depth"]  # (height, width) tensor with predicted depth in meters
        ```"""
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, **kwargs)
        feature_maps = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.backbone_config.patch_size
        patch_size_height = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_width = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_height
        patch_width = width // patch_size_width

        fused = self.neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        logits = self.decoder(fused[-1])
        predicted_depth = self.bin_regressor(logits)

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        return DepthEstimatorOutput(
            loss=loss,
            predicted_depth=predicted_depth,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    TIPSv2-DPT Model with a surface normal estimation head.
    """
)
class Tipsv2DptForNormalEstimation(Tipsv2DptPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = {"depth_head", "segmentation_head"}

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = Tipsv2DptNeck(config)
        self.decoder = Tipsv2DptDecoder(config, out_channels=3)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tipsv2DptNormalEstimatorOutput:
        r"""
        Example:

        ```python
        >>> import torch
        >>> from transformers import Tipsv2DptForNormalEstimation, AutoImageProcessor
        >>> from transformers.image_utils import load_image

        >>> model_id = "google/tipsv2-b14-dpt"
        >>> model = Tipsv2DptForNormalEstimation.from_pretrained(model_id, device_map="auto")
        >>> image_processor = AutoImageProcessor.from_pretrained(model_id)

        >>> image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/room.jpg")
        >>> inputs = image_processor(images=image, return_tensors="pt").to(model.device)

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> results = image_processor.post_process_normal_estimation(outputs, target_sizes=[(image.height, image.width)])
        >>> normals = results[0]["normals"]  # (3, height, width) tensor with normals in XYZ format (L2-normalized)
        ```"""
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, **kwargs)
        feature_maps = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.backbone_config.patch_size
        patch_size_height = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_width = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_height
        patch_width = width // patch_size_width

        fused = self.neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        normals = self.decoder(fused[-1])  # (B, 3, H', W') — unnormalized

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        return Tipsv2DptNormalEstimatorOutput(
            loss=loss,
            normals=normals,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    TIPSv2-DPT Model with a semantic segmentation head.
    """
)
class Tipsv2DptForSemanticSegmentation(Tipsv2DptPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = {"depth_head", "normals_head"}

    def __init__(self, config: Tipsv2DptConfig):
        super().__init__(config)
        self.backbone = load_backbone(config)
        self.neck = Tipsv2DptNeck(config)
        self.decoder = Tipsv2DptDecoder(config, out_channels=config.num_labels)
        self.post_init()

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SemanticSegmenterOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
        >>> from transformers.image_utils import load_image

        >>> model_id = "google/tipsv2-b14-dpt"
        >>> model = AutoModelForSemanticSegmentation.from_pretrained(model_id, device_map="auto")
        >>> image_processor = AutoImageProcessor.from_pretrained(model_id)

        >>> image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/room.jpg")
        >>> inputs = image_processor(images=image, return_tensors="pt").to(model.device)

        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> results = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[(image.height, image.width)])
        >>> segmentation_map = results[0]  # (height, width) tensor with class ids
        ```
        """
        outputs = self.backbone.forward_with_filtered_kwargs(pixel_values, **kwargs)
        feature_maps = outputs.feature_maps

        _, _, height, width = pixel_values.shape
        patch_size = self.config.backbone_config.patch_size
        patch_size_height = patch_size if isinstance(patch_size, int) else patch_size[0]
        patch_size_width = patch_size if isinstance(patch_size, int) else patch_size[1]
        patch_height = height // patch_size_height
        patch_width = width // patch_size_width

        feature_maps_fused = self.neck(feature_maps, patch_height=patch_height, patch_width=patch_width)
        logits = self.decoder(feature_maps_fused[-1])

        loss = None
        if labels is not None:
            upsampled_logits = nn.functional.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = self.loss_function(
                upsampled_logits,
                labels,
                ignore_index=self.config.semantic_loss_ignore_index,
            )

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "Tipsv2DptPreTrainedModel",
    "Tipsv2DptForDensePrediction",
    "Tipsv2DptForDepthEstimation",
    "Tipsv2DptForNormalEstimation",
    "Tipsv2DptForSemanticSegmentation",
]
