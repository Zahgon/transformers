

import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...masking_utils import create_causal_mask
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from ..clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from ..clip.modeling_clip import (
    CLIPMLP,
    CLIPAttention,
    CLIPForImageClassification,
    CLIPModel,
    CLIPPreTrainedModel,
    CLIPTextEmbeddings,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionEmbeddings,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "facebook/metaclip-2-worldwide-huge-quickgelu"
_CONFIG_FOR_DOC = "MetaClip2Config"


@auto_docstring(checkpoint="facebook/metaclip-2-worldwide-huge-quickgelu")
@strict
class MetaClip2TextConfig(CLIPTextConfig):
    pass


@auto_docstring(checkpoint="facebook/metaclip-2-worldwide-huge-quickgelu")
@strict
class MetaClip2VisionConfig(CLIPVisionConfig):
    pass


@auto_docstring(checkpoint="facebook/metaclip-2-worldwide-huge-quickgelu")
@strict
class MetaClip2Config(CLIPConfig):
    pass


class MetaClip2TextEmbeddings(CLIPTextEmbeddings):
    pass


class MetaClip2VisionEmbeddings(CLIPVisionEmbeddings):
    pass


class MetaClip2Attention(CLIPAttention):
    pass


class MetaClip2MLP(CLIPMLP):
    pass


@auto_docstring
class MetaClip2PreTrainedModel(CLIPPreTrainedModel):
    base_model_prefix = "metaclip_2"

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if hasattr(module, "logit_scale"):
            init.constant_(module.logit_scale, self.config.logit_scale_init_value)


class MetaClip2TextModel(CLIPTextModel):

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MetaClip2TextModel

        >>> model = MetaClip2TextModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
        ```"""
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        attention_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            past_key_values=None,
        )

        kwargs.pop("is_causal", None)
        encoder_outputs: BaseModelOutput = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            is_causal=True,
            **kwargs,
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id).int().argmax(dim=-1),
        ]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MetaClip2TextModelWithProjection(CLIPTextModelWithProjection):

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MetaClip2TextModelWithProjection

        >>> model = MetaClip2TextModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> text_embeds = outputs.text_embeds
        ```"""
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )


class MetaClip2Model(CLIPModel):

    def __init__(self, config: MetaClip2Config):
        super().__init__(config)

        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = MetaClip2TextModel._from_config(text_config)
        self.vision_model = MetaClip2VisionModel._from_config(vision_config)

        self.visual_projection = nn.Linear(self.vision_embed_dim, self.projection_dim, bias=False)
        self.text_projection = nn.Linear(self.text_embed_dim, self.projection_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        return_loss: bool | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.

        Examples:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MetaClip2Model

        >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )

        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_loss=return_loss,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from transformers import AutoTokenizer, MetaClip2Model

        >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
        >>> text_features = model.get_text_features(**inputs)
        ```"""
        return super().get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs,
        )

    @can_return_tuple
    @auto_docstring
    def get_image_features(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | BaseModelOutputWithPooling:
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MetaClip2Model

        >>> model = MetaClip2Model.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> image_features = model.get_image_features(**inputs)
        ```"""
        return super().get_image_features(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=True,
            **kwargs,
        )


class MetaClip2VisionModel(CLIPVisionModel):

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MetaClip2VisionModel

        >>> model = MetaClip2VisionModel.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> last_hidden_state = outputs.last_hidden_state
        >>> pooled_output = outputs.pooler_output  # pooled CLS states
        ```"""
        return super().forward(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )


class MetaClip2VisionModelWithProjection(CLIPVisionModelWithProjection):

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        Examples:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoProcessor, MetaClip2VisionModelWithProjection

        >>> model = MetaClip2VisionModelWithProjection.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")
        >>> processor = AutoProcessor.from_pretrained("facebook/metaclip-2-worldwide-huge-quickgelu")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)
        >>> image_embeds = outputs.image_embeds
        ```"""
        return super().forward(
            pixel_values=pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )


class MetaClip2ForImageClassification(CLIPForImageClassification):
    pass


__all__ = [
    "MetaClip2Config",
    "MetaClip2TextConfig",
    "MetaClip2VisionConfig",
    "MetaClip2Model",
    "MetaClip2PreTrainedModel",
    "MetaClip2TextModel",
    "MetaClip2TextModelWithProjection",
    "MetaClip2VisionModel",
    "MetaClip2VisionModelWithProjection",
    "MetaClip2ForImageClassification",
]
