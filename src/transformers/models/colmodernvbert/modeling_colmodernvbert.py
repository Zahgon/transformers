
from dataclasses import dataclass

import torch
from torch import nn

from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring
from ...utils.generic import can_return_tuple
from ..auto.modeling_auto import AutoModel
from .configuration_colmodernvbert import ColModernVBertConfig


@auto_docstring
class ColModernVBertPreTrainedModel(PreTrainedModel):
    config: ColModernVBertConfig
    base_model_prefix = "model"
    input_modalities = ("image", "text")
    _no_split_modules = []
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)


@auto_docstring(
    custom_intro="""
    Base class for ColModernVBert embeddings output.
    """
)
@dataclass
class ColModernVBertForRetrievalOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    embeddings: torch.Tensor | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    image_hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring(
    custom_intro="""
    Following the ColPali approach, ColModernVBert leverages VLMs to construct efficient multi-vector embeddings directly
    from document images (“screenshots”) for document retrieval. The model is trained to maximize the similarity
    between these document embeddings and the corresponding query embeddings, using the late interaction method
    introduced in ColBERT.

    Using ColModernVBert removes the need for potentially complex and brittle layout recognition and OCR pipelines with
    a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

    ColModernVBert is trained on top of ModernVBert, and was introduced in the following paper:
    [*ModernVBERT: Towards Smaller Visual Document Retrievers*](https://arxiv.org/abs/2510.01149).

    ColModernVBert is part of the ColVision model family, which was introduced with ColPali in the following paper:
    [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://huggingface.co/papers/2407.01449).
    """
)
class ColModernVBertForRetrieval(ColModernVBertPreTrainedModel):
    base_model_prefix = "vlm"

    def __init__(self, config: ColModernVBertConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vlm_config.text_config.vocab_size
        self.vlm = AutoModel.from_config(config.vlm_config)

        self.embedding_dim = self.config.embedding_dim
        self.embedding_proj_layer = nn.Linear(
            self.config.vlm_config.text_config.hidden_size,
            self.embedding_dim,
        )

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ColModernVBertForRetrievalOutput:
        vlm_output = self.vlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **kwargs,
        )

        last_hidden_states = vlm_output[0]  # (batch_size, sequence_length, hidden_size)
        proj_dtype = self.embedding_proj_layer.weight.dtype
        embeddings = self.embedding_proj_layer(last_hidden_states.to(proj_dtype))  # (batch_size, sequence_length, dim)

        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=embeddings.dtype, device=embeddings.device)
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return ColModernVBertForRetrievalOutput(
            embeddings=embeddings,
            hidden_states=vlm_output.hidden_states,
            attentions=vlm_output.attentions,
            image_hidden_states=vlm_output.image_hidden_states,
        )


__all__ = ["ColModernVBertForRetrieval", "ColModernVBertPreTrainedModel"]
