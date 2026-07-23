
from dataclasses import dataclass
from typing import Optional, Union

import torch
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import Unpack
from ...tokenization_utils_base import TextInput
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, logging
from ...utils.generic import can_return_tuple
from ...utils.import_utils import requires
from ..auto import CONFIG_MAPPING
from ..auto.modeling_auto import AutoModel
from ..colpali.modeling_colpali import ColPaliForRetrieval, ColPaliPreTrainedModel
from ..colqwen2.configuration_colqwen2 import ColQwen2Config
from ..idefics3.processing_idefics3 import Idefics3Processor, Idefics3ProcessorKwargs


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="ModernVBERT/colmodernvbert-merged")
@strict
class ColModernVBertConfig(ColQwen2Config):

    model_type = "colmodernvbert"
    sub_configs = {"vlm_config": PreTrainedConfig}

    vlm_config: dict | PreTrainedConfig | None = None
    embedding_dim: int = 128
    initializer_range: float = 0.02

    def __post_init__(self, **kwargs):
        if self.vlm_config is None:
            self.vlm_config = CONFIG_MAPPING["modernvbert"]()
            logger.info(
                "`vlm_config` is `None`. Initializing `vlm_config` with the `ModernVBertConfig` with default values."
            )
        elif isinstance(self.vlm_config, dict):
            self.vlm_config = CONFIG_MAPPING[self.vlm_config["model_type"]](**self.vlm_config)

        if not hasattr(self.vlm_config, "vocab_size"):
            self.vlm_config.vocab_size = self.vlm_config.get_text_config().vocab_size

        super().__post_init__(**kwargs)


class ColModernVBertProcessorKwargs(Idefics3ProcessorKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": "longest",
        },
        "images_kwargs": {
            "return_row_col_info": True,
            "data_format": "channels_first",
            "do_convert_rgb": True,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@requires(backends=("torch",))
@auto_docstring
class ColModernVBertProcessor(Idefics3Processor):
    def __init__(
        self,
        image_processor,
        tokenizer=None,
        chat_template=None,
        image_seq_len: int = 64,
        visual_prompt_prefix: str | None = None,
        query_prefix: str | None = None,
        **kwargs,
    ):
        r"""
        image_seq_len (`int`, *optional*, defaults to 64):
            The length of the image sequence i.e. the number of <image> tokens per image in the input.
        visual_prompt_prefix (`str`, *optional*):
            A string that gets tokenized and prepended to the image tokens.
        query_prefix (`str`, *optional*):
            A prefix to be used for the query.
        """
        chat_template = None  # ColModernVBert does not use chat templates

        super().__init__(
            image_processor,
            tokenizer,
            chat_template=chat_template,
            image_seq_len=image_seq_len,
            **kwargs,
        )

        self.visual_prompt_prefix = visual_prompt_prefix or (
            f"<|begin_of_text|>User:{self.image_token}Describe the image.<end_of_utterance>\nAssistant:"
        )
        self.query_prefix = query_prefix or ""
        self.query_augmentation_token = self.end_of_utterance_token

    def process_images(
        self,
        images: ImageInput | None = None,
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchFeature:
        pass

    def process_queries(
        self,
        text: TextInput | list[TextInput],
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchFeature:
        pass

    def score_retrieval(
        self,
        query_embeddings: Union["torch.Tensor", list["torch.Tensor"]],
        passage_embeddings: Union["torch.Tensor", list["torch.Tensor"]],
        batch_size: int = 128,
        output_dtype: Optional["torch.dtype"] = None,
        output_device: Union["torch.device", str] = "cpu",
    ) -> "torch.Tensor":
        pass


@auto_docstring
class ColModernVBertPreTrainedModel(ColPaliPreTrainedModel):
    config: ColModernVBertConfig


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
class ColModernVBertForRetrieval(ColPaliForRetrieval):
    def __init__(self, config: ColModernVBertConfig):
        super().__init__(config)
        self.vlm = AutoModel.from_config(config.vlm_config)
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


__all__ = [
    "ColModernVBertConfig",
    "ColModernVBertForRetrieval",
    "ColModernVBertPreTrainedModel",
    "ColModernVBertProcessor",
]
