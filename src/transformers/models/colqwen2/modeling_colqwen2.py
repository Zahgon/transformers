
from dataclasses import dataclass

from torch import nn

from transformers import AutoModel

from ...cache_utils import Cache
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, auto_docstring, can_return_tuple, is_torch_available
from .configuration_colqwen2 import ColQwen2Config


if is_torch_available():
    import torch


@auto_docstring
class ColQwen2PreTrainedModel(PreTrainedModel):
    config: ColQwen2Config
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
    Base class for ColQwen2 embeddings output.
    """
)
@dataclass
class ColQwen2ForRetrievalOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    embeddings: torch.Tensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None


@auto_docstring(
    custom_intro="""
    Following the ColPali approach, ColQwen2 leverages VLMs to construct efficient multi-vector embeddings directly
    from document images (“screenshots”) for document retrieval. The model is trained to maximize the similarity
    between these document embeddings and the corresponding query embeddings, using the late interaction method
    introduced in ColBERT.

    Using ColQwen2 removes the need for potentially complex and brittle layout recognition and OCR pipelines with
    a single model that can take into account both the textual and visual content (layout, charts, ...) of a document.

    ColQwen2 is part of the ColVision model family, which was introduced with ColPali in the following paper:
    [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://huggingface.co/papers/2407.01449).
    """
)
class ColQwen2ForRetrieval(ColQwen2PreTrainedModel):
    base_model_prefix = "vlm"

    def __init__(self, config: ColQwen2Config):
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
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        labels: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ) -> ColQwen2ForRetrievalOutput:
        r"""
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        """
        if pixel_values is not None and image_grid_thw is not None:
            offsets = image_grid_thw[:, 1] * image_grid_thw[:, 2]  # (batch_size,)
            arange = torch.arange(pixel_values.shape[1], device=offsets.device)  # (max_len,)
            mask = arange.unsqueeze(0) < offsets.unsqueeze(1)  # (batch_size, max_len)
            pixel_values = pixel_values[mask]  # (total_valid_patches, channels, height, width)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if inputs_embeds is None:
            inputs_embeds = self.vlm.get_input_embeddings()(input_ids)

            if pixel_values is not None:
                image_embeds = self.vlm.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True).pooler_output
                image_mask = (input_ids == self.config.vlm_config.image_token_id).unsqueeze(-1)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        vlm_output = self.vlm(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        vlm_hidden_states = vlm_output.hidden_states if output_hidden_states else None

        last_hidden_states = vlm_output[0]  # (batch_size, sequence_length, hidden_size)
        proj_dtype = self.embedding_proj_layer.weight.dtype
        embeddings = self.embedding_proj_layer(last_hidden_states.to(proj_dtype))  # (batch_size, sequence_length, dim)

        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (batch_size, sequence_length, dim)
        if attention_mask is not None:
            embeddings = embeddings * attention_mask.unsqueeze(-1)  # (batch_size, sequence_length, dim)

        return ColQwen2ForRetrievalOutput(
            embeddings=embeddings,
            past_key_values=vlm_output.past_key_values,
            hidden_states=vlm_hidden_states,
            attentions=vlm_output.attentions,
        )


__all__ = ["ColQwen2ForRetrieval", "ColQwen2PreTrainedModel"]
