
import re
from itertools import accumulate
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import AddedToken, BatchEncoding, TextInput
from ...utils import auto_docstring
from ...utils.import_utils import requires


if TYPE_CHECKING:
    from ...tokenization_utils_base import PreTokenizedInput


class ColModernVBertProcessorKwargs(ProcessingKwargs, total=False):
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
class ColModernVBertProcessor(ProcessorMixin):
    valid_processor_kwargs = ColModernVBertProcessorKwargs

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
        self.fake_image_token = AddedToken("<fake_token_around_image>", normalized=False, special=True).content
        self.image_token = AddedToken("<image>", normalized=False, special=True).content
        self.end_of_utterance_token = AddedToken("<end_of_utterance>", normalized=False, special=True).content
        self.global_image_tag = "<global-img>"  # https://github.com/huggingface/transformers/pull/32473/files/8063e5e17362571b693f1db95167f5443a3be1b2#r1734825341
        self.image_seq_len = image_seq_len
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.fake_image_token_id = tokenizer.convert_tokens_to_ids(self.fake_image_token)
        self.global_image_token_id = tokenizer.convert_tokens_to_ids(self.global_image_tag)
        self.row_col_ids = [
            tokenizer.convert_tokens_to_ids(f"<row_{i + 1}_col_{j + 1}>") for i in range(6) for j in range(6)
        ]

        self._regex_to_remove_extra_special_tokens = re.compile(r"(\n?<global-img>\n?|<row_\d+_col_\d+>\n?)+")

        tokens_to_add = {
            "additional_special_tokens": [
                self.fake_image_token,
                self.image_token,
                self.end_of_utterance_token,
            ]
        }
        tokenizer.add_special_tokens(tokens_to_add)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

        super().__init__(image_processor, tokenizer, chat_template=chat_template, **kwargs)

        self.visual_prompt_prefix = visual_prompt_prefix or (
            f"<|begin_of_text|>User:{self.image_token}Describe the image.<end_of_utterance>\nAssistant:"
        )
        self.query_prefix = query_prefix or ""
        self.query_augmentation_token = self.end_of_utterance_token

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]] = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        image_seq_len: int | None = None,
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ) -> BatchEncoding:
        r"""
        image_seq_len (`int`, *optional*):
            The length of the image sequence. If not provided, the default value of self.image_seq_len is used.
            image_seq_len should be equal to int(((image_size // patch_size) ** 2) / (scale_factor**2))
        """
        images, text = self.prepare_inputs_layout(images=images, text=text, **kwargs)
        self.validate_inputs(images=images, text=text, **kwargs)

        output_kwargs = self._merge_kwargs(
            ColModernVBertProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_seq_len = image_seq_len if image_seq_len is not None else self.image_seq_len
        return_text_replacement_offsets = output_kwargs["text_kwargs"].pop("return_text_replacement_offsets", False)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        image_inputs = text_inputs = {}
        if images is not None:
            image_inputs, images_replacements = self._process_images(images, **output_kwargs["images_kwargs"])

            image_inputs.pop("rows", None)
            image_inputs.pop("cols", None)

            if text is not None:
                text, text_replacement_offsets = self.get_text_with_replacements(
                    text, images_replacements=images_replacements
                )
                text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
                if return_text_replacement_offsets:
                    text_inputs["text_replacement_offsets"] = text_replacement_offsets

                batch_image_seq_lengths = []
                for batch_id, text_replacement_offset in enumerate(text_replacement_offsets):
                    image_seq_lens = []
                    for data in text_replacement_offset:
                        start, end = data["new_span"]
                        start_id_pos = text_inputs.char_to_token(batch_id, start)
                        end_id_pos = text_inputs.char_to_token(batch_id, end - 1)
                        image_seq_lens.append(end_id_pos - start_id_pos + 1)
                    batch_image_seq_lengths.append(image_seq_lens)

                if return_mm_token_type_ids:
                    text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(
                        text_inputs["input_ids"], batch_image_seq_lengths
                    )
                self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        elif text is not None:
            text_inputs = self.tokenizer(text=text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        **kwargs: Unpack[ColModernVBertProcessorKwargs],
    ):
        pass

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: Union[TextInput, "PreTokenizedInput", list[TextInput], list["PreTokenizedInput"]] = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        pass

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    def create_mm_token_type_ids(self, input_ids: list, batch_image_seq_lengths: list[int]) -> list[list[int]]:
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

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


__all__ = ["ColModernVBertProcessor"]
