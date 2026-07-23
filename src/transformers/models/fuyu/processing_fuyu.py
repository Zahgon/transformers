
import re
from typing import Union

import numpy as np

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging, requires_backends
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch


TEXT_REPR_BBOX_OPEN = "<box>"
TEXT_REPR_BBOX_CLOSE = "</box>"
TEXT_REPR_POINT_OPEN = "<point>"
TEXT_REPR_POINT_CLOSE = "</point>"

TOKEN_BBOX_OPEN_STRING = "<0x00>"  # <bbox>
TOKEN_BBOX_CLOSE_STRING = "<0x01>"  # </bbox>
TOKEN_POINT_OPEN_STRING = "<0x02>"  # <point>
TOKEN_POINT_CLOSE_STRING = "<0x03>"  # </point>
BEGINNING_OF_ANSWER_STRING = "<0x04>"  # <boa>


class FuyuProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_attention_mask": True,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
            "return_mm_token_type_ids": False,
        },
    }


def full_unpacked_stream_to_tensor(
    all_bi_tokens_to_place: list[int],
    full_unpacked_stream: list["torch.Tensor"],
    fill_value: int,
    batch_size: int,
    new_seq_len: int,
    offset: int,
) -> "torch.Tensor":
    pass


def construct_full_unpacked_stream(
    num_real_text_tokens: Union[list[list[int]], "torch.Tensor"],
    input_stream: "torch.Tensor",
    image_tokens: list[list["torch.Tensor"]],
    batch_size: int,
    num_sub_sequences: int,
) -> list["torch.Tensor"]:
    pass


def _replace_string_repr_with_token_tags(prompt: str) -> str:
    pass


def _segment_prompt_into_text_token_conversions(prompt: str) -> list:
    pass


def _transform_coordinates_and_tokenize(prompt: str, scale_factor: float, tokenizer) -> list[int]:
    pass


def _transform_within_tags(text: str, scale_factor: float, tokenizer) -> list[int]:
    pass


def _tokenize_prompts_with_image_and_batch(
    tokenizer,
    prompts: list[list[str]],
    scale_factors: list[list["torch.Tensor"]] | None,
    max_tokens_to_generate: int,
    max_position_embeddings: int,
    add_BOS: bool,  # Same issue with types as above
    add_beginning_of_answer_token: bool,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    pass


def original_to_transformed_h_coords(original_coords, scale_h):
    pass


def original_to_transformed_w_coords(original_coords, scale_w):
    pass


def scale_point_to_transformed_image(x: float, y: float, scale_factor: float) -> list[int]:
    pass


def scale_bbox_to_transformed_image(
    top: float, left: float, bottom: float, right: float, scale_factor: float
) -> list[int]:
    pass


@requires(backends=("vision",))
@auto_docstring
class FuyuProcessor(ProcessorMixin):
    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        """
        Override for BC. Fuyu uses TokenizersBackend and requires token_type_ids to be removed from model_input_names
        because Fuyu uses mm_token_type_ids instead for multimodal token identification.    `
        """
        from ...tokenization_utils_tokenizers import TokenizersBackend

        tokenizer = TokenizersBackend.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if "token_type_ids" in tokenizer.model_input_names:
            tokenizer.model_input_names.remove("token_type_ids")
        return tokenizer

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_tokens_to_generate = 10
        self.max_position_embeddings = 16384  # TODO Can't derive this from model files: where to set it?
        self.pad_token_id = 0
        self.dummy_image_index = -1
        vocab = tokenizer.get_vocab()
        self.image_token_id = vocab["|SPEAKER|"]
        self.image_newline_id = vocab["|NEWLINE|"]

    @property
    def image_token_ids(self) -> list[int]:
        pass

    def _left_pad_inputs_with_attention_mask(self, model_inputs: list[dict], return_attention_mask: bool):
        pass

    def get_sample_encoding(
        self,
        prompts,
        scale_factors,
        image_unpadded_heights,
        image_unpadded_widths,
        image_placeholder_id,
        image_newline_id,
        tensor_batch_images,
    ):
        pass

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: str | list[str] | TextInput | PreTokenizedInput | None = None,
        **kwargs: Unpack[FuyuProcessorKwargs],
    ) -> "BatchFeature":
        r"""
        Returns:
            [`FuyuBatchEncoding`]: A [`FuyuBatchEncoding`] with the following fields:

            - **input_ids** -- Tensor of token ids to be fed to a model. Returned when `text` is not `None`.
            - **image_patches** -- List of Tensor of image patches. Returned when `images` is not `None`.
            - **image_patches_indices** -- Tensor of indices where patch embeddings have to be inserted by the model.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model when
              `return_attention_mask=True`.
        """
        requires_backends(self, ["torch"])

        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be None.")

        output_kwargs = self._merge_kwargs(
            FuyuProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)

        if not output_kwargs["text_kwargs"].setdefault("return_attention_mask", True):
            raise ValueError("`return_attention_mask=False` is not supported for this model.")

        if text is not None and images is None:
            logger.warning("You are processing a text with no associated image. Make sure it is intended.")
            text_encoding = self.tokenizer(text, **output_kwargs["text_kwargs"])
            return text_encoding

        if text is None and images is not None:
            logger.warning("You are processing an image with no associated text. Make sure it is intended.")
            prompts = [[""]]
        if text is not None and images is not None:
            if isinstance(text, str):
                prompts = [[text]]
            elif isinstance(text, list):
                prompts = [[text_seq] for text_seq in text]


        output_kwargs["images_kwargs"]["return_tensors"] = "pt"
        image_encoding = self.image_processor.preprocess(images, **output_kwargs["images_kwargs"])
        batch_images = image_encoding["images"]
        image_unpadded_heights = image_encoding["image_unpadded_heights"]
        image_unpadded_widths = image_encoding["image_unpadded_widths"]
        scale_factors = image_encoding["image_scale_factors"]
        self.subsequence_length = 1  # Each batch contains only one sequence.
        self.batch_size = len(batch_images)

        all_encodings = []

        for prompt, scale_factor, image_unpadded_height, image_unpadded_width, tensor_batch_image in zip(
            prompts, scale_factors, image_unpadded_heights, image_unpadded_widths, batch_images
        ):
            sample_encoding = self.get_sample_encoding(
                prompts=[prompt],
                scale_factors=[scale_factor],
                image_unpadded_heights=torch.tensor([image_unpadded_height]).unsqueeze(0),
                image_unpadded_widths=torch.tensor([image_unpadded_width]).unsqueeze(0),
                image_placeholder_id=self.image_token_id,
                image_newline_id=self.image_newline_id,
                tensor_batch_images=tensor_batch_image.unsqueeze(0),
            )
            all_encodings.append(sample_encoding)

        batch_encoding = self._left_pad_inputs_with_attention_mask(
            model_inputs=all_encodings, return_attention_mask=True
        )
        if return_mm_token_type_ids:
            batch_encoding["mm_token_type_ids"] = self.create_mm_token_type_ids(batch_encoding["input_ids"])
            batch_encoding["mm_token_type_ids"] = torch.tensor(batch_encoding["mm_token_type_ids"])
        return BatchFeature(data=batch_encoding)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    def post_process_box_coordinates(self, outputs, target_sizes=None):
        pass

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-processes the output of `FuyuForConditionalGeneration` to only return the text output.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                containing the token ids of the generated sequences.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text output.
        """
        beginning_of_answer = self.tokenizer.convert_tokens_to_ids(BEGINNING_OF_ANSWER_STRING)
        unpadded_output_sequences = [
            seq[(seq == beginning_of_answer).nonzero(as_tuple=True)[0] + 1 :] for seq in generated_outputs
        ]
        max_len = max(len(seq) for seq in unpadded_output_sequences)
        padded_output_sequences = torch.full((len(unpadded_output_sequences), max_len), self.pad_token_id)
        for i, seq in enumerate(unpadded_output_sequences):
            padded_output_sequences[i, : len(seq)] = torch.tensor(seq)

        return self.batch_decode(padded_output_sequences, skip_special_tokens=skip_special_tokens, **kwargs)

    @property
    def model_input_names(self):
        pass


__all__ = ["FuyuProcessor"]
