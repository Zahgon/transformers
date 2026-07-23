

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, make_nested_list_of_images
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class MllamaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "image_kwargs": {
            "max_image_tiles": 4,
        },
    }


def get_cross_attention_token_mask(input_ids: list[int], image_token_id: int) -> list[list[int]]:
    pass


def convert_sparse_cross_attention_mask_to_dense(
    cross_attention_token_mask: list[list[list[int]]],
    num_tiles: list[list[int]],
    max_num_tiles: int,
    length: int,
) -> np.ndarray:
    pass


def build_string_from_input(prompt: str, bos_token: str, image_token: str) -> str:
    """
    Builds a string from the input prompt by adding `bos_token` if not already present.

    Args:
        prompt (`str`):
            The input prompt string.
        bos_token (`str`):
            The beginning of sentence token to be added.
        image_token (`str`):
            The image token used to identify the start of an image sequence.

    Returns:
        str: The modified prompt string with the `bos_token` added if necessary.

    Examples:
        >>> build_string_from_input("Hello world", "<begin_of_text>", "<|image|>")
        '<begin_of_text>Hello world'

        >>> build_string_from_input("<|image|>Hello world", "<begin_of_text>", "<|image|>")
        '<|image|><begin_of_text>Hello world'

        >>> build_string_from_input("<begin_of_text>Hello world", "<begin_of_text>", "<|image|>")
        '<begin_of_text>Hello world'
    """

    if bos_token in prompt:
        return prompt

    num_image_tokens_on_start = 0
    while prompt.startswith(image_token):
        prompt = prompt[len(image_token) :]
        num_image_tokens_on_start += 1

    return f"{image_token * num_image_tokens_on_start}{bos_token}{prompt}"


@auto_docstring
class MllamaProcessor(ProcessorMixin):
    valid_processor_kwargs = MllamaProcessorKwargs

    def __init__(self, image_processor, tokenizer, chat_template=None):
        if not hasattr(tokenizer, "image_token"):
            self.image_token = "<|image|>"
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        else:
            self.image_token = tokenizer.image_token
            self.image_token_id = tokenizer.image_token_id

        self.python_token = "<|python_tag|>"
        self.python_token_id = tokenizer.convert_tokens_to_ids(self.python_token)
        self.bos_token = tokenizer.bos_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[MllamaProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            TODO: add aspect_ratio_ids and aspect_ratio_mask and cross_attention_mask
        """
        images, text = self.prepare_inputs_layout(images=images, text=text)
        self.validate_inputs(images=images, text=text, **kwargs)

        output_kwargs = self._merge_kwargs(
            MllamaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)

        text_inputs = {}
        if text is not None:
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image"])

        image_inputs = {}
        if images is not None:
            image_inputs, _ = self._process_images(images, **output_kwargs["images_kwargs"])
            num_tiles = image_inputs.pop("num_tiles")

        if images is not None and text is not None:
            cross_attention_token_mask = [
                get_cross_attention_token_mask(token_ids, self.image_token_id)
                for token_ids in text_inputs["input_ids"]
            ]
            cross_attention_mask = convert_sparse_cross_attention_mask_to_dense(
                cross_attention_token_mask,
                num_tiles=num_tiles,
                max_num_tiles=self.image_processor.max_image_tiles,
                length=max(len(input_ids) for input_ids in text_inputs["input_ids"]),
            )
            text_inputs["cross_attention_mask"] = cross_attention_mask

        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs,
    ):
        pass

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        pass

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        pass


__all__ = ["MllamaProcessor"]
