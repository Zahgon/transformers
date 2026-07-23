
import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_vision_available, logging
from ...utils.import_utils import requires


if is_vision_available():
    from .image_processing_pixtral import get_resize_output_image_size


logger = logging.get_logger(__name__)


class PixtralProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


def is_url(val) -> bool:
    pass


def is_image_or_image_url(elem):
    pass


@auto_docstring
@requires(backends=("torchvision", "torch"))
class PixtralProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        chat_template=None,
        image_token="[IMG]",  # set the default and let users change if they have peculiar special tokens in rare cases
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to 16):
            Patch size from the vision tower.
        spatial_merge_size (`int`, *optional*, defaults to 1):
            The downsampling factor for the spatial merge operation.
        image_token (`str`, *optional*, defaults to `"[IMG]"`):
            Special token used to denote image location.
        image_break_token (`str`, *optional*, defaults to `"[IMG_BREAK]"`):
            Special token used to denote the end of a line of pixels in an image.
        image_end_token (`str`, *optional*, defaults to `"[IMG_END]"`):
            Special token used to denote the end of an image input.
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token = image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token = image_break_token
        self.image_end_token = image_end_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_break_token_id = tokenizer.convert_tokens_to_ids(self.image_break_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)

    @property
    def image_token_ids(self) -> list[int]:
        pass

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[PixtralProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
            `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            PixtralProcessorKwargs,
            tokenizer_init_kwargs=getattr(self.tokenizer, "init_kwargs", {}),
            **kwargs,
        )

        patch_size = self.patch_size * self.spatial_merge_size

        if images is not None:
            output_kwargs["images_kwargs"]["patch_size"] = patch_size
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        else:
            image_inputs = {}

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise TypeError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = text
        if image_inputs.get("pixel_values") is not None:
            image_sizes = iter(image_inputs["image_sizes"])
            prompt_strings = []
            replace_strings = []

            for sample in text:
                while self.image_token in sample:
                    height, width = next(image_sizes)
                    num_height_tokens = height // patch_size
                    num_width_tokens = width // patch_size
                    replace_tokens = [
                        [self.image_token] * num_width_tokens + [self.image_break_token]
                    ] * num_height_tokens
                    replace_tokens = [item for sublist in replace_tokens for item in sublist]
                    replace_tokens[-1] = self.image_end_token
                    replace_str = "".join(replace_tokens)
                    replace_strings.append(replace_str)
                    sample = sample.replace(self.image_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    replace_str = replace_strings.pop(0)
                    sample = sample.replace("<placeholder>", replace_str, 1)
                prompt_strings.append(sample)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        output_kwargs["text_kwargs"].pop("return_token_type_ids", None)
        text_inputs = self.tokenizer(prompt_strings, **output_kwargs["text_kwargs"], return_tensors=None)
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**text_inputs, **image_inputs}, tensor_type=return_tensors)

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    @property
    def model_input_names(self):
        pass


__all__ = ["PixtralProcessor"]
