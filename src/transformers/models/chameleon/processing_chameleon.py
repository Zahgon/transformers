
from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class ChameleonTextKwargs(TextKwargs, total=False):

    return_for_text_completion: bool


class ChameleonProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: ChameleonTextKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_for_text_completion": False,
            "return_mm_token_type_ids": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class ChameleonProcessor(ProcessorMixin):
    valid_processor_kwargs = ChameleonProcessorKwargs

    def __init__(self, image_processor, tokenizer, image_seq_length: int = 1024, image_token: str = "<image>"):
        r"""
        image_seq_length (`int`, *optional*, defaults to 1024):
            Sequence length of one image embedding.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The special token used to indicate image in the text.
        """
        super().__init__(image_processor, tokenizer)

        self.image_seq_length = image_seq_length
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_start_token = (
            tokenizer.boi_token if hasattr(tokenizer, "boi_token") else "<racm3:break>"
        )  # fixed tokens for start and end, so can hardcode
        self.image_end_token = tokenizer.eoi_token if hasattr(tokenizer, "eoi_token") else "<eoss>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.image_start_token_id = tokenizer.convert_tokens_to_ids(self.image_start_token)
        self.image_end_token_id = tokenizer.convert_tokens_to_ids(self.image_end_token)

    @property
    def image_token_ids(self) -> list[int]:
        pass

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[ChameleonProcessorKwargs],
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
        if isinstance(text, str):
            text = [text]

        text = [f"{sample}{self.tokenizer.sep_token}" for sample in text]
        model_inputs = super().__call__(images=images, text=text, **kwargs)
        return model_inputs

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass


__all__ = ["ChameleonProcessor"]
