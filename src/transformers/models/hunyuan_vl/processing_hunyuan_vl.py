import re

from ...image_utils import ImageInput, make_flat_list_of_images
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class HunYuanVLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "add_special_tokens": False,
            "return_mm_token_type_ids": True,
        },
    }


@auto_docstring
class HunYuanVLProcessor(ProcessorMixin):

    valid_processor_kwargs = HunYuanVLProcessorKwargs

    def __init__(
        self, image_processor=None, tokenizer=None, chat_template=None, cat_extra_token: bool = True, **kwargs
    ):
        r"""
        cat_extra_token (`bool`, *optional*, defaults to `True`):
            Whether to account for the two extra tokens that HunYuanVL inserts around each image span when computing
            the expanded image token sequence.
        """
        self.tokenizer = tokenizer

        for attr in (
            "image_token",
            "image_token_id",
            "image_start_token",
            "image_start_token_id",
            "image_end_token",
            "image_end_token_id",
            "pad_token",
            "pad_token_id",
        ):
            if getattr(tokenizer, attr, None) is None:
                raise ValueError(
                    f"Tokenizer is missing required attribute '{attr}'. "
                    "Add the corresponding mapping to `extra_special_tokens` in `tokenizer_config.json` or set the "
                    "attribute manually before constructing the processor."
                )

        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        self.image_start_token = tokenizer.image_start_token
        self.image_start_token_id = tokenizer.image_start_token_id
        self.image_end_token = tokenizer.image_end_token
        self.image_end_token_id = tokenizer.image_end_token_id
        self.pad_token_id = tokenizer.pad_token_id

        self.cat_extra_token = cat_extra_token
        chat_template = chat_template if chat_template is not None else getattr(tokenizer, "chat_template", None)

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    @staticmethod
    def _has_wrappers(prompt: str, token_start: int, start_token: str, token: str, end_token: str) -> bool:
        pass

    def validate_inputs(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[HunYuanVLProcessorKwargs],
    ):
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    @property
    def model_input_names(self):
        pass


__all__ = ["HunYuanVLProcessor"]
