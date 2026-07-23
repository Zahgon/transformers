from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import TensorType, auto_docstring
from ..auto import AutoTokenizer


class AriaImagesKwargs(ImagesKwargs, total=False):

    split_image: bool
    max_image_size: int
    min_image_size: int


class AriaProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: AriaImagesKwargs

    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "max_image_size": 980,
            "split_image": False,
        },
        "return_tensors": TensorType.PYTORCH,
    }


@auto_docstring
class AriaProcessor(ProcessorMixin):
    valid_processor_kwargs = AriaProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer: AutoTokenizer | str = None,
        chat_template: str | None = None,
        size_conversion: dict[float | int, int] | None = None,
    ):
        r"""
        size_conversion (`Dict`, *optional*):
            A dictionary indicating size conversions for images.
        """
        if size_conversion is None:
            size_conversion = {490: 128, 980: 256}
        self.size_conversion = {int(k): v for k, v in size_conversion.items()}

        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id
        if tokenizer is not None and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    @property
    def unused_input_names(self) -> list[str]:
        pass


__all__ = ["AriaProcessor"]
