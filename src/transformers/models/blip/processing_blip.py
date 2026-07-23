
from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class BlipProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
        },
    }


@auto_docstring
class BlipProcessor(ProcessorMixin):
    valid_processor_kwargs = BlipProcessorKwargs

    def __init__(self, image_processor, tokenizer, **kwargs):
        tokenizer.return_token_type_ids = False
        super().__init__(image_processor, tokenizer)

    @property
    def unused_input_names(self) -> list[str]:
        pass


__all__ = ["BlipProcessor"]
