
from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring


class TrOCRProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
class TrOCRProcessor(ProcessorMixin):
    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        **kwargs: Unpack[TrOCRProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None:
            raise ValueError("You need to specify either an `images` or `text` input to process.")

        output_kwargs = self._merge_kwargs(
            TrOCRProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            inputs = self.image_processor(images, **output_kwargs["images_kwargs"])
        if text is not None:
            encodings = self.tokenizer(text, **output_kwargs["text_kwargs"])

        if text is None:
            return inputs
        elif images is None:
            return encodings
        else:
            inputs["labels"] = encodings["input_ids"]
            return inputs

    @property
    def model_input_names(self):
        pass


__all__ = ["TrOCRProcessor"]
