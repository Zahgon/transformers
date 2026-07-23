
from ...audio_utils import AudioInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class SeamlessM4TTextKwargs(TextKwargs):

    src_lang: str | None
    tgt_lang: str | None


class SeamlessM4TProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: SeamlessM4TTextKwargs
    _defaults = {}


@auto_docstring
class SeamlessM4TProcessor(ProcessorMixin):
    valid_processor_kwargs = SeamlessM4TProcessorKwargs

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        r"""
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **input_features** -- Audio input features to be fed to a model. Returned when `audios` is not `None`.
        """
        if text is not None and audio is not None:
            raise ValueError(
                "Text and audios are mututally exclusive when passed to `SeamlessM4T`. Specify one or another."
            )
        return super().__call__(text=text, audio=audio, **kwargs)


__all__ = ["SeamlessM4TProcessor"]
