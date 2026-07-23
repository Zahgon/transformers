

from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring
class ClvpProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)

    @auto_docstring
    def __call__(self, *args, **kwargs):
        raw_speech = kwargs.pop("raw_speech", None)
        if raw_speech is not None:
            logger.warning(
                "Using `raw_speech` keyword argument is deprecated when calling ClvpProcessor, instead use `audio`."
            )
        kwargs["audio"] = raw_speech
        return super().__call__(*args, **kwargs)


__all__ = ["ClvpProcessor"]
