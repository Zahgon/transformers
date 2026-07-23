
from ...processing_utils import ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring
class ClapProcessor(ProcessorMixin):
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)


__all__ = ["ClapProcessor"]
