
from ...utils import logging
from ..albert.tokenization_albert import AlbertTokenizer


logger = logging.get_logger(__name__)


class FNetTokenizer(AlbertTokenizer):

    model_input_names = ["input_ids", "token_type_ids"]


FNetTokenizerFast = FNetTokenizer

__all__ = ["FNetTokenizer", "FNetTokenizerFast"]
