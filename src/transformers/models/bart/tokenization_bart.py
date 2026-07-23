

from ..roberta.tokenization_roberta import RobertaTokenizer as _RobertaTokenizer


BartTokenizer = _RobertaTokenizer
BartTokenizerFast = _RobertaTokenizer

__all__ = ["BartTokenizer", "BartTokenizerFast"]
