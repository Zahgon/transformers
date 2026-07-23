
from ...models.bert.tokenization_bert import BertTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class DistilBertTokenizer(BertTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, *args, do_lower_case: bool = True, **kwargs):
        """
        Construct a DistilBERT tokenizer (backed by HuggingFace's tokenizers library). Based on WordPiece.

        This tokenizer inherits from [`BertTokenizer`] which contains most of the main methods. Users should refer to
        this superclass for more information regarding those methods.

        Args:
            do_lower_case (`bool`, *optional*, defaults to `True`):
                Whether or not to lowercase the input when tokenizing.
        """
        super().__init__(*args, do_lower_case=do_lower_case, **kwargs)


DistilBertTokenizerFast = DistilBertTokenizer

__all__ = ["DistilBertTokenizer", "DistilBertTokenizerFast"]
