
import collections

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class BertTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
    model = WordPiece

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        do_lower_case: bool = True,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        **kwargs,
    ):
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        if vocab is None:
            vocab = {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
            }
        self._vocab = vocab
        self._tokenizer = Tokenizer(WordPiece(self._vocab, unk_token=str(unk_token)))
        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.WordPiece(prefix="##")
        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        cls_token_id = self.cls_token_id if self.cls_token_id is not None else 2
        sep_token_id = self.sep_token_id if self.sep_token_id is not None else 3

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{str(self.cls_token)}:0 $A:0 {str(self.sep_token)}:0",
            pair=f"{str(self.cls_token)}:0 $A:0 {str(self.sep_token)}:0 $B:1 {str(self.sep_token)}:1",
            special_tokens=[
                (str(self.cls_token), cls_token_id),
                (str(self.sep_token), sep_token_id),
            ],
        )


__all__ = ["BertTokenizer"]

BertTokenizerFast = BertTokenizer
