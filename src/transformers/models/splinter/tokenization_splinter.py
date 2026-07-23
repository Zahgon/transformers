
import collections

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def load_vocab(vocab_file):
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip("\n")
        vocab[token] = index
    return vocab


class SplinterTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
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
        question_token: str = "[QUESTION]",
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        **kwargs,
    ):
        self._vocab = (
            vocab
            if vocab is not None
            else {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
                str(question_token): 5,
                ".": 6,
            }
        )

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
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            question_token=question_token,
            do_lower_case=do_lower_case,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.question_token = question_token
        if self.question_token not in self.all_special_tokens:
            self.add_tokens([self.question_token], special_tokens=True)
        self.update_post_processor()

    @property
    def question_token_id(self):
        pass

    def update_post_processor(self):
        cls = self.cls_token
        sep = self.sep_token
        question = self.question_token
        dot = "."
        cls_token_id = self.cls_token_id
        sep_token_id = self.sep_token_id
        question_token_id = self.question_token_id
        dot_token_id = self.convert_tokens_to_ids(".")

        if cls is None or sep is None:
            return

        if self.padding_side == "right":
            pair = f"{cls}:0 $A:0 {question} {dot} {sep}:0 $B:1 {sep}:1"
        else:
            pair = f"{cls}:0 $A:0 {sep}:0 $B:1 {question} {dot} {sep}:1"

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls}:0 $A:0 {sep}:0",
            pair=pair,
            special_tokens=[
                (cls, cls_token_id),
                (sep, sep_token_id),
                (question, question_token_id),
                (dot, dot_token_id),
            ],
        )


__all__ = ["SplinterTokenizer"]
