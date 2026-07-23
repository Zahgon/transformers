
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from ...tokenization_python import AddedToken
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt", "tokenizer_file": "tokenizer.json"}


class MPNetTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = WordPiece

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        do_lower_case=True,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="[UNK]",
        pad_token="<pad>",
        mask_token="<mask>",
        tokenize_chinese_chars=True,
        strip_accents=None,
        **kwargs,
    ):
        self._vocab = vocab if vocab is not None else {}

        self._tokenizer = Tokenizer(WordPiece(self._vocab, unk_token=str(unk_token)))

        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=True,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )

        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

        self._tokenizer.decoder = decoders.WordPiece(prefix="##")

        self.do_lower_case = do_lower_case

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        super().__init__(
            do_lower_case=do_lower_case,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            **kwargs,
        )

        cls_str = str(self.cls_token)
        sep_str = str(self.sep_token)
        cls_token_id = self.cls_token_id if self.cls_token_id is not None else 0
        sep_token_id = self.sep_token_id if self.sep_token_id is not None else 2

        self._tokenizer.post_processor = processors.RobertaProcessing(
            sep=(sep_str, sep_token_id),
            cls=(cls_str, cls_token_id),
            trim_offsets=True,
            add_prefix_space=False,
        )

    @property
    def mask_token(self) -> str:
        pass

    @mask_token.setter
    def mask_token(self, value):
        pass


__all__ = ["MPNetTokenizer"]
