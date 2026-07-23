
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

_model_names = [
    "small",
    "small-base",
    "medium",
    "medium-base",
    "intermediate",
    "intermediate-base",
    "large",
    "large-base",
    "xlarge",
    "xlarge-base",
]


class FunnelTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model = WordPiece
    cls_token_type_id: int = 2

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        do_lower_case: bool = True,
        unk_token: str = "<unk>",
        sep_token: str = "<sep>",
        pad_token: str = "<pad>",
        cls_token: str = "<cls>",
        mask_token: str = "<mask>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        clean_text: bool = True,
        tokenize_chinese_chars: bool = True,
        strip_accents: bool | None = None,
        wordpieces_prefix: str = "##",
        **kwargs,
    ):
        self.do_lower_case = do_lower_case
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents
        self.clean_text = clean_text
        self.wordpieces_prefix = wordpieces_prefix

        self._vocab = (
            vocab
            if vocab is not None
            else {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
                str(bos_token): 5,
                str(eos_token): 6,
            }
        )

        self._tokenizer = Tokenizer(WordPiece(self._vocab, unk_token=str(unk_token)))

        self._tokenizer.normalizer = normalizers.BertNormalizer(
            clean_text=clean_text,
            handle_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            lowercase=do_lower_case,
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        super().__init__(
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            clean_text=clean_text,
            tokenize_chinese_chars=tokenize_chinese_chars,
            strip_accents=strip_accents,
            wordpieces_prefix=wordpieces_prefix,
            **kwargs,
        )
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token}:2 $A:0 {sep_token}:0",  # token_type_id is 2 for Funnel transformer
            pair=f"{cls_token}:2 $A:0 {sep_token}:0 $B:1 {sep_token}:1",
            special_tokens=[
                (str(cls_token), self.cls_token_id),
                (str(sep_token), self.sep_token_id),
            ],
        )


__all__ = ["FunnelTokenizer"]
