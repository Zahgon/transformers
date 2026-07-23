
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram

from ...tokenization_python import AddedToken
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


SPIECE_UNDERLINE = "▁"


class BarthezTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        vocab: str | dict | list | None = None,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        add_prefix_space=True,
        **kwargs,
    ):
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        self.add_prefix_space = add_prefix_space

        if vocab is not None:
            self._vocab = vocab
        else:
            self._vocab = [
                (str(pad_token), 0.0),
                (str(unk_token), 0.0),
                (str(cls_token), 0.0),
                (str(sep_token), 0.0),
                (str(mask_token), 0.0),
            ]

        self._tokenizer = Tokenizer(Unigram(self._vocab, unk_id=3, byte_fallback=False))

        self._tokenizer.normalizer = normalizers.Sequence(
            [
                normalizers.Replace(Regex(r"\s{2,}|[\n\r\t]"), " "),
                normalizers.NFC(),
                normalizers.Strip(left=False, right=True),
            ]
        )
        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )


__all__ = ["BarthezTokenizer"]
