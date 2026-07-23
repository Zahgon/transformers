
from tokenizers import AddedToken, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_base import _get_prepend_scheme
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

SPIECE_UNDERLINE = "▁"

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class XLNetTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    padding_side = "left"
    model = Unigram

    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
        unk_id: int = 0,
        do_lower_case=False,
        remove_space=True,
        keep_accents=False,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="<sep>",
        pad_token="<pad>",
        cls_token="<cls>",
        mask_token="<mask>",
        _spm_precompiled_charsmap=None,
        additional_special_tokens=None,
        **kwargs,
    ):
        if additional_special_tokens is None:
            additional_special_tokens = ["<eop>", "<eod>"]

        if vocab is not None:
            self._vocab = vocab
        else:
            self._vocab = [(str(unk_token), 0.0)]

        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab,
                unk_id=unk_id,
                byte_fallback=False,
            )
        )

        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        list_normalizers.append(normalizers.NFKD())
        list_normalizers.append(normalizers.StripAccents())
        if do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        if _spm_precompiled_charsmap is not None:
            list_normalizers.append(normalizers.Precompiled(_spm_precompiled_charsmap))
        self._tokenizer.normalizer = normalizers.Sequence(list_normalizers)

        add_prefix_space = True
        prepend_scheme = _get_prepend_scheme(add_prefix_space, self)
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme),
            ]
        )

        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        self._pad_token_type_id = 3
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        super().__init__(
            unk_id=unk_id,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A:0 {str(self.sep_token)}:0 {str(self.cls_token)}:2",
            pair=f"$A:0 {str(self.sep_token)}:0 $B:1 {str(self.sep_token)}:1 {str(self.cls_token)}:2",
            special_tokens=[
                (str(self.sep_token), self.sep_token_id),
                (str(self.cls_token), self.cls_token_id),
            ],
        )


__all__ = ["XLNetTokenizer"]
