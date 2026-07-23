
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


class AlbertTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
        do_lower_case: bool = True,
        keep_accents: bool = False,
        bos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
        unk_token: str = "<unk>",
        sep_token: str = "[SEP]",
        pad_token: str = "<pad>",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        _spm_precompiled_charsmap: str | None = None,
        add_prefix_space: bool = True,
        trim_offsets: bool = True,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space
        self.trim_offsets = trim_offsets
        self.do_lower_case = do_lower_case
        self.keep_accents = keep_accents
        self._spm_precompiled_charsmap = _spm_precompiled_charsmap

        if vocab is not None:
            self._vocab_scores = vocab
        else:
            self._vocab_scores = [
                (str(pad_token), 0.0),
                (str(unk_token), 0.0),
                (str(cls_token), 0.0),
                (str(sep_token), 0.0),
                (str(mask_token), 0.0),
            ]

        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab_scores,
                unk_id=1,
                byte_fallback=False,
            )
        )

        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
        ]
        if not self.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        if _spm_precompiled_charsmap is not None:
            list_normalizers.append(normalizers.Precompiled(_spm_precompiled_charsmap))
        self._tokenizer.normalizer = normalizers.Sequence(list_normalizers)

        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme),
            ]
        )

        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS]:0 $A:0 [SEP]:0",
            pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", self._tokenizer.token_to_id(str(cls_token))),
                ("[SEP]", self._tokenizer.token_to_id(str(sep_token))),
            ],
        )

        super().__init__(
            do_lower_case=self.do_lower_case,
            keep_accents=self.keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            _spm_precompiled_charsmap=_spm_precompiled_charsmap,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )


__all__ = ["AlbertTokenizer"]
