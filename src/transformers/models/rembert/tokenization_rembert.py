
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.model", "tokenizer_file": "tokenizer.json"}


class RemBertTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
        do_lower_case: bool = False,
        keep_accents: bool = True,
        bos_token: str = "[CLS]",
        eos_token: str = "[SEP]",
        unk_token: str = "<unk>",
        sep_token: str = "[SEP]",
        pad_token: str = "<pad>",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        _spm_precompiled_charsmap: str | None = None,
        add_prefix_space: bool = True,
        remove_space: bool = True,
        **kwargs,
    ):
        self.remove_space = remove_space
        self.do_lower_case = do_lower_case
        self.keep_accents = keep_accents

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
                unk_id=2,
                byte_fallback=False,
            )
        )


        list_normalizers = [
            normalizers.Replace("``", '"'),
            normalizers.Replace("''", '"'),
            normalizers.Replace(Regex(" {2,}"), " "),
        ]
        if not self.keep_accents:
            list_normalizers.append(normalizers.NFKD())
            list_normalizers.append(normalizers.StripAccents())
        if self.do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        if _spm_precompiled_charsmap is not None:
            list_normalizers.extend([normalizers.Precompiled(_spm_precompiled_charsmap)])

        self._tokenizer.normalizer = normalizers.Sequence(list_normalizers)

        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)

        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        super().__init__(
            add_prefix_space=add_prefix_space,
            do_lower_case=do_lower_case,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            remove_space=remove_space,
            **kwargs,
        )

        cls_token_str = str(cls_token)
        sep_token_str = str(sep_token)
        cls_token_id = self.convert_tokens_to_ids(cls_token_str)
        sep_token_id = self.convert_tokens_to_ids(sep_token_str)

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token_str}:0 $A:0 {sep_token_str}:0",
            pair=f"{cls_token_str}:0 $A:0 {sep_token_str}:0 $B:1 {sep_token_str}:1",
            special_tokens=[
                (cls_token_str, cls_token_id),
                (sep_token_str, sep_token_id),
            ],
        )

        super()._post_init()


__all__ = ["RemBertTokenizer"]
