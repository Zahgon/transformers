
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm.model", "tokenizer_file": "tokenizer.json"}


class DebertaV2Tokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    model = Unigram

    def __init__(
        self,
        vocab: str | dict | list | None = None,
        do_lower_case=False,
        split_by_punct=False,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        add_prefix_space=True,
        unk_id=1,
        **kwargs,
    ):
        self.do_lower_case = do_lower_case
        self.split_by_punct = split_by_punct
        self.add_prefix_space = add_prefix_space

        if vocab is None:
            vocab = [
                (str(pad_token), 0.0),
                (str(unk_token), 0.0),
                (str(bos_token), 0.0),
                (str(eos_token), 0.0),
                (str(sep_token), 0.0),
                (str(cls_token), 0.0),
                (str(mask_token), 0.0),
            ]
            unk_id = 1
        elif isinstance(vocab, list):
            unk_id = vocab.index((str(unk_token), 0.0)) if (str(unk_token), 0.0) in vocab else unk_id

        self._vocab = vocab
        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab,
                unk_id=unk_id,
                byte_fallback=False,
            )
        )

        list_normalizers = []
        if do_lower_case:
            list_normalizers.append(normalizers.Lowercase())

        list_normalizers.extend(
            [
                normalizers.Replace(Regex(r"\s{2,}|[\n\r\t]"), " "),
                normalizers.NFC(),
                normalizers.Strip(left=False, right=True),
            ]
        )
        self._tokenizer.normalizer = normalizers.Sequence(list_normalizers)

        list_pretokenizers = []
        if split_by_punct:
            list_pretokenizers.append(pre_tokenizers.Punctuation(behavior="isolated"))

        prepend_scheme = "always" if add_prefix_space else "first"
        list_pretokenizers.append(pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme))

        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(list_pretokenizers)
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_id=unk_id,
            do_lower_case=do_lower_case,
            split_by_punct=split_by_punct,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        cls_token_id = self.cls_token_id if self.cls_token_id is not None else 0
        sep_token_id = self.sep_token_id if self.sep_token_id is not None else 0

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{str(self.cls_token)}:0 $A:0 {str(self.sep_token)}:0",
            pair=f"{str(self.cls_token)}:0 $A:0 {str(self.sep_token)}:0 $B:1 {str(self.sep_token)}:1",
            special_tokens=[
                (str(self.cls_token), cls_token_id),
                (str(self.sep_token), sep_token_id),
            ],
        )


__all__ = ["DebertaV2Tokenizer"]
