
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_python import AddedToken
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


SPIECE_UNDERLINE = "▁"


class BigBirdTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    prefix_tokens: list[int] = []
    model = Unigram

    def __init__(
        self,
        vocab: str | dict | list | None = None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        sep_token="[SEP]",
        mask_token="[MASK]",
        cls_token="[CLS]",
        add_prefix_space=True,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.add_prefix_space = add_prefix_space

        if vocab is None:
            vocab = [(str(pad_token), 0.0), (str(eos_token), 0.0), (str(bos_token), 0.0), (str(unk_token), 0.0)]
            unk_id = 3
        elif isinstance(vocab, list):
            unk_id = vocab.index((str(unk_token), 0.0)) if (str(unk_token), 0.0) in vocab else 100

        self._tokenizer = Tokenizer(Unigram(vocab, unk_id=unk_id, byte_fallback=False))
        self._tokenizer.normalizer = normalizers.Sequence(
            [normalizers.Strip(left=False, right=False), normalizers.Replace(Regex(r" {2,}"), SPIECE_UNDERLINE)]
        )

        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(
            replacement="▁", prepend_scheme=prepend_scheme, split=True
        )
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme, split=True)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            cls_token=cls_token,
            sep_token=sep_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        cls_token_str = str(cls_token)
        sep_token_str = str(sep_token)
        cls_token_id = self.cls_token_id
        sep_token_id = self.sep_token_id

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{cls_token_str}:0 $A:0 {sep_token_str}:0",
            pair=f"{cls_token_str}:0 $A:0 {sep_token_str}:0 $B:1 {sep_token_str}:1",
            special_tokens=[(cls_token_str, cls_token_id), (sep_token_str, sep_token_id)],
        )


__all__ = ["BigBirdTokenizer"]
