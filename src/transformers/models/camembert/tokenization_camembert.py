
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_python import AddedToken
from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


SPIECE_UNDERLINE = "▁"


class CamembertTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    slow_tokenizer_class = None

    def __init__(
        self,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        _spm_precompiled_charsmap=None,
        mask_token="<mask>",
        additional_special_tokens=None,
        add_prefix_space=True,
        vocab_file=None,
        vocab: str | dict | list | None = None,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.add_prefix_space = add_prefix_space

        mask_token = AddedToken(mask_token, lstrip=True, special=True) if isinstance(mask_token, str) else mask_token

        if additional_special_tokens is None:
            additional_special_tokens = ["<s>NOTUSED", "</s>NOTUSED", "<unk>NOTUSED"]

        if vocab is not None:
            self._vocab = vocab
            unk_index = next((i for i, (tok, _) in enumerate(self._vocab) if tok == str(unk_token)), 0)
            self._tokenizer = Tokenizer(Unigram(self._vocab, unk_id=unk_index, byte_fallback=False))
        else:
            self._vocab = [
                ("<s>NOTUSED", 0.0),
                (str(pad_token), 0.0),
                ("</s>NOTUSED", 0.0),
                (str(unk_token), 0.0),
                ("<unk>NOTUSED", -100),
                (str(mask_token), 0.0),
            ]
            self._tokenizer = Tokenizer(Unigram(self._vocab, unk_id=3, byte_fallback=False))

        if _spm_precompiled_charsmap is not None:
            self._tokenizer.normalizer = normalizers.Precompiled(_spm_precompiled_charsmap)

        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement=SPIECE_UNDERLINE, prepend_scheme=prepend_scheme, split=True),
            ]
        )
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            pair=f"{self.bos_token} $A {self.eos_token} {self.eos_token} $B {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )


__all__ = ["CamembertTokenizer"]
