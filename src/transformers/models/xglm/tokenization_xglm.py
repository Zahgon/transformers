
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}


class XGLMTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        _spm_precompiled_charsmap: str | None = None,
        add_prefix_space: bool = True,
        **kwargs,
    ):
        self.num_madeup_words = 7
        madeup_words = [f"<madeupword{i}>" for i in range(self.num_madeup_words)]
        kwargs["additional_special_tokens"] = kwargs.get("additional_special_tokens", []) or []
        kwargs["additional_special_tokens"] += [
            word for word in madeup_words if word not in kwargs["additional_special_tokens"]
        ]

        self.add_prefix_space = add_prefix_space

        if vocab is not None:
            self._vocab = vocab
        else:
            self._vocab = [
                (str(bos_token), 0.0),
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
                (str(unk_token), 0.0),
            ]

        self._tokenizer = Tokenizer(Unigram(vocab=self._vocab, unk_id=3, byte_fallback=False))

        normalizers_ = [normalizers.Replace(Regex(r" {2,}"), " ")]

        if _spm_precompiled_charsmap is not None:
            normalizers_.insert(0, normalizers.Precompiled(_spm_precompiled_charsmap))

        self._tokenizer.normalizer = normalizers.Sequence(normalizers_)

        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme=prepend_scheme)
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.eos_token} $A",
            pair=f"{self.eos_token} $A {self.eos_token} {self.eos_token} $B",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )


__all__ = ["XGLMTokenizer"]
