
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "tokenizer_file": "tokenizer.json"}


class XLMRobertaTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
        add_prefix_space: bool = True,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        _spm_precompiled_charsmap: str | None = None,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space

        if vocab is not None:
            self._vocab = vocab
        else:
            self._vocab = [
                (str(bos_token), 0.0),
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
                (str(unk_token), 0.0),
                (str(mask_token), 0.0),
            ]

        self._tokenizer = Tokenizer(Unigram(vocab=self._vocab, unk_id=3, byte_fallback=False))

        if _spm_precompiled_charsmap is not None:
            self._tokenizer.normalizer = normalizers.Precompiled(_spm_precompiled_charsmap)

        prepend_scheme = "always" if add_prefix_space else "never"
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement="▁", prepend_scheme=prepend_scheme),
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
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=[str(bos_token), "$A", str(eos_token)],
            pair=[str(bos_token), "$A", str(eos_token), str(eos_token), "$B", str(eos_token)],
            special_tokens=[
                (str(bos_token), self.bos_token_id),
                (str(eos_token), self.eos_token_id),
            ],
        )


__all__ = ["XLMRobertaTokenizer"]
