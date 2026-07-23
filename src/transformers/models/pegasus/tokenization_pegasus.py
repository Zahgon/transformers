
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


class PegasusTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask_2>",
        mask_token_sent="<mask_1>",
        _spm_precompiled_charsmap=None,
        additional_special_tokens=None,
        offset=103,
        **kwargs,
    ):
        self.offset = offset

        if additional_special_tokens is None or mask_token_sent not in additional_special_tokens:
            additional_special_tokens = [mask_token_sent] if mask_token_sent is not None else []
        else:
            additional_special_tokens = []
        additional_special_tokens += [f"<unk_{i}>" for i in range(2, self.offset)]

        if vocab is None:
            vocab = [(str(unk_token), 0.0), (str(pad_token), 0.0), (str(eos_token), 0.0), (str(mask_token), 0.0)]

        self._vocab = vocab
        self._tokenizer = Tokenizer(Unigram(vocab=vocab, unk_id=self._vocab.index((str(unk_token), 0.0), 1)))
        if _spm_precompiled_charsmap is not None:
            self._tokenizer.normalizer = normalizers.Sequence(
                [normalizers.Precompiled(_spm_precompiled_charsmap), normalizers.Replace(Regex(r" {2,}"), " ")]
            )
        else:
            self._tokenizer.normalizer = normalizers.Sequence(
                [normalizers.Replace(Regex(r"\n"), " "), normalizers.Replace(Regex(r" {2,}"), " ")]
            )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)

        super().__init__(
            pad_token=pad_token,
            eos_token=eos_token,
            unk_token=unk_token,
            mask_token=mask_token,
            mask_token_sent=mask_token_sent,
            offset=offset,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A {eos_token}",
            pair=f"$A $B {eos_token}",
            special_tokens=[(str(eos_token), self.convert_tokens_to_ids(str(eos_token)))],
        )


__all__ = ["PegasusTokenizer"]
