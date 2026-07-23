

import re

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend


VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


class VideoPrismTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        vocab: str | list[tuple[str, float]] | None = None,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        _spm_precompiled_charsmap=None,
        extra_ids=100,
        additional_special_tokens=None,
        **kwargs,
    ):
        self._extra_ids = extra_ids

        if additional_special_tokens is not None:
            extra_tokens = [x for x in additional_special_tokens if "<extra_id_" in str(x)]
            if len(extra_tokens) < 1:
                additional_special_tokens += [f"<extra_id_{i}>" for i in range(extra_ids)]
            elif extra_ids > 0 and extra_ids != len(extra_tokens):
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to VideoPrismTokenizer. In this case the additional_special_tokens must include the extra_ids"
                    " tokens"
                )
        else:
            extra_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
            additional_special_tokens = extra_tokens

        if vocab is not None:
            self._vocab_scores = vocab
        else:
            self._vocab_scores = [
                (str(pad_token), 0.0),
                (str(eos_token), 0.0),
                (str(unk_token), 0.0),
                ("▁", -2.0),  # Space token
            ]
            for i in range(extra_ids - 1, -1, -1):
                self._vocab_scores.append((f"<extra_id_{i}>", 0.0))

        self._tokenizer = Tokenizer(
            Unigram(
                self._vocab_scores,
                unk_id=2,
                byte_fallback=False,
            )
        )

        if _spm_precompiled_charsmap is not None:
            self._tokenizer.normalizer = normalizers.Precompiled(_spm_precompiled_charsmap)

        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True),
            ]
        )
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=extra_ids,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    def get_sentinel_tokens(self):
        pass

    def get_sentinel_token_ids(self):
        pass


__all__ = ["VideoPrismTokenizer"]
