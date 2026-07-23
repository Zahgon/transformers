
import itertools
import re

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import Unigram

from ...tokenization_utils_tokenizers import TokenizersBackend


VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}


class LasrTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = Unigram

    def __init__(
        self,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        _spm_precompiled_charsmap=None,
        extra_ids=100,
        additional_special_tokens=None,
        vocab=None,
        vocab_file=None,
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
                    " provided to LasrTokenizer. In this case the additional_special_tokens must include the extra_ids"
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
                unk_id=3,
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

        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.eos_token_id),
            ],
        )

    def get_sentinel_tokens(self):
        pass

    def get_sentinel_token_ids(self):
        pass

    def _decode(
        self,
        token_ids: int | list[int],
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        group_tokens: bool = True,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if group_tokens:
            token_ids = [token_group[0] for token_group in itertools.groupby(token_ids)]

        token_ids = [token for token in token_ids if token != self.pad_token_id]

        return super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["LasrTokenizer"]
