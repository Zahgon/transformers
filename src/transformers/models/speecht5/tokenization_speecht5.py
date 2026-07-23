
from typing import Any

from ...tokenization_utils_sentencepiece import SentencePieceBackend
from ...utils import logging
from ...utils.import_utils import requires
from .number_normalizer import EnglishNumberNormalizer


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spm_char.model"}


@requires(backends=("sentencepiece",))
class SpeechT5Tokenizer(SentencePieceBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    is_fast = False

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        normalize=False,
        sp_model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.normalize = normalize
        self._normalizer = None

        if sp_model_kwargs is not None:
            kwargs["sp_model_kwargs"] = sp_model_kwargs

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            normalize=normalize,
            **kwargs,
        )

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        normalize = kwargs.pop("normalize", self.normalize)
        if is_split_into_words:
            text = " " + text
        if normalize:
            text = self.normalizer(text)
        return (text, kwargs)

    @property
    def normalizer(self):
        if self._normalizer is None:
            self._normalizer = EnglishNumberNormalizer()
        return self._normalizer

    @normalizer.setter
    def normalizer(self, value):
        self._normalizer = value

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> list[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        suffix_ones = [1]
        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + suffix_ones
        return ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. SpeechT5 does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`list[int]`):
                List of IDs.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of zeros.
        """
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(token_ids_0 + eos) * [0]
        return len(token_ids_0 + token_ids_1 + eos) * [0]


__all__ = ["SpeechT5Tokenizer"]
