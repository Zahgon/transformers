
from typing import Any

from ...tokenization_utils_sentencepiece import SentencePieceBackend
from ...utils import logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


@requires(backends=("sentencepiece",))
class BertGenerationTokenizer(SentencePieceBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    prefix_tokens: list[int] = []
    model_input_names = ["input_ids", "attention_mask"]
    is_fast = False

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        sep_token="<::::>",
        sp_model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            sp_model_kwargs=self.sp_model_kwargs,
            special_tokens_pattern="none",
            **kwargs,
        )


__all__ = ["BertGenerationTokenizer"]
