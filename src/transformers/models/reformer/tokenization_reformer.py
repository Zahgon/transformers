
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)


SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class ReformerTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        _spm_precompiled_charsmap: str | None = None,
        additional_special_tokens: list | None = None,
        **kwargs,
    ):
        self._vocab = vocab or {}
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                unk_token=str(unk_token),
                fuse_unk=True,
                byte_fallback=False,
                dropout=None,
            )
        )

        if _spm_precompiled_charsmap is not None:
            self._tokenizer.normalizer = normalizers.Sequence(
                [
                    normalizers.Precompiled(_spm_precompiled_charsmap),
                    normalizers.Replace(pattern=Regex(" {2,}"), content=" "),
                ]
            )

        self._tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
        self._tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            additional_special_tokens=additional_special_tokens or [],
            **kwargs,
        )

        super()._post_init()


__all__ = ["ReformerTokenizer"]
