
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend


VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}


class Tipsv2Tokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE
    padding_side = "right"

    def __init__(
        self,
        vocab: dict[str, int] | None = None,
        merges: list[tuple[str, str]] | None = None,
        unk_token: str | None = "<unk>",
        pad_token: str | None = "<pad>",
        bos_token: str | None = None,
        eos_token: str | None = None,
        model_max_length: int = 64,
        do_lower_case: bool = True,
        token_type_ids_pattern: str = "all_zeros",
        _spm_precompiled_charsmap=None,
        **kwargs,
    ) -> None:
        if vocab is None:
            vocab = {
                str(pad_token): 0,
                str(unk_token): 1,
            }
        self._vocab = vocab
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                fuse_unk=True,
                byte_fallback=True,
                dropout=None,
            )
        )

        list_normalizers = []
        if do_lower_case:
            list_normalizers.append(normalizers.Lowercase())
        if _spm_precompiled_charsmap:
            list_normalizers.append(normalizers.Precompiled(_spm_precompiled_charsmap))
        self._tokenizer.normalizer = normalizers.Sequence(list_normalizers)

        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always"),
            ]
        )
        self._tokenizer.decoder = decoders.Sequence(
            [
                decoders.Metaspace(replacement="▁", prepend_scheme="always"),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ]
        )

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            model_max_length=model_max_length,
            do_lower_case=do_lower_case,
            token_type_ids_pattern=token_type_ids_pattern,
            **kwargs,
        )


__all__ = ["Tipsv2Tokenizer"]
