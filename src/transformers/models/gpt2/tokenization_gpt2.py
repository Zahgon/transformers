
from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import AddedToken, TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}


class GPT2Tokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        errors: str = "replace",
        unk_token: AddedToken | str = "<|endoftext|>",
        bos_token: AddedToken | str = "<|endoftext|>",
        eos_token: AddedToken | str = "<|endoftext|>",
        pad_token: AddedToken | str | None = None,
        add_prefix_space=False,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space
        self._vocab = vocab if vocab is not None else {}
        self._merges = merges or []
        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        self._tokenizer.decoder = decoders.ByteLevel()
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )


__all__ = ["GPT2Tokenizer"]
