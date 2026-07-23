
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class GPTNeoXTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        errors: str = "replace",
        unk_token: str = "<|endoftext|>",
        bos_token: str = "<|endoftext|>",
        eos_token: str = "<|endoftext|>",
        pad_token: str = "<|padding|>",
        add_prefix_space: bool = False,
        trim_offsets: bool = True,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space
        self.trim_offsets = trim_offsets

        self._vocab = vocab if vocab is not None else {str(unk_token): 0, str(pad_token): 1}
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

        self._tokenizer.normalizer = normalizers.NFC()
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=add_prefix_space, trim_offsets=trim_offsets
        )
        self._tokenizer.decoder = decoders.ByteLevel()

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )


__all__ = ["GPTNeoXTokenizer"]
