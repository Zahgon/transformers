
import json
import os

from ...tokenization_python import PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class MgpstrTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, vocab_file, unk_token="[GO]", bos_token="[GO]", eos_token="[s]", pad_token="[GO]", **kwargs):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.vocab = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.vocab.items()}
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            special_tokens_pattern="none",
            **kwargs,
        )

    @property
    def vocab_size(self):
        pass

    def get_vocab(self):
        vocab = dict(self.vocab).copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        """Tokenize a string."""
        char_tokens = []
        for s in text:
            char_tokens.extend(s)
        return char_tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(self.vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        return (vocab_file,)


__all__ = ["MgpstrTokenizer"]
