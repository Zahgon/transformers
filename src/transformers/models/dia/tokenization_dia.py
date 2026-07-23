
from ...tokenization_python import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


class DiaTokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        pad_token: str | None = "<pad>",
        unk_token: str | None = "<pad>",
        max_length: int | None = 1024,
        offset: int = 0,
        **kwargs,
    ):
        pad_token = AddedToken(pad_token) if isinstance(pad_token, str) else pad_token
        unk_token = AddedToken(unk_token) if isinstance(unk_token, str) else unk_token

        self._utf_vocab_size = 2**8  # utf is 8 bits
        self._added_tokens_decoder = {0: pad_token, 1: AddedToken("[S1]"), 2: AddedToken("[S2]")}
        self.offset = offset
        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            max_length=max_length,
            offset=offset,
            token_type_ids_pattern="all_zeros",
            token_type_ids_include_special_tokens=True,
            special_tokens_pattern="none",
            **kwargs,
        )

    @property
    def vocab_size(self):
        pass

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words"""
        tokens = [chr(i) for i in text.encode("utf-8")]
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""

        if len(token) != 1:
            token_id = None
        else:
            token_id = ord(token) + self.offset

        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = chr(index - self.offset)
        return token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""
        for token in tokens:
            if token in self._added_tokens_decoder:
                added_token_obj = self._added_tokens_decoder[token]
                tok_string = str(added_token_obj).encode("utf-8")
            elif token in self._added_tokens_encoder:
                tok_string = token.encode("utf-8")
            else:
                tok_string = token.encode("utf-8")  # Assume general string token
            bstring += tok_string
        string = bstring.decode("utf-8", errors="ignore")
        return string


__all__ = ["DiaTokenizer"]
