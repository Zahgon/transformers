
import json
import os
import warnings
from collections import defaultdict

from ...tokenization_python import AddedToken, PreTrainedTokenizer
from ...utils import logging


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "byte_maps.json"}


class ByteRewriter:

    LEAF = "[LEAF]"

    def __init__(self, rewriting_rules: str | dict[str, str]):
        if isinstance(rewriting_rules, str):
            with open(rewriting_rules, "r") as f:
                rewriting_rules = json.load(f)
        elif not isinstance(rewriting_rules, dict):
            raise TypeError(
                f"rewriting_rules should be either a path to json file or a dict, got {type(rewriting_rules)}"
            )

        self.hash_tree = self.construct_hash_tree(rewriting_rules)
        reverse_rewriting_rules = {v: k for k, v in rewriting_rules.items()}
        self.reverse_hash_tree = self.construct_hash_tree(reverse_rewriting_rules)

    def add_leaf(self, hash_tree: dict[str, dict | list[str]], byte_in_sequence: str, byte_out_sequence: str):
        """
        Add a leaf with the output byte sequence to the hash tree.
        """
        byte_in_list = byte_in_sequence.split(" ")
        byte_out_list = byte_out_sequence.split(" ")

        tree_pointer = hash_tree
        for b in byte_in_list:
            if b not in tree_pointer:
                tree_pointer[b] = {}
            tree_pointer = tree_pointer[b]

        tree_pointer[self.LEAF] = byte_out_list

    def construct_hash_tree(self, rewriting_rules: dict[str, str]) -> dict[str, dict | list[str]]:
        """
        Construct a hash tree for rewritten byte sequences.
        """
        hash_tree = defaultdict(dict)
        for b in (f"{x:02x}" for x in range(256)):
            hash_tree[b][self.LEAF] = [b]

        for in_sequence, out_sequence in rewriting_rules.items():
            self.add_leaf(hash_tree, in_sequence, out_sequence)

        return hash_tree

    def search_hash_tree(self, byte_sequence: list[str]) -> None | list[str]:
        pass

    def rewrite_bytes(self, in_bytes: list[str], reverse=False) -> list[str]:
        """
        Rewrite a sequence of bytes using the hash tree.

        Args:
            in_bytes (`list[str]`): A list of bytes to be rewritten.
            reverse (`bool`): If True, decoding is performed with the reverse hash tree.
        Returns:
            `list[str]`: The rewritten byte sequence.
        """
        out_bytes = []
        b_start = 0
        b_end = 0

        while b_start < len(in_bytes):
            tree_pointer = self.hash_tree if not reverse else self.reverse_hash_tree
            for j in range(b_start, len(in_bytes)):
                b = in_bytes[j]
                if b in tree_pointer:
                    tree_pointer = tree_pointer[b]
                elif j == b_start:
                    cur_leaf = [b]
                    b_end = j
                    break
                else:
                    break
                if self.LEAF in tree_pointer:
                    cur_leaf = tree_pointer[self.LEAF]
                    b_end = j
            out_bytes.extend(cur_leaf)
            b_start = b_end + 1

        return out_bytes


class MyT5Tokenizer(PreTrainedTokenizer):

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=125,
        additional_special_tokens=None,
        **kwargs,
    ) -> None:
        if extra_ids > 0 and additional_special_tokens is None:
            additional_special_tokens = [f"<extra_id_{i}>" for i in range(extra_ids)]
        elif extra_ids > 0 and additional_special_tokens is not None and len(additional_special_tokens) > 0:
            extra_tokens = len(set(filter(lambda x: bool("extra_id" in str(x)), additional_special_tokens)))
            if extra_tokens != extra_ids:
                raise ValueError(
                    f"Both extra_ids ({extra_ids}) and additional_special_tokens ({additional_special_tokens}) are"
                    " provided to MyT5Tokenizer. In this case the additional_special_tokens must include the"
                    " extra_ids tokens"
                )

        pad_token = AddedToken(pad_token, lstrip=True, rstrip=True) if isinstance(pad_token, str) else pad_token
        eos_token = AddedToken(eos_token, lstrip=True, rstrip=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=True, rstrip=True) if isinstance(unk_token, str) else unk_token
        self._added_tokens_decoder = {0: pad_token, 1: eos_token, 2: unk_token}
        self.offset = len(self._added_tokens_decoder)
        self._utf_vocab_size = 2**8  # utf is 8 bits

        self.byte_maps = json.load(open(vocab_file, "r"))

        self.decompose_rewriter = ByteRewriter(self.byte_maps["decompose_map"])
        self.merge_rewriter = ByteRewriter(self.byte_maps["merge_map"])

        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            extra_ids=0,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self):
        pass

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size + self.offset)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def get_special_tokens_mask(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`list[int]`):
                List of IDs.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `list[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return ([0] * len(token_ids_0)) + [1]
        return ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def _add_eos_if_not_present(self, token_ids: list[int]) -> list[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. MyT5 does not
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
        return len(token_ids_0 + eos + token_ids_1 + eos) * [0]

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`list[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words.
        Represents tokens in two character hex format"""

        tokens = [f"{i:02x}" for i in text.encode("utf-8")]
        tokens = self.morphological_encode(tokens)
        return tokens

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""

        if len(token) != 2:
            token_id = None
        else:
            token_id = int(token, 16) + self.offset

        return token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = f"{index - self.offset:02x}"
        return token

    def morphological_encode(self, indices: list[str]) -> list[str]:
        indices = self.decompose_rewriter.rewrite_bytes(indices, reverse=False)
        indices = self.merge_rewriter.rewrite_bytes(indices, reverse=False)
        return indices

    def morphological_decode(self, indices: list[str]) -> list[str]:
        indices = self.merge_rewriter.rewrite_bytes(indices, reverse=True)
        indices = self.decompose_rewriter.rewrite_bytes(indices, reverse=True)
        return indices

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        bstring = b""

        out_tokens = []
        for token in tokens:
            if token in self._added_tokens_decoder:
                out_tokens.append(self._added_tokens_decoder[token])
            elif token in self._added_tokens_encoder:
                out_tokens.append(token)
            else:
                out_tokens.append(token)

        out_tokens = self.morphological_decode(out_tokens)
        _added_tokens = set(self.added_tokens_decoder.values()) | set(self.added_tokens_encoder)
        for token in out_tokens:
            if token in _added_tokens:
                bstring += bytes(token, "utf-8")
            else:
                bstring += bytes.fromhex(token)
        string = bstring.decode("utf-8", errors="ignore")
        return string

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory
        with open(vocab_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(self.byte_maps, indent=2, ensure_ascii=False))
        return (vocab_file,)


__all__ = ["MyT5Tokenizer"]
