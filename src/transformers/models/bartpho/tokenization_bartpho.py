
import os
from shutil import copyfile
from typing import Any

from ...tokenization_python import AddedToken
from ...tokenization_utils_sentencepiece import SentencePieceBackend
from ...utils import logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model", "monolingual_vocab_file": "dict.txt"}


@requires(backends=("sentencepiece",))
class BartphoTokenizer(SentencePieceBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    is_fast = False

    def __init__(
        self,
        vocab_file,
        monolingual_vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token

        self.monolingual_vocab_file = monolingual_vocab_file

        self.fairseq_tokens_to_ids = {}
        cnt = 0
        for token in [bos_token, pad_token, eos_token, unk_token, sep_token, cls_token]:
            if str(token) not in self.fairseq_tokens_to_ids:
                self.fairseq_tokens_to_ids[str(token)] = cnt
                cnt += 1
        with open(monolingual_vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip().split()[0]
                self.fairseq_tokens_to_ids[token] = len(self.fairseq_tokens_to_ids)
        if str(mask_token) not in self.fairseq_tokens_to_ids:
            self.fairseq_tokens_to_ids[str(mask_token)] = len(self.fairseq_tokens_to_ids)

        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

        if sp_model_kwargs is not None:
            kwargs["sp_model_kwargs"] = sp_model_kwargs

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
        self._align_added_tokens_with_fairseq_vocab()

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An BARTPho sequence has the following format:

        - single sequence: `<s> X </s>`
        - pair of sequences: `<s> A </s></s> B </s>`

        Args:
            token_ids_0 (`list[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """

        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

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
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1, 1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. BARTPho does not
        make use of token type ids, therefore a list of zeros is returned.

        Args:
            token_ids_0 (`list[int]`):
                List of IDs.
            token_ids_1 (`list[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `list[int]`: List of zeros.

        """

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep + sep + token_ids_1 + sep) * [0]

    @property
    def vocab_size(self):
        pass

    def get_vocab(self):
        """Override to use fairseq vocabulary"""
        vocab = dict(self.fairseq_tokens_to_ids)
        if hasattr(self, "_added_tokens_encoder"):
            for token, idx in self._added_tokens_encoder.items():
                if token not in vocab:
                    vocab[token] = idx
        return vocab

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the fairseq vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        else:
            return self.unk_token_id

    def _convert_token_to_id_with_added_voc(self, token):
        """Override to use fairseq vocab instead of sp_model vocab."""
        if token is None:
            return None

        if token in self._added_tokens_encoder:
            return self._added_tokens_encoder[token]
        return self._convert_token_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the fairseq vocab."""
        return self.fairseq_ids_to_tokens[index]

    def _align_added_tokens_with_fairseq_vocab(self):
        """
        The slow tokenizer base class populates `_added_tokens_*` using SentencePiece ids. Remap those entries so that
        every token present in the reduced fairseq dictionary uses the same ids everywhere, otherwise conversions and
        special-token setters observe two different vocabularies.
        """
        if not hasattr(self, "_added_tokens_decoder") or not hasattr(self, "_added_tokens_encoder"):
            return

        remapped_decoder: dict[int, AddedToken] = {}
        for original_id, token_obj in self._added_tokens_decoder.items():
            token = token_obj.content
            new_id = self.fairseq_tokens_to_ids.get(token, original_id)
            remapped_decoder[new_id] = token_obj

        self._added_tokens_decoder = remapped_decoder
        self._added_tokens_encoder = {token.content: idx for idx, token in remapped_decoder.items()}

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )
        out_monolingual_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["monolingual_vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        if os.path.abspath(self.monolingual_vocab_file) != os.path.abspath(
            out_monolingual_vocab_file
        ) and os.path.isfile(self.monolingual_vocab_file):
            copyfile(self.monolingual_vocab_file, out_monolingual_vocab_file)
        elif not os.path.isfile(self.monolingual_vocab_file):
            with open(out_monolingual_vocab_file, "w", encoding="utf-8") as fp:
                for token in self.fairseq_tokens_to_ids:
                    if token not in self.all_special_tokens:
                        fp.write(f"{str(token)} \n")

        return out_vocab_file, out_monolingual_vocab_file


__all__ = ["BartphoTokenizer"]
