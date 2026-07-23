
import json
import os
from pathlib import Path
from shutil import copyfile
from typing import Any

import sentencepiece

from ...tokenization_python import PreTrainedTokenizer
from ...utils import logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "spm_file": "sentencepiece.bpe.model",
}


MAX_MODEL_INPUT_SIZES = {
    "facebook/s2t-small-librispeech-asr": 1024,
}

MUSTC_LANGS = ["pt", "fr", "ru", "nl", "ro", "it", "es", "de"]

LANGUAGES = {"mustc": MUSTC_LANGS}


@requires(backends=("sentencepiece",))
class Speech2TextTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: list[int] = []

    def __init__(
        self,
        vocab_file,
        spm_file,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        do_upper_case=False,
        do_lower_case=False,
        tgt_lang=None,
        lang_codes=None,
        additional_special_tokens=None,
        sp_model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.do_upper_case = do_upper_case
        self.do_lower_case = do_lower_case

        self.encoder = load_json(vocab_file)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.spm_file = spm_file
        self.sp_model = load_spm(spm_file, self.sp_model_kwargs)

        if lang_codes is not None:
            self.lang_codes = lang_codes
            self.langs = LANGUAGES[lang_codes]
            self.lang_tokens = [f"<lang:{lang}>" for lang in self.langs]
            self.lang_code_to_id = {lang: self.sp_model.PieceToId(f"<lang:{lang}>") for lang in self.langs}
            if additional_special_tokens is not None:
                additional_special_tokens = self.lang_tokens + additional_special_tokens
            else:
                additional_special_tokens = self.lang_tokens
            self._tgt_lang = tgt_lang if tgt_lang is not None else self.langs[0]

            self.set_tgt_lang_special_tokens(self._tgt_lang)
        else:
            self.lang_code_to_id = {}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            do_upper_case=do_upper_case,
            do_lower_case=do_lower_case,
            tgt_lang=tgt_lang,
            lang_codes=lang_codes,
            sp_model_kwargs=self.sp_model_kwargs,
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        pass

    def get_vocab(self) -> dict:
        vocab = self.encoder.copy()
        vocab.update(self.added_tokens_encoder)
        return vocab

    @property
    def tgt_lang(self) -> str | None:
        pass

    @tgt_lang.setter
    def tgt_lang(self, new_tgt_lang) -> None:
        pass

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[eos, tgt_lang_code] and suffix=[eos]."""
        lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = [lang_code_id]

    def _tokenize(self, text: str) -> list[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.encoder.get(token, self.encoder[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        return self.decoder.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        all_special_tokens = set(self.all_special_tokens)
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            if token in all_special_tokens:
                decoded = self.sp_model.decode(current_sub_tokens)
                out_string += (decoded.upper() if self.do_upper_case else decoded) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        decoded = self.sp_model.decode(current_sub_tokens)
        out_string += decoded.upper() if self.do_upper_case else decoded
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> list[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + [self.eos_token_id]
        return self.prefix_tokens + token_ids_0 + token_ids_1 + [self.eos_token_id]

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

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1]
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return prefix_ones + ([0] * len(token_ids_0)) + ([0] * len(token_ids_1)) + suffix_ones

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d: dict) -> None:
        self.__dict__ = d

        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = load_spm(self.spm_file, self.sp_model_kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        save_dir = Path(save_directory)
        assert save_dir.is_dir(), f"{save_directory} should be a directory"
        vocab_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        )
        spm_save_path = save_dir / (
            (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["spm_file"]
        )

        save_json(self.encoder, vocab_save_path)

        if os.path.abspath(self.spm_file) != os.path.abspath(spm_save_path) and os.path.isfile(self.spm_file):
            copyfile(self.spm_file, spm_save_path)
        elif not os.path.isfile(self.spm_file):
            with open(spm_save_path, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (str(vocab_save_path), str(spm_save_path))


def load_spm(path: str, sp_model_kwargs: dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    spm.Load(str(path))
    return spm


def load_json(path: str) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


__all__ = ["Speech2TextTokenizer"]
