import json
import os
import warnings
from pathlib import Path
from shutil import copyfile
from typing import Any

import sentencepiece

from ...tokenization_python import PreTrainedTokenizer
from ...utils import logging
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "source_spm": "source.spm",
    "target_spm": "target.spm",
    "vocab": "vocab.json",
    "target_vocab_file": "target_vocab.json",
    "tokenizer_config_file": "tokenizer_config.json",
}


SPIECE_UNDERLINE = "▁"



@requires(backends=("sentencepiece",))
class MarianTokenizer(PreTrainedTokenizer):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        source_spm,
        target_spm,
        vocab,
        target_vocab_file=None,
        source_lang=None,
        target_lang=None,
        unk_token="<unk>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=512,
        sp_model_kwargs: dict[str, Any] | None = None,
        separate_vocabs=False,
        **kwargs,
    ) -> None:
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        assert Path(source_spm).exists(), f"cannot find spm source {source_spm}"

        self.separate_vocabs = separate_vocabs
        self.encoder = load_json(vocab)
        if str(unk_token) not in self.encoder:
            raise KeyError("<unk> token must be in the vocab")

        if separate_vocabs:
            self.target_encoder = load_json(target_vocab_file)
            self.decoder = {v: k for k, v in self.target_encoder.items()}
            self.supported_language_codes = []
        else:
            self.decoder = {v: k for k, v in self.encoder.items()}
            self.supported_language_codes: list = [k for k in self.encoder if k.startswith(">>") and k.endswith("<<")]

        self.source_lang = source_lang
        self.target_lang = target_lang
        self.spm_files = [source_spm, target_spm]

        self.spm_source = load_spm(source_spm, self.sp_model_kwargs)
        self.spm_target = load_spm(target_spm, self.sp_model_kwargs)
        self.current_spm = self.spm_source
        self.current_encoder = self.encoder


        self._setup_normalizer()

        self._decode_use_source_tokenizer = False

        super().__init__(
            source_lang=source_lang,
            target_lang=target_lang,
            unk_token=unk_token,
            eos_token=eos_token,
            pad_token=pad_token,
            model_max_length=model_max_length,
            sp_model_kwargs=self.sp_model_kwargs,
            target_vocab_file=target_vocab_file,
            separate_vocabs=separate_vocabs,
            **kwargs,
        )

    def _setup_normalizer(self):
        try:
            from sacremoses import MosesPunctNormalizer

            self.punc_normalizer = MosesPunctNormalizer(self.source_lang).normalize
        except (ImportError, FileNotFoundError):
            warnings.warn("Recommended: pip install sacremoses.")
            self.punc_normalizer = lambda x: x

    def normalize(self, x: str) -> str:
        """Cover moses empty string edge case. They return empty list for '' input!"""
        return self.punc_normalizer(x) if x else ""

    def _convert_token_to_id(self, token):
        if token in self.current_encoder:
            return self.current_encoder[token]
        return self.current_encoder[self.unk_token]

    def remove_language_code(self, text: str):
        """Remove language codes like >>fr<< before sentencepiece"""
        code = []
        if text.startswith(">>") and (end_loc := text.find("<<")) != -1:
            code.append(text[: end_loc + 2])
            text = text[end_loc + 2 :]
        return code, text

    def _tokenize(self, text: str) -> list[str]:
        code, text = self.remove_language_code(text)
        pieces = self.current_spm.encode(text, out_type=str)
        return code + pieces

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the decoder."""
        if index in self.decoder:
            return self.decoder[index]
        spm_model = self.spm_source if self._decode_use_source_tokenizer else self.spm_target
        piece = spm_model.IdToPiece(index)
        return piece if piece else self.unk_token

    def batch_decode(self, sequences, **kwargs):
        """
        Convert a list of lists of token ids into a list of strings by calling decode.

        Args:
            sequences (`Union[list[int], list[list[int]], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            use_source_tokenizer (`bool`, *optional*, defaults to `False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `list[str]`: The list of decoded sentences.
        """
        return super().batch_decode(sequences, **kwargs)

    def decode(self, token_ids, **kwargs):
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, list[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces` (available in the `tokenizer_config`).
            use_source_tokenizer (`bool`, *optional*, defaults to `False`):
                Whether or not to use the source tokenizer to decode sequences (only applicable in sequence-to-sequence
                problems).
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        return super().decode(token_ids, **kwargs)

    def _decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        **kwargs,
    ) -> str:
        """Internal decode method that handles use_source_tokenizer parameter."""
        default_use_source = not self.separate_vocabs
        self._decode_use_source_tokenizer = kwargs.pop("use_source_tokenizer", default_use_source)
        return super()._decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Uses source spm if _decode_use_source_tokenizer is True, and target spm otherwise"""
        sp_model = self.spm_source if self._decode_use_source_tokenizer else self.spm_target
        all_special_tokens = set(self.all_special_tokens)
        current_sub_tokens = []
        out_string = ""
        for token in tokens:
            if token in all_special_tokens:
                out_string += sp_model.decode_pieces(current_sub_tokens) + token + " "
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
        out_string += sp_model.decode_pieces(current_sub_tokens)
        out_string = out_string.replace(SPIECE_UNDERLINE, " ")
        return out_string.strip()

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> list[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def _switch_to_input_mode(self):
        pass

    def _switch_to_target_mode(self):
        pass

    @property
    def vocab_size(self) -> int:
        pass

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        saved_files = []

        if self.separate_vocabs:
            out_src_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"],
            )
            out_tgt_vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["target_vocab_file"],
            )
            save_json(self.encoder, out_src_vocab_file)
            save_json(self.target_encoder, out_tgt_vocab_file)
            saved_files.append(out_src_vocab_file)
            saved_files.append(out_tgt_vocab_file)
        else:
            out_vocab_file = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab"]
            )
            save_json(self.encoder, out_vocab_file)
            saved_files.append(out_vocab_file)

        for spm_save_filename, spm_orig_path, spm_model in zip(
            [VOCAB_FILES_NAMES["source_spm"], VOCAB_FILES_NAMES["target_spm"]],
            self.spm_files,
            [self.spm_source, self.spm_target],
        ):
            spm_save_path = os.path.join(
                save_directory, (filename_prefix + "-" if filename_prefix else "") + spm_save_filename
            )
            if os.path.abspath(spm_orig_path) != os.path.abspath(spm_save_path) and os.path.isfile(spm_orig_path):
                copyfile(spm_orig_path, spm_save_path)
                saved_files.append(spm_save_path)
            elif not os.path.isfile(spm_orig_path):
                with open(spm_save_path, "wb") as fi:
                    content_spiece_model = spm_model.serialized_model_proto()
                    fi.write(content_spiece_model)
                saved_files.append(spm_save_path)

        return tuple(saved_files)

    def get_vocab(self) -> dict:
        return self.get_src_vocab()

    def get_src_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)

    def get_tgt_vocab(self):
        pass

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.update(
            dict.fromkeys(["spm_source", "spm_target", "current_spm", "punc_normalizer", "target_vocab_file"])
        )
        return state

    def __setstate__(self, d: dict) -> None:
        self.__dict__ = d

        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}
        if not hasattr(self, "_decode_use_source_tokenizer"):
            self._decode_use_source_tokenizer = False

        self.spm_source, self.spm_target = (load_spm(f, self.sp_model_kwargs) for f in self.spm_files)
        self.current_spm = self.spm_source
        self._setup_normalizer()

    def num_special_tokens_to_add(self, *args, **kwargs):
        """Just EOS"""
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: list, token_ids_1: list | None = None, already_has_special_tokens: bool = False
    ) -> list[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]


def load_spm(path: str, sp_model_kwargs: dict[str, Any]) -> sentencepiece.SentencePieceProcessor:
    spm = sentencepiece.SentencePieceProcessor(**sp_model_kwargs)
    spm.Load(path)
    return spm


def save_json(data, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)


__all__ = ["MarianTokenizer"]
