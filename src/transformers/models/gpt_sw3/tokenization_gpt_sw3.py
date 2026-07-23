
import re
import unicodedata
from typing import Any, Union

from ...tokenization_utils_sentencepiece import SentencePieceBackend
from ...utils import is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


@requires(backends=("sentencepiece",))
class GPTSw3Tokenizer(SentencePieceBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    is_fast = False

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=False,
        keep_accents=False,
        pad_token=None,
        unk_token=None,
        eos_token=None,
        bos_token=None,
        sp_model_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        name_or_path = kwargs.get("name_or_path")
        if name_or_path is None:
            logger.warning(
                "name_or_path not provided, will work for all GPTSw3 models except gpt-sw3-7b,"
                " you are testing the model, this can safely be ignored"
            )
            name_or_path = "None"

        eos_token = "<|endoftext|>" if eos_token is None else eos_token
        unk_token = "<unk>" if unk_token is None else unk_token
        if "gpt-sw3-7b" in name_or_path:
            pad_token = unk_token if pad_token is None else pad_token
            bos_token = eos_token if bos_token is None else bos_token
        else:
            pad_token = "<pad>" if pad_token is None else pad_token
            bos_token = "<s>" if bos_token is None else bos_token

        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents

        self.whitespaces = {" ", " ", " ", " ", " ", "　", " ", " ", " ", " ", "￼", ""}

        self.non_printing_characters_re = re.compile(
            f"[{''.join(map(chr, list(range(0, 9)) + list(range(11, 32)) + list(range(127, 160)) + [160, 173, 8203]))}]"
        )

        kwargs["sp_model_kwargs"] = sp_model_kwargs if sp_model_kwargs is not None else {}

        super().__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            remove_space=remove_space,
            keep_accents=keep_accents,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            special_tokens_pattern="none",
            **kwargs,
        )

    def preprocess_text(self, text: str) -> str:
        """
        Returns the preprocessed text. This procedure is identical to what was used when training the tokenizer.
        """

        text = self.non_printing_characters_re.sub("", text)

        text = "".join([char if char not in self.whitespaces else " " for char in text])

        text = unicodedata.normalize("NFC", text)
        return text

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        text = self.preprocess_text(text)
        return self.sp_model.encode(text, out_type=str)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        """Converts a sequence of tokens (strings) to a single string. Special tokens remain intact."""
        all_special_tokens = set(self.all_special_tokens)
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            if token in all_special_tokens:
                if not prev_is_special:
                    out_string += " "

                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)

        return out_string

    def encode_fast(
        self, text: str | list[str], return_tensors: str | bool = False
    ) -> Union[list[int], list[list[int]], "torch.Tensor"]:
        pass

    def decode_fast(self, token_ids: int | list[int]) -> str:
        pass


__all__ = ["GPTSw3Tokenizer"]
