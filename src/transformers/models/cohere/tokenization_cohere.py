

from typing import Literal

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "tokenizer_file": {
        "Cohere/Command-nightly": "https://huggingface.co/Cohere/Command-nightly/blob/main/tokenizer.json",
    },
}

DEFAULT_SYSTEM_PROMPT = "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere."
DEFAULT_RAG_PREAMBLE = """## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."""


class CohereTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        errors: str = "replace",
        unk_token: str = "<UNK>",
        bos_token: str = "<BOS_TOKEN>",
        eos_token: str = "<|END_OF_TURN_TOKEN|>",
        pad_token: str = "<PAD>",
        cls_token: str = "<CLS>",
        sep_token: str = "<SEP>",
        mask_token: str = "<MASK_TOKEN>",
        use_default_system_prompt: bool = False,
        add_prefix_space: bool = False,
        **kwargs,
    ):
        self.use_default_system_prompt = use_default_system_prompt
        self.add_prefix_space = add_prefix_space
        self.grounded_generation_template = kwargs.pop("grounded_generation_template", None)
        self.tool_use_template = kwargs.pop("tool_use_template", None)

        self._vocab = (
            vocab
            if vocab is not None
            else {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
                str(bos_token): 5,
            }
        )

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
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Digits(individual_digits=True),
                pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space, trim_offsets=True),
            ]
        )
        self._tokenizer.decoder = decoders.ByteLevel(add_prefix_space=add_prefix_space, trim_offsets=True)

        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token,
            use_default_system_prompt=use_default_system_prompt,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

        self._post_init()

    def apply_tool_use_template(
        self,
        conversation: list[dict[str, str]],
        tools: list[dict],
        **kwargs,
    ) -> str | list[int]:
        pass

    def apply_grounded_generation_template(
        self,
        conversation: list[dict[str, str]],
        documents: list[dict],
        citation_mode: Literal["fast", "accurate"] = "accurate",
        **kwargs,
    ) -> str | list[int]:
        pass


__all__ = ["CohereTokenizer"]
