
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class OpenAIGPTTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        unk_token: str = "<unk>",
        **kwargs,
    ):
        self._vocab = vocab if vocab is not None else {str(unk_token): 0}
        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                continuing_subword_prefix="",
                end_of_word_suffix="</w>",
                fuse_unk=False,
                unk_token=str(unk_token),
            )
        )

        self._tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        super().__init__(
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def do_lower_case(self):
        pass


__all__ = ["OpenAIGPTTokenizer"]
