

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}


class HerbertTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        sep_token: str = "</s>",
        vocab_file: str | None = None,
        merges_file: str | None = None,
        **kwargs,
    ):
        self._vocab = vocab if vocab is not None else {str(unk_token): 0}
        self._merges = merges or []
        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=str(unk_token),
                end_of_word_suffix="</w>",
            )
        )

        self._tokenizer.normalizer = normalizers.BertNormalizer(
            lowercase=False, strip_accents=False, clean_text=True, handle_chinese_chars=True
        )
        self._tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        self._tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")

        super().__init__(
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sep_token=sep_token,
            **kwargs,
        )

        self._tokenizer.post_processor = processors.BertProcessing(
            sep=(self.sep_token, 2),
            cls=(self.cls_token, 0),
        )


__all__ = ["HerbertTokenizer"]
