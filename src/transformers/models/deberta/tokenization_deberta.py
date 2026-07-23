
from tokenizers import AddedToken, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class DebertaTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        errors="replace",
        bos_token="[CLS]",
        eos_token="[SEP]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        add_prefix_space=False,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space

        self._vocab = (
            vocab
            if vocab is not None
            else {
                str(unk_token): 0,
                str(cls_token): 1,
                str(sep_token): 2,
                str(pad_token): 3,
                str(mask_token): 4,
            }
        )

        self._merges = merges or []

        self._tokenizer = Tokenizer(
            BPE(
                vocab=self._vocab,
                merges=self._merges,
                dropout=None,
                unk_token=None,
                continuing_subword_prefix="",
                end_of_word_suffix="",
                fuse_unk=False,
            )
        )

        self._tokenizer.normalizer = None

        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        self._tokenizer.decoder = decoders.ByteLevel()

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.cls_token} $A {self.sep_token}",
            pair=f"{self.cls_token} $A {self.sep_token} {self.sep_token} $B {self.sep_token}",
            special_tokens=[
                (self.cls_token, self.cls_token_id),
                (self.sep_token, self.sep_token_id),
            ],
        )

    @property
    def mask_token(self) -> str:
        pass

    @mask_token.setter
    def mask_token(self, value):
        pass


__all__ = ["DebertaTokenizer"]
