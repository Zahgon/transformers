
from tokenizers import Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...tokenization_utils_tokenizers import TokenizersBackend
from ...utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt", "tokenizer_file": "tokenizer.json"}


class RobertaTokenizer(TokenizersBackend):

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]
    model = BPE

    def __init__(
        self,
        vocab: str | dict[str, int] | None = None,
        merges: str | list[str] | None = None,
        errors: str = "replace",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        sep_token: str = "</s>",
        cls_token: str = "<s>",
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        mask_token: str = "<mask>",
        add_prefix_space: bool = False,
        trim_offsets: bool = True,
        **kwargs,
    ):
        self.add_prefix_space = add_prefix_space
        self.trim_offsets = trim_offsets

        if vocab is None:
            vocab = {
                str(pad_token): 0,
                str(unk_token): 1,
                str(cls_token): 2,
                str(sep_token): 3,
                str(mask_token): 4,
            }
        self._vocab = vocab

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

        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space)
        self._tokenizer.decoder = decoders.ByteLevel()

        super().__init__(
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            unk_token=unk_token,
            pad_token=pad_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
            **kwargs,
        )
        self._tokenizer.post_processor = processors.RobertaProcessing(
            sep=(str(sep_token), self.sep_token_id),
            cls=(str(cls_token), self.cls_token_id),
            add_prefix_space=add_prefix_space,
            trim_offsets=trim_offsets,
        )


__all__ = ["RobertaTokenizer"]
