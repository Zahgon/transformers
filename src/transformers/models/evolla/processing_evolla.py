
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import (
    ProcessorMixin,
)
from ...utils import auto_docstring


PROTEIN_VALID_KEYS = ["aa_seq", "foldseek", "msa"]


@auto_docstring
class EvollaProcessor(ProcessorMixin):
    def __init__(self, protein_tokenizer, tokenizer=None, protein_max_length=1024, text_max_length=512, **kwargs):
        r"""
        protein_tokenizer (`EsmTokenizer`):
            An instance of [`EsmTokenizer`]. The protein tokenizer is a required input.
        protein_max_length (`int`, *optional*, defaults to 1024):
            The maximum length of the sequence to be generated.
        text_max_length (`int`, *optional*, defaults to 512):
            The maximum length of the text to be generated.
        """
        if protein_tokenizer is None:
            raise ValueError("You need to specify an `protein_tokenizer`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(protein_tokenizer, tokenizer)

        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.protein_max_length = protein_max_length
        self.text_max_length = text_max_length

    def process_proteins(self, proteins, protein_max_length=1024):
        pass

    def process_text(
        self,
        texts,
        text_max_length: int = 512,
    ):
        pass

    @auto_docstring
    def __call__(
        self,
        proteins: list[dict] | dict | None = None,
        messages_list: list[list[dict]] | list[dict] | None = None,
        protein_max_length: int | None = None,
        text_max_length: int | None = None,
        **kwargs,
    ):
        r"""
        proteins (`Union[List[dict], dict]`):
            A list of dictionaries or a single dictionary containing the following keys:
                - `"aa_seq"` (`str`) -- The amino acid sequence of the protein.
                - `"foldseek"` (`str`) -- The foldseek string of the protein.
        messages_list (`Union[List[List[dict]], List[dict]]`):
            A list of lists of dictionaries or a list of dictionaries containing the following keys:
                - `"role"` (`str`) -- The role of the message.
                - `"content"` (`str`) -- The content of the message.
        protein_max_length (`int`, *optional*, defaults to 1024):
            The maximum length of the sequence to be generated.
        text_max_length (`int`, *optional*, defaults to 512):
            The maximum length of the text.

        Return:
            a dict with following keys:
                - `protein_input_ids` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The input IDs for the protein sequence.
                - `protein_attention_mask` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The attention mask for the protein sequence.
                - `text_input_ids` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The input IDs for the text sequence.
                - `text_attention_mask` (`torch.Tensor` of shape `(batch_size, sequence_length)`) -- The attention mask for the text sequence.
        """
        if proteins is None or messages_list is None:
            raise ValueError("You need to specify `messages_list` and `proteins`.")

        protein_max_length = protein_max_length if protein_max_length is not None else self.protein_max_length
        text_max_length = text_max_length if text_max_length is not None else self.text_max_length

        if isinstance(proteins, dict):
            proteins = [proteins]
        if isinstance(messages_list, (list, tuple)) and not isinstance(messages_list[0], (list, tuple)):
            messages_list = [messages_list]
        if isinstance(proteins, (list, tuple)) and not all(isinstance(p, dict) for p in proteins):
            raise ValueError("The proteins should be a list of dictionaries, but not all elements are dictionaries.")
        if isinstance(proteins, (list, tuple)) and not all(
            all(k in PROTEIN_VALID_KEYS for k in p.keys()) for p in proteins
        ):
            raise ValueError(
                "There should be a list of dictionaries with keys: "
                f"{', '.join(PROTEIN_VALID_KEYS)} for each protein."
                f"But got: {proteins}"
            )
        if isinstance(messages_list, (list, tuple)):
            for messages in messages_list:
                if not isinstance(messages, (list, tuple)):
                    raise TypeError(f"Each messages in messages_list should be a list instead of {type(messages)}.")
                if not all(isinstance(m, dict) for m in messages):
                    raise ValueError(
                        "Each message in messages_list should be a list of dictionaries, but not all elements are dictionaries."
                    )
                if any(len(m.keys()) != 2 for m in messages) or any(
                    set(m.keys()) != {"role", "content"} for m in messages
                ):
                    raise ValueError(
                        "Each message in messages_list should be a list of dictionaries with two keys: 'role' and 'content'."
                        f"But got: {messages}"
                    )
        else:
            raise ValueError(
                f"The messages_list should be a list of lists of dictionaries, but it's {type(messages_list)}."
            )
        sa_tokens = self.process_proteins(proteins, protein_max_length)

        text_tokens = self.process_text(messages_list, text_max_length)

        return BatchFeature(
            data={
                "protein_input_ids": sa_tokens["input_ids"],
                "protein_attention_mask": sa_tokens["attention_mask"],
                "input_ids": text_tokens["input_ids"],
                "attention_mask": text_tokens["attention_mask"],
            }
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def protein_batch_decode(self, *args, **kwargs):
        pass

    def protein_decode(self, *args, **kwargs):
        pass


__all__ = ["EvollaProcessor"]
