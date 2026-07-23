
import multiprocessing as mp
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from random import randint
from typing import Any

import numpy as np

from ..tokenization_utils_base import PreTrainedTokenizerBase
from ..utils import PaddingStrategy


InputDataClass = Any

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch tensors or NumPy arrays.
"""
DataCollator = Callable[[list[InputDataClass]], dict[str, Any]]


class DataCollatorMixin:
    def __call__(self, features, return_tensors: str | None = None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    pass


def default_data_collator(features: list[InputDataClass], return_tensors="pt") -> dict[str, Any]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """


    if return_tensors == "pt":
        return torch_default_data_collator(features)
    elif return_tensors == "np":
        return numpy_default_data_collator(features)


@dataclass
class DefaultDataCollator(DataCollatorMixin):

    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]], return_tensors=None) -> dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        return default_data_collator(features, return_tensors)


def torch_default_data_collator(features: list[InputDataClass]) -> dict[str, Any]:
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if isinstance(first["label_ids"][0], int) else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def numpy_default_data_collator(features: list[InputDataClass]) -> dict[str, Any]:
    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], np.ndarray) else first["label"]
        dtype = np.int64 if isinstance(label, int) else np.float32
        batch["labels"] = np.array([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], np.ndarray):
            batch["labels"] = np.stack([f["label_ids"] for f in features])
        else:
            dtype = np.int64 if isinstance(first["label_ids"][0], int) else np.float32
            batch["labels"] = np.array([f["label_ids"] for f in features], dtype=dtype)

    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, np.ndarray):
                batch[k] = np.stack([f[k] for f in features])
            else:
                batch[k] = np.array([f[k] for f in features])

    return batch


@dataclass
class DataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch


@dataclass
class DataCollatorForTokenClassification(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        pass

    def numpy_call(self, features):
        pass


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: int | None = None):
    pass


def _numpy_collate_batch(examples, tokenizer, pad_to_multiple_of: int | None = None):
    pass


@dataclass
class DataCollatorForMultipleChoice(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]):  # Refactored implementation from the docs.
        pass


@dataclass
class DataCollatorForSeq2Seq:

    tokenizer: PreTrainedTokenizerBase
    model: Any | None = None
    padding: bool | str | PaddingStrategy = True
    max_length: int | None = None
    pad_to_multiple_of: int | None = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0] else None
        if labels is not None and all(label is None for label in labels):
            labels = None
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                if isinstance(features[0][label_name], list):
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label,
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.int64),
                                label,
                            ]
                        )
                        for label in labels
                    ]

        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch

                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.int64)
            else:
                batch["labels"] = np.array(batch["labels"], dtype=np.int64)
        else:
            batch["labels"] = None

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch


@dataclass
class DataCollatorForLanguageModeling(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True
    whole_word_mask: bool = False
    mlm_probability: float | None = 0.15
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"
    seed: int | None = None

    def __post_init__(self):
        if self.mlm:
            if self.tokenizer.mask_token is None:
                raise ValueError(
                    "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                    "You should pass `mlm=False` to train on causal language modeling instead."
                )
            if self.mlm_probability is None or self.mlm_probability < 0 or self.mlm_probability > 1:
                raise ValueError("mlm_probability should be between 0 and 1.")
            self.mlm_probability = float(self.mlm_probability)
        elif self.whole_word_mask:
            raise ValueError(
                "Whole word masking can only be used with mlm=True."
                "If you want to use whole word masking, please set mlm=True."
            )
        if self.mask_replace_prob + self.random_replace_prob > 1:
            raise ValueError("The sum of mask_replace_prob and random_replace_prob should not exceed 1")
        if self.mask_replace_prob < 0 or self.mask_replace_prob > 1:
            raise ValueError("mask_replace_prob should be between 0 and 1.")
        if self.random_replace_prob < 0 or self.random_replace_prob > 1:
            raise ValueError("random_replace_prob should be between 0 and 1.")

        if self.whole_word_mask:
            if not self.tokenizer.is_fast:
                warnings.warn(
                    "Whole word masking depends on offset mapping which is only natively available with fast tokenizers.",
                    UserWarning,
                )

            if self.mask_replace_prob < 1:
                warnings.warn(
                    "Random token replacement is not supported with whole word masking. "
                    "Setting mask_replace_prob to 1.",
                )
                self.mask_replace_prob = 1
                self.random_replace_prob = 0

        self.mask_replace_prob = float(self.mask_replace_prob)
        self.random_replace_prob = float(self.random_replace_prob)

        self.generator = None

    def get_generator(self, seed):
        pass

    def create_rng(self):
        pass

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        pass

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Any | None = None, offset_mapping: Any | None = None
    ) -> tuple[Any, Any]:
        pass

    def numpy_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        pass

    def numpy_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Any | None = None,
        offset_mapping: Any | None = None,
    ) -> tuple[Any, Any]:
        pass

    @staticmethod
    def _calc_word_ids_and_prob_mask(
        offsets: np.ndarray[np.ndarray[tuple[int, int]]], special_tokens_mask: np.ndarray[np.ndarray[int]]
    ) -> tuple[np.ndarray[np.ndarray[int]], np.ndarray[np.ndarray[int]]]:
        pass

    @staticmethod
    def _whole_word_mask(word_ids: np.ndarray[np.ndarray[int]], mask: Any) -> Any:
        pass


@dataclass
class DataCollatorForWholeWordMask(DataCollatorForLanguageModeling):

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DataCollatorForWholeWordMask is deprecated and will be removed in a future version, you can now use "
            "DataCollatorForLanguageModeling with whole_word_mask=True instead.",
            FutureWarning,
        )
        super().__init__(*args, **kwargs)
        self.mlm = True  # Force masked language modeling
        self.whole_word_mask = True  # Force whole word masking


def tolist(x) -> list[Any]:
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):
        x = x.numpy()
    return x.tolist()


def to_numpy(x) -> np.ndarray[Any]:
    if isinstance(x, np.ndarray):
        return x
    elif hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    else:
        return np.array(x)


@dataclass
class DataCollatorForSOP(DataCollatorForLanguageModeling):

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DataCollatorForSOP is deprecated and will be removed in a future version, you can now use "
            "DataCollatorForLanguageModeling instead.",
            FutureWarning,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        input_ids = [example["input_ids"] for example in examples]
        input_ids = _torch_collate_batch(input_ids, self.tokenizer)
        input_ids, labels, attention_mask = self.mask_tokens(input_ids)

        token_type_ids = [example["token_type_ids"] for example in examples]
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        sop_label_list = [example["sentence_order_label"] for example in examples]
        sentence_order_label = torch.stack(sop_label_list)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "sentence_order_label": sentence_order_label,
        }

    def mask_tokens(self, inputs: Any) -> tuple[Any, Any, Any]:
        pass


@dataclass
class DataCollatorForPermutationLanguageModeling(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase
    plm_probability: float = 1 / 6
    max_span_length: int = 5  # maximum length of a span of masked tokens
    return_tensors: str = "pt"

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        pass

    def numpy_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        pass

    def torch_mask_tokens(self, inputs: Any) -> tuple[Any, Any, Any, Any]:
        pass

    def numpy_mask_tokens(self, inputs: Any) -> tuple[Any, Any, Any, Any]:
        pass


@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):

    def __init__(
        self,
        *args,
        return_position_ids=True,
        separator_id=-100,
        return_flash_attn_kwargs=False,
        return_seq_idx=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id
        self.return_flash_attn_kwargs = return_flash_attn_kwargs
        self.return_seq_idx = return_seq_idx
        self._int_64_keys = {"labels", "position_ids", "input_ids"}
        self._batch_dim_keys = {"labels", "position_ids", "input_ids", "seq_idx"}
        self._py_int_keys = {"max_length_q", "max_length_k"}

    def __call__(self, features, return_tensors=None, separator_id=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id
        is_labels_provided = "labels" in features[0]
        batch = {"input_ids": [], "labels": []}
        if self.return_position_ids:
            batch.update({"position_ids": []})
        if self.return_seq_idx:
            batch.update({"seq_idx": []})
        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0
        for seq_idx, sample in enumerate(features):
            input_ids = sample["input_ids"]
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            batch["input_ids"] += input_ids

            if is_labels_provided:
                labels = sample["labels"]
                if hasattr(labels, "tolist"):
                    labels = labels.tolist()
                batch["labels"] += [separator_id] + labels[1:]
            else:
                batch["labels"] += [separator_id] + input_ids[1:]
            if self.return_position_ids:
                batch["position_ids"] += list(range(len(input_ids)))
            if self.return_seq_idx:
                batch["seq_idx"] += [seq_idx for _ in range(len(input_ids))]
            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))

        if self.return_flash_attn_kwargs:
            batch["cu_seq_lens_q"] = batch["cu_seq_lens_k"] = cu_seq_lens
            batch["max_length_q"] = batch["max_length_k"] = max_length

        if return_tensors == "pt":
            import torch

            data_cls = torch.tensor
            dtype_64 = torch.int64
            dtype_32 = torch.int32
        elif return_tensors == "np":
            data_cls = np.array
            dtype_64 = np.int64
            dtype_32 = np.int32
        else:
            raise ValueError(f'return_tensors must be one of ("pt", "np"), {return_tensors=} not supported')

        for k, v in batch.items():
            if k in self._batch_dim_keys:
                v = [v]
            if k not in self._py_int_keys:
                batch[k] = data_cls(v, dtype=dtype_64 if k in self._int_64_keys else dtype_32)

        return batch
