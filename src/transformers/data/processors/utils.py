
import csv
import dataclasses
import json
from dataclasses import dataclass

from ...utils import is_torch_available, logging


logger = logging.get_logger(__name__)


@dataclass
class InputExample:

    guid: str
    text_a: str
    text_b: str | None = None
    label: str | None = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:

    input_ids: list[int]
    attention_mask: list[int] | None = None
    token_type_ids: list[int] | None = None
    label: int | float | None = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class DataProcessor:

    def get_example_from_tensor_dict(self, tensor_dict):
        """
        Gets an example from a dict.

        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        raise NotImplementedError()

    def get_train_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of [`InputExample`] for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def tfds_map(self, example):
        pass

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


class SingleSentenceClassificationProcessor(DataProcessor):

    def __init__(self, labels=None, examples=None, mode="classification", verbose=False):
        self.labels = [] if labels is None else labels
        self.examples = [] if examples is None else examples
        self.mode = mode
        self.verbose = verbose

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SingleSentenceClassificationProcessor(labels=self.labels, examples=self.examples[idx])
        return self.examples[idx]

    @classmethod
    def create_from_csv(
        cls, file_name, split_name="", column_label=0, column_text=1, column_id=None, skip_first_row=False, **kwargs
    ):
        pass

    @classmethod
    def create_from_examples(cls, texts_or_text_and_labels, labels=None, **kwargs):
        pass

    def add_examples_from_csv(
        self,
        file_name,
        split_name="",
        column_label=0,
        column_text=1,
        column_id=None,
        skip_first_row=False,
        overwrite_labels=False,
        overwrite_examples=False,
    ):
        pass

    def add_examples(
        self, texts_or_text_and_labels, labels=None, ids=None, overwrite_labels=False, overwrite_examples=False
    ):
        pass

    def get_features(
        self,
        tokenizer,
        max_length=None,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
        return_tensors=None,
    ):
        pass
