
import os
import pickle
import time
from collections.abc import Iterable

import numpy as np

from ...tokenization_python import PreTrainedTokenizer
from ...tokenization_utils_base import BatchEncoding
from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends, strtobool
from .configuration_rag import RagConfig
from .tokenization_rag import RagTokenizer


if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss


logger = logging.get_logger(__name__)


LEGACY_INDEX_PATH = "https://storage.googleapis.com/huggingface-nlp/datasets/wiki_dpr/"


class Index:

    def get_doc_dicts(self, doc_ids: np.ndarray) -> list[dict]:
        """
        Returns a list of dictionaries, containing titles and text of the retrieved documents.

        Args:
            doc_ids (`np.ndarray` of shape `(batch_size, n_docs)`):
                A tensor of document indices.
        """
        raise NotImplementedError

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> tuple[np.ndarray, np.ndarray]:
        """
        For each query in the batch, retrieves `n_docs` documents.

        Args:
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`):
                An array of query vectors.
            n_docs (`int`):
                The number of docs retrieved per query.

        Returns:
            `np.ndarray` of shape `(batch_size, n_docs)`: A tensor of indices of retrieved documents. `np.ndarray` of
            shape `(batch_size, vector_size)`: A tensor of vector representations of retrieved documents.
        """
        raise NotImplementedError

    def is_initialized(self):
        """
        Returns `True` if index is already initialized.
        """
        raise NotImplementedError

    def init_index(self):
        """
        A function responsible for loading the index into memory. Should be called only once per training run of a RAG
        model. E.g. if the model is trained on multiple GPUs in a distributed setup, only one of the workers will load
        the index.
        """
        raise NotImplementedError


class LegacyIndex(Index):

    INDEX_FILENAME = "hf_bert_base.hnswSQ8_correct_phi_128.c_index"
    PASSAGE_FILENAME = "psgs_w100.tsv.pkl"

    def __init__(self, vector_size, index_path):
        requires_backends(self, ["faiss"])
        self.index_id_to_db_id = []
        self.index_path = index_path
        self.passages = self._load_passages()
        self.vector_size = vector_size
        self.index = None
        self._index_initialized = False

    def _resolve_path(self, index_path, filename):
        is_local = os.path.isdir(index_path)
        try:
            resolved_archive_file = cached_file(index_path, filename)
        except OSError:
            msg = (
                f"Can't load '{filename}'. Make sure that:\n\n"
                f"- '{index_path}' is a correct remote path to a directory containing a file named {filename}\n\n"
                f"- or '{index_path}' is the correct path to a directory containing a file named {filename}.\n\n"
            )
            raise OSError(msg)
        if is_local:
            logger.info(f"loading file {resolved_archive_file}")
        else:
            logger.info(f"loading file {filename} from cache at {resolved_archive_file}")
        return resolved_archive_file

    def _load_passages(self):
        logger.info(f"Loading passages from {self.index_path}")
        passages_path = self._resolve_path(self.index_path, self.PASSAGE_FILENAME)
        if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
            raise ValueError(
                "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
            )
        with open(passages_path, "rb") as passages_file:
            passages = pickle.load(passages_file)
        return passages

    def _deserialize_index(self):
        logger.info(f"Loading index from {self.index_path}")
        resolved_index_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index.dpr")
        self.index = faiss.read_index(resolved_index_path)
        resolved_meta_path = self._resolve_path(self.index_path, self.INDEX_FILENAME + ".index_meta.dpr")
        if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
            raise ValueError(
                "This part uses `pickle.load` which is insecure and will execute arbitrary code that is potentially "
                "malicious. It's recommended to never unpickle data that could have come from an untrusted source, or "
                "that could have been tampered with. If you already verified the pickle data and decided to use it, "
                "you can set the environment variable `TRUST_REMOTE_CODE` to `True` to allow it."
            )
        with open(resolved_meta_path, "rb") as metadata_file:
            self.index_id_to_db_id = pickle.load(metadata_file)
        assert len(self.index_id_to_db_id) == self.index.ntotal, (
            "Deserialized index_id_to_db_id should match faiss index size"
        )

    def is_initialized(self):
        return self._index_initialized

    def init_index(self):
        index = faiss.IndexHNSWFlat(self.vector_size + 1, 512)
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        self.index = index
        self._deserialize_index()
        self._index_initialized = True

    def get_doc_dicts(self, doc_ids: np.ndarray):
        pass

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> tuple[np.ndarray, np.ndarray]:
        pass


class HFIndexBase(Index):
    def __init__(self, vector_size, dataset, index_initialized=False):
        requires_backends(self, ["faiss"])
        self.vector_size = vector_size
        self.dataset = dataset
        self._index_initialized = index_initialized
        self._check_dataset_format(with_index=index_initialized)
        dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")

    def _check_dataset_format(self, with_index: bool):
        if not isinstance(self.dataset, Dataset):
            raise TypeError(f"Dataset should be a datasets.Dataset object, but got {type(self.dataset)}")
        if len({"title", "text", "embeddings"} - set(self.dataset.column_names)) > 0:
            raise ValueError(
                "Dataset should be a dataset with the following columns: "
                "title (str), text (str) and embeddings (arrays of dimension vector_size), "
                f"but got columns {self.dataset.column_names}"
            )
        if with_index and "embeddings" not in self.dataset.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )

    def init_index(self):
        raise NotImplementedError()

    def is_initialized(self):
        return self._index_initialized

    def get_doc_dicts(self, doc_ids: np.ndarray) -> list[dict]:
        pass

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> tuple[np.ndarray, np.ndarray]:
        pass


class CanonicalHFIndex(HFIndexBase):

    def __init__(
        self,
        vector_size: int,
        dataset_name: str = "wiki_dpr",
        dataset_split: str = "train",
        index_name: str | None = None,
        index_path: str | None = None,
        use_dummy_dataset=False,
        dataset_revision=None,
    ):
        requires_backends(self, ["faiss"])
        if int(index_path is None) + int(index_name is None) != 1:
            raise ValueError("Please provide `index_name` or `index_path`.")
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        self.dataset_revision = dataset_revision
        logger.info(f"Loading passages from {self.dataset_name}")
        dataset = load_dataset(
            self.dataset_name,
            with_index=False,
            split=self.dataset_split,
            dummy=self.use_dummy_dataset,
            revision=dataset_revision,
        )
        super().__init__(vector_size, dataset, index_initialized=False)

    def init_index(self):
        if self.index_path is not None:
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
        else:
            logger.info(f"Loading index from {self.dataset_name} with index name {self.index_name}")
            self.dataset = load_dataset(
                self.dataset_name,
                with_embeddings=True,
                with_index=True,
                split=self.dataset_split,
                index_name=self.index_name,
                dummy=self.use_dummy_dataset,
                revision=self.dataset_revision,
            )
            self.dataset.set_format("numpy", columns=["embeddings"], output_all_columns=True)
        self._index_initialized = True


class CustomHFIndex(HFIndexBase):

    def __init__(self, vector_size: int, dataset, index_path=None):
        requires_backends(self, ["faiss"])
        super().__init__(vector_size, dataset, index_initialized=index_path is None)
        self.index_path = index_path

    @classmethod
    def load_from_disk(cls, vector_size, dataset_path, index_path):
        logger.info(f"Loading passages from {dataset_path}")
        if dataset_path is None or index_path is None:
            raise ValueError(
                "Please provide `dataset_path` and `index_path` after calling `dataset.save_to_disk(dataset_path)` "
                "and `dataset.get_index('embeddings').save(index_path)`."
            )
        dataset = load_from_disk(dataset_path)
        return cls(vector_size=vector_size, dataset=dataset, index_path=index_path)

    def init_index(self):
        if not self.is_initialized():
            logger.info(f"Loading index from {self.index_path}")
            self.dataset.load_faiss_index("embeddings", file=self.index_path)
            self._index_initialized = True


class RagRetriever:

    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
        self._init_retrieval = init_retrieval
        requires_backends(self, ["datasets"])
        super().__init__()
        self.index = index or self._build_index(config)
        self.generator_tokenizer = generator_tokenizer
        self.question_encoder_tokenizer = question_encoder_tokenizer

        self.n_docs = config.n_docs
        self.batch_size = config.retrieval_batch_size

        self.config = config
        if self._init_retrieval:
            self.init_retrieval()

        self.ctx_encoder_tokenizer = None
        self.return_tokenized_docs = False

    @staticmethod
    def _build_index(config):
        if config.index_name == "legacy":
            return LegacyIndex(
                config.retrieval_vector_size,
                config.index_path or LEGACY_INDEX_PATH,
            )
        elif config.index_name == "custom":
            return CustomHFIndex.load_from_disk(
                vector_size=config.retrieval_vector_size,
                dataset_path=config.passages_path,
                index_path=config.index_path,
            )
        else:
            return CanonicalHFIndex(
                vector_size=config.retrieval_vector_size,
                dataset_name=config.dataset,
                dataset_split=config.dataset_split,
                index_name=config.index_name,
                index_path=config.index_path,
                use_dummy_dataset=config.use_dummy_dataset,
                dataset_revision=config.dataset_revision,
            )

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        requires_backends(cls, ["datasets"])
        config = kwargs.pop("config", None) or RagConfig.from_pretrained(retriever_name_or_path, **kwargs)
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator
        if indexed_dataset is not None:
            config.index_name = "custom"
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:
            index = cls._build_index(config)
        return cls(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
        )

    def save_pretrained(self, save_directory):
        if isinstance(self.index, CustomHFIndex):
            if self.config.index_path is None:
                index_path = os.path.join(save_directory, "hf_dataset_index.faiss")
                self.index.dataset.get_index("embeddings").save(index_path)
                self.config.index_path = index_path
            if self.config.passages_path is None:
                passages_path = os.path.join(save_directory, "hf_dataset")
                faiss_index = self.index.dataset._indexes.pop("embeddings")
                self.index.dataset.save_to_disk(passages_path)
                self.index.dataset._indexes["embeddings"] = faiss_index
                self.config.passages_path = passages_path
        self.config.save_pretrained(save_directory)
        rag_tokenizer = RagTokenizer(
            question_encoder=self.question_encoder_tokenizer,
            generator=self.generator_tokenizer,
        )
        rag_tokenizer.save_pretrained(save_directory)

    def init_retrieval(self):
        """
        Retriever initialization function. It loads the index into memory.
        """

        logger.info("initializing retrieval")
        self.index.init_index()

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        pass

    def _chunk_tensor(self, t: Iterable, chunk_size: int) -> list[Iterable]:
        pass

    def _main_retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> tuple[np.ndarray, np.ndarray]:
        pass

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        pass

    def set_ctx_encoder_tokenizer(self, ctx_encoder_tokenizer: PreTrainedTokenizer):
        pass

    def __call__(
        self,
        question_input_ids: list[list[int]],
        question_hidden_states: np.ndarray,
        prefix=None,
        n_docs=None,
        return_tensors=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified `question_hidden_states`.

        Args:
            question_input_ids (`list[list[int]]`) batch of input ids
            question_hidden_states (`np.ndarray` of shape `(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix (`str`, *optional*):
                The prefix used by the generator's tokenizer.
            n_docs (`int`, *optional*):
                The number of docs retrieved per query.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.

        Returns: [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              [What are input IDs?](../glossary#input-ids)

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

              [What are attention masks?](../glossary#attention-mask)

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else getattr(self.config.generator, "prefix", None)
        retrieved_doc_embeds, doc_ids, docs = self.retrieve(question_hidden_states, n_docs)

        input_strings = self.question_encoder_tokenizer.decode(question_input_ids, skip_special_tokens=True)
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        if self.return_tokenized_docs:
            retrieved_doc_text = []
            retrieved_doc_title = []

            for b_idx in range(len(docs)):
                for doc_idx in range(n_docs):
                    retrieved_doc_text.append(docs[b_idx]["text"][doc_idx])
                    retrieved_doc_title.append(docs[b_idx]["title"][doc_idx])

            tokenized_docs = self.ctx_encoder_tokenizer(
                retrieved_doc_title,
                retrieved_doc_text,
                truncation=True,
                padding="longest",
                return_tensors=return_tensors,
            )

            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                    "tokenized_doc_ids": tokenized_docs["input_ids"],
                    "tokenized_doc_attention_mask": tokenized_docs["attention_mask"],
                },
                tensor_type=return_tensors,
            )

        else:
            return BatchEncoding(
                {
                    "context_input_ids": context_input_ids,
                    "context_attention_mask": context_attention_mask,
                    "retrieved_doc_embeds": retrieved_doc_embeds,
                    "doc_ids": doc_ids,
                },
                tensor_type=return_tensors,
            )


__all__ = ["RagRetriever"]
