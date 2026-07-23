import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..utils.generic import ModelOutput


class PipelineDataset(Dataset):
    def __init__(self, dataset, process, params):
        self.dataset = dataset
        self.process = process
        self.params = params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        processed = self.process(item, **self.params)
        return processed


class PipelineIterator(IterableDataset):
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        Roughly equivalent to

        ```
        for item in loader:
            yield infer(item, **params)
        ```

                Arguments:
                    loader (`torch.utils.data.DataLoader` or `Iterable`):
                        The iterator that will be used to apply `infer` on.
                    infer (any function):
                        The function to apply of each element of `loader`.
                    params (`dict`):
                        The parameters passed to `infer` along with every item
                    loader_batch_size (`int`, *optional*):
                        If specified, the items of `loader` are supposed to come as batch, and are loader_batched here
                        making it roughly behave as


        ```
        for items in loader:
            for i in loader_batch_size:
                item = items[i]
                yield infer(item, **params)
        ```"""
        self.loader = loader
        self.infer = infer
        self.params = params
        if loader_batch_size == 1:
            loader_batch_size = None
        self.loader_batch_size = loader_batch_size

        self._loader_batch_index = None
        self._loader_batch_data = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def loader_batch_item(self):
        pass

    def __next__(self):
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            return self.loader_batch_item()

        item = next(self.iterator)
        processed = self.infer(item, **self.params)
        if self.loader_batch_size is not None:
            if isinstance(processed, torch.Tensor):
                first_tensor = processed
            elif isinstance(processed, tuple):
                first_tensor = processed[0]
            else:
                key = list(processed.keys())[0]
                first_tensor = processed[key]

            if isinstance(first_tensor, list):
                observed_batch_size = len(first_tensor)
            else:
                observed_batch_size = first_tensor.shape[0]
            if 0 < observed_batch_size < self.loader_batch_size:
                self.loader_batch_size = observed_batch_size
            self._loader_batch_data = processed[0] if isinstance(processed, tuple) else processed
            self._loader_batch_index = 0
            return self.loader_batch_item()
        else:
            return processed


class PipelineChunkIterator(PipelineIterator):
    def __init__(self, loader, infer, params, loader_batch_size=None):
        """
        Roughly equivalent to

        ```
        for iterator in loader:
            for item in iterator:
                yield infer(item, **params)
        ```

                Arguments:
                    loader (`torch.utils.data.DataLoader` or `Iterable`):
                        The iterator that will be used to apply `infer` on.
                    infer (any function):
                        The function to apply of each element of `loader`.
                    params (`dict`):
                        The parameters passed to `infer` along with every item
        """
        super().__init__(loader, infer, params)

    def __iter__(self):
        self.iterator = iter(self.loader)
        self.subiterator = None
        return self

    def __next__(self):
        if self.subiterator is None:
            "Subiterator None means we haven't started a `preprocess` iterator. so start it"
            self.subiterator = self.infer(next(self.iterator), **self.params)
        try:
            processed = next(self.subiterator)
        except StopIteration:
            self.subiterator = self.infer(next(self.iterator), **self.params)
            processed = next(self.subiterator)
        return processed


class PipelinePackIterator(PipelineIterator):

    def __iter__(self):
        self.iterator = iter(self.loader)
        return self

    def __next__(self):

        is_last = False
        accumulator = []
        if self._loader_batch_index is not None and self._loader_batch_index < self.loader_batch_size:
            while self._loader_batch_index < self.loader_batch_size:
                item = self.loader_batch_item()
                is_last = item.pop("is_last")
                accumulator.append(item)
                if is_last:
                    return accumulator

        while not is_last:
            processed = self.infer(next(self.iterator), **self.params)
            if self.loader_batch_size is not None:
                if isinstance(processed, torch.Tensor):
                    first_tensor = processed
                else:
                    key = list(processed.keys())[0]
                    first_tensor = processed[key]
                if isinstance(first_tensor, list):
                    observed_batch_size = len(first_tensor)
                else:
                    observed_batch_size = first_tensor.shape[0]
                if 0 < observed_batch_size < self.loader_batch_size:
                    self.loader_batch_size = observed_batch_size
                self._loader_batch_data = processed
                self._loader_batch_index = 0
                while self._loader_batch_index < self.loader_batch_size:
                    item = self.loader_batch_item()
                    is_last = item.pop("is_last")
                    accumulator.append(item)
                    if is_last:
                        return accumulator
            else:
                item = processed
                is_last = item.pop("is_last")
                accumulator.append(item)
        return accumulator


class KeyDataset(Dataset):
    def __init__(self, dataset: Dataset, key: str):
        self.dataset = dataset
        self.key = key

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i][self.key]


class KeyPairDataset(Dataset):
    def __init__(self, dataset: Dataset, key1: str, key2: str):
        self.dataset = dataset
        self.key1 = key1
        self.key2 = key2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return {"text": self.dataset[i][self.key1], "text_pair": self.dataset[i][self.key2]}
