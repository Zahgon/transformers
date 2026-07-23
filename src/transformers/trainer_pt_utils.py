
import contextlib
import copy
import datetime
import io
import json
import math
import os
import re
import sys
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain
from logging import StreamHandler
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import Dataset, IterableDataset, RandomSampler, Sampler
from torch.utils.data.distributed import DistributedSampler

from .integrations.deepspeed import is_deepspeed_zero3_enabled
from .tokenization_utils_base import BatchEncoding
from .utils import (
    is_sagemaker_mp_enabled,
    is_torch_available,
    is_torch_xla_available,
    is_training_run_on_sagemaker,
    logging,
)


if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))

if is_torch_xla_available():
    import torch_xla.runtime as xr

if is_torch_available():
    from torch.optim.lr_scheduler import LRScheduler


logger = logging.get_logger(__name__)


def get_dataloader_sampler(dataloader):
    pass


def atleast_1d(tensor_or_array: torch.Tensor | np.ndarray):
    if isinstance(tensor_or_array, torch.Tensor):
        tensor_or_array = torch.atleast_1d(tensor_or_array)
    else:
        tensor_or_array = np.atleast_1d(tensor_or_array)
    return tensor_or_array


def torch_pad_and_concatenate(tensor1, tensor2, padding_index=-100):
    """Concatenates `tensor1` and `tensor2` on first axis, applying padding on the second if necessary."""
    tensor1 = atleast_1d(tensor1)
    tensor2 = atleast_1d(tensor2)

    if len(tensor1.shape) == 1 or tensor1.shape[1] == tensor2.shape[1]:
        return torch.cat((tensor1, tensor2), dim=0)

    new_shape = (tensor1.shape[0] + tensor2.shape[0], max(tensor1.shape[1], tensor2.shape[1])) + tensor1.shape[2:]

    result = tensor1.new_full(new_shape, padding_index)
    result[: tensor1.shape[0], : tensor1.shape[1]] = tensor1
    result[tensor1.shape[0] :, : tensor2.shape[1]] = tensor2
    return result


def numpy_pad_and_concatenate(array1, array2, padding_index=-100):
    """Concatenates `array1` and `array2` on first axis, applying padding on the second if necessary."""
    array1 = atleast_1d(array1)
    array2 = atleast_1d(array2)

    if len(array1.shape) == 1 or array1.shape[1] == array2.shape[1]:
        return np.concatenate((array1, array2), axis=0)

    new_shape = (array1.shape[0] + array2.shape[0], max(array1.shape[1], array2.shape[1])) + array1.shape[2:]

    result = np.full_like(array1, padding_index, shape=new_shape)
    result[: array1.shape[0], : array1.shape[1]] = array1
    result[array1.shape[0] :, : array2.shape[1]] = array2
    return result


def nested_concat(tensors, new_tensors, padding_index=-100):
    """
    Concat the `new_tensors` to `tensors` on the first dim and pad them on the second if needed. Works for tensors or
    nested list/tuples/dict of tensors.
    """
    if not (isinstance(tensors, torch.Tensor) and isinstance(new_tensors, torch.Tensor)):
        assert type(tensors) is type(new_tensors), (
            f"Expected `tensors` and `new_tensors` to have the same type but found {type(tensors)} and {type(new_tensors)}."
        )
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_concat(t, n, padding_index=padding_index) for t, n in zip(tensors, new_tensors))
    elif isinstance(tensors, torch.Tensor):
        return torch_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    elif isinstance(tensors, Mapping):
        return type(tensors)(
            {k: nested_concat(t, new_tensors[k], padding_index=padding_index) for k, t in tensors.items()}
        )
    elif isinstance(tensors, np.ndarray):
        return numpy_pad_and_concatenate(tensors, new_tensors, padding_index=padding_index)
    else:
        raise TypeError(f"Unsupported type for concatenation: got {type(tensors)}")


def find_batch_size(tensors):
    """
    Find the first dimension of a tensor in a nested list/tuple/dict of tensors.
    """
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            result = find_batch_size(t)
            if result is not None:
                return result
    elif isinstance(tensors, Mapping):
        for value in tensors.values():
            result = find_batch_size(value)
            if result is not None:
                return result
    elif isinstance(tensors, (torch.Tensor, np.ndarray)):
        return tensors.shape[0] if len(tensors.shape) >= 1 else None


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        t = t.to(torch.float32)
    return t.numpy()


def nested_detach(tensors):
    "Detach `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_detach(t) for t in tensors)
    elif isinstance(tensors, Mapping):
        return type(tensors)({k: nested_detach(t) for k, t in tensors.items()})
    return tensors.detach() if isinstance(tensors, torch.Tensor) else tensors


def nested_xla_mesh_reduce(tensors, name):
    pass


def distributed_concat(tensor: Any, num_total_examples: int | None = None) -> Any:
    pass


def nested_gather(tensors, parallel_mode, name=None):
    pass


def is_attention_mask_causal(attention_mask):
    pass


def distributed_broadcast_scalars(
    scalars: list[int | float],
    num_total_examples: int | None = None,
    device: torch.device | None = torch.device("cuda"),
) -> torch.Tensor:
    pass


def reissue_pt_warnings(caught_warnings):
    pass


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    pass


class DistributedSamplerWithLoop(DistributedSampler):

    def __init__(self, dataset, batch_size, **kwargs):
        super().__init__(dataset, **kwargs)
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        remainder = 0 if len(indices) % self.batch_size == 0 else self.batch_size - len(indices) % self.batch_size
        start_remainder = 1 if self.rank < len(self.dataset) % self.num_replicas else 0
        indices += indices[start_remainder : start_remainder + remainder]
        return iter(indices)


class EvalLoopContainer:

    def __init__(self, do_nested_concat: bool = True, padding_index: int = -100):
        self.do_nested_concat = do_nested_concat
        self.padding_index = padding_index
        self.tensors = None
        self.arrays = None

    def add(self, tensors) -> None:
        """Add tensors to the stored objects. If `do_nested_concat=True`, the tensors will be concatenated recursively."""
        if self.tensors is None:
            self.tensors = tensors if self.do_nested_concat else [tensors]
        elif self.do_nested_concat:
            self.tensors = nested_concat(self.tensors, tensors, padding_index=self.padding_index)
        else:
            self.tensors.append(tensors)

    def to_cpu_and_numpy(self) -> None:
        """Move tensors in stored objects to CPU and convert them to numpy arrays."""

        if self.tensors is None:
            return

        new_arrays = nested_numpify(self.tensors)
        if self.arrays is None:
            self.arrays = new_arrays
        elif self.do_nested_concat:
            self.arrays = nested_concat(self.arrays, new_arrays, padding_index=self.padding_index)
        else:
            self.arrays.extend(new_arrays)

        self.tensors = None

    def get_arrays(self):
        """Returns the numpified and moved to CPU stored objects."""
        self.to_cpu_and_numpy()
        return self.arrays


def get_tpu_sampler(dataset: torch.utils.data.Dataset, batch_size: int):
    pass


def nested_new_like(arrays, num_samples, padding_index=-100):
    pass


def expand_like(arrays, new_seq_length, padding_index=-100):
    pass


def nested_truncate(tensors, limit):
    pass


@dataclass
class LabelSmoother:

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels, shift_labels=False, num_items_in_batch=None):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        if num_items_in_batch is None:
            denominator = padding_mask.numel() - padding_mask.long().sum()
        elif torch.is_tensor(num_items_in_batch):
            denominator = num_items_in_batch.to(nll_loss.device)
        else:
            denominator = num_items_in_batch
        nll_loss = nll_loss.sum() / denominator
        smoothed_loss = smoothed_loss.sum() / (denominator * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


def get_length_grouped_indices(lengths, batch_size, mega_batch_mult=None, generator=None):
    pass


class LengthGroupedSampler(Sampler):

    def __init__(
        self,
        batch_size: int,
        dataset: Dataset | None = None,
        lengths: list[int] | None = None,
        model_input_name: str | None = None,
        generator=None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")

        self.batch_size = batch_size
        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if not isinstance(dataset[0], (dict, BatchEncoding)) or model_input_name not in dataset[0]:
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to list[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths
        self.generator = generator

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=self.generator)
        return iter(indices)


class DistributedLengthGroupedSampler(DistributedSampler):

    def __init__(
        self,
        batch_size: int,
        dataset: Dataset | None = None,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
        drop_last: bool = False,
        lengths: list[int] | None = None,
        model_input_name: str | None = None,
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if not isinstance(dataset[0], (dict, BatchEncoding)) or model_input_name not in dataset[0]:
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, DistributedLengthGroupedSampler will be slow. Converting lengths to"
                " list[int]..."
            )
            lengths = lengths.tolist()

        self.lengths = lengths

        if self.drop_last and len(self.lengths) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.lengths) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.lengths) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.seed = seed

    def __iter__(self) -> Iterator:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = get_length_grouped_indices(self.lengths, self.batch_size, generator=g)

        if not self.drop_last:
            indices += indices[: (self.total_size - len(indices))]
        else:
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)


class ShardSampler(Sampler):

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index

        self.total_batch_size = total_batch_size = batch_size * num_processes

        num_batches = len(dataset) // total_batch_size if drop_last else math.ceil(len(dataset) / total_batch_size)
        self.total_num_samples = num_batches * total_batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        while len(indices) < self.total_num_samples:
            indices += indices[: (self.total_num_samples - len(indices))]

        result = []
        for batch_start in range(self.batch_size * self.process_index, self.total_num_samples, self.total_batch_size):
            result += indices[batch_start : batch_start + self.batch_size]

        return iter(result)

    def __len__(self):
        return self.total_num_samples // self.num_processes


class IterableDatasetShard(IterableDataset):

    def __init__(
        self,
        dataset: IterableDataset,
        batch_size: int = 1,
        drop_last: bool = False,
        num_processes: int = 1,
        process_index: int = 0,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_processes = num_processes
        self.process_index = process_index
        self.seed = seed
        self.epoch = 0
        self.num_examples = 0

    def set_epoch(self, epoch):
        pass

    def __iter__(self):
        self.num_examples = 0
        if (
            not hasattr(self.dataset, "set_epoch")
            and hasattr(self.dataset, "generator")
            and isinstance(self.dataset.generator, torch.Generator)
        ):
            self.dataset.generator.manual_seed(self.seed + self.epoch)
        real_batch_size = self.batch_size * self.num_processes
        process_slice = range(self.process_index * self.batch_size, (self.process_index + 1) * self.batch_size)

        first_batch = None
        current_batch = []
        for element in self.dataset:
            self.num_examples += 1
            current_batch.append(element)
            if len(current_batch) == real_batch_size:
                for i in process_slice:
                    yield current_batch[i]
                if first_batch is None:
                    first_batch = current_batch.copy()
                current_batch = []

        if not self.drop_last and len(current_batch) > 0:
            if first_batch is None:
                first_batch = current_batch.copy()
            while len(current_batch) < real_batch_size:
                current_batch += first_batch
            for i in process_slice:
                yield current_batch[i]

    def __len__(self):
        if self.drop_last:
            return (len(self.dataset) // (self.batch_size * self.num_processes)) * self.batch_size
        else:
            return math.ceil(len(self.dataset) / (self.batch_size * self.num_processes)) * self.batch_size


def _secs2timedelta(secs):
    """
    Convert seconds to hh:mm:ss.msec, msecs rounded to 2 decimal places.
    """

    msec = int(abs(secs - int(secs)) * 100)
    return f"{datetime.timedelta(seconds=int(secs))}.{msec:02d}"


def metrics_format(metrics: dict[str, float]) -> dict[str, float]:
    """
    Reformat Trainer metrics values to a human-readable format.

    Args:
        metrics (`dict[str, float]`):
            The metrics returned from train/evaluate/predict

    Returns:
        metrics (`dict[str, float]`): The reformatted metrics
    """

    metrics_copy = metrics.copy()
    for k, v in metrics_copy.items():
        if "_mem_" in k:
            metrics_copy[k] = f"{v >> 20}MB"
        elif "_runtime" in k:
            metrics_copy[k] = _secs2timedelta(v)
        elif k == "total_flos":
            metrics_copy[k] = f"{int(v) >> 30}GF"
        elif isinstance(metrics_copy[k], float):
            metrics_copy[k] = round(v, 4)

    return metrics_copy


def log_metrics(self, split, metrics):
    """
    Log metrics in a specially formatted way.

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`
        metrics (`dict[str, float]`):
            The metrics returned from train/evaluate/predictmetrics: metrics dict

    Notes on memory reports:

    In order to get memory usage report you need to install `psutil`. You can do that with `pip install psutil`.

    Now when this method is run, you will see a report that will include:

    ```
    init_mem_cpu_alloc_delta   =     1301MB
    init_mem_cpu_peaked_delta  =      154MB
    init_mem_gpu_alloc_delta   =      230MB
    init_mem_gpu_peaked_delta  =        0MB
    train_mem_cpu_alloc_delta  =     1345MB
    train_mem_cpu_peaked_delta =        0MB
    train_mem_gpu_alloc_delta  =      693MB
    train_mem_gpu_peaked_delta =        7MB
    ```

    **Understanding the reports:**

    - the first segment, e.g., `train__`, tells you which stage the metrics are for. Reports starting with `init_`
        will be added to the first stage that gets run. So that if only evaluation is run, the memory usage for the
        `__init__` will be reported along with the `eval_` metrics.
    - the third segment, is either `cpu` or `gpu`, tells you whether it's the general RAM or the gpu0 memory
        metric.
    - `*_alloc_delta` - is the difference in the used/allocated memory counter between the end and the start of the
        stage - it can be negative if a function released more memory than it allocated.
    - `*_peaked_delta` - is any extra memory that was consumed and then freed - relative to the current allocated
        memory counter - it is never negative. When you look at the metrics of any stage you add up `alloc_delta` +
        `peaked_delta` and you know how much memory was needed to complete that stage.

    The reporting happens only for process of rank 0 and gpu 0 (if there is a gpu). Typically this is enough since the
    main process does the bulk of work, but it could be not quite so if model parallel is used and then other GPUs may
    use a different amount of gpu memory. This is also not the same under DataParallel where gpu0 may require much more
    memory than the rest since it stores the gradient and optimizer states for all participating GPUs. Perhaps in the
    future these reports will evolve to measure those too.

    The CPU RAM metric measures RSS (Resident Set Size) includes both the memory which is unique to the process and the
    memory shared with other processes. It is important to note that it does not include swapped out memory, so the
    reports could be imprecise.

    The CPU peak memory is measured using a sampling thread. Due to python's GIL it may miss some of the peak memory if
    that thread didn't get a chance to run when the highest memory was used. Therefore this report can be less than
    reality. Using `tracemalloc` would have reported the exact peak memory, but it doesn't report memory allocations
    outside of python. So if some C++ CUDA extension allocated its own memory it won't be reported. And therefore it
    was dropped in favor of the memory sampling approach, which reads the current process memory usage.

    The GPU allocated and peak memory reporting is done with `torch.cuda.memory_allocated()` and
    `torch.cuda.max_memory_allocated()`. This metric reports only "deltas" for pytorch-specific allocations, as
    `torch.cuda` memory management system doesn't track any memory allocated outside of pytorch. For example, the very
    first cuda call typically loads CUDA kernels, which may take from 0.5 to 2GB of GPU memory.

    Note that this tracker doesn't account for memory allocations outside of [`Trainer`]'s `__init__`, `train`,
    `evaluate` and `predict` calls.

    Because `evaluation` calls may happen during `train`, we can't handle nested invocations because
    `torch.cuda.max_memory_allocated` is a single counter, so if it gets reset by a nested eval call, `train`'s tracker
    will report incorrect info. If this [pytorch issue](https://github.com/pytorch/pytorch/issues/16266) gets resolved
    it will be possible to change this class to be re-entrant. Until then we will only track the outer level of
    `train`, `evaluate` and `predict` methods. Which means that if `eval` is called during `train`, it's the latter
    that will account for its memory usage and that of the former.

    This also means that if any other tool that is used along the [`Trainer`] calls
    `torch.cuda.reset_peak_memory_stats`, the gpu peak memory stats could be invalid. And the [`Trainer`] will disrupt
    the normal behavior of any such tools that rely on calling `torch.cuda.reset_peak_memory_stats` themselves.

    For best performance you may want to consider turning the memory profiling off for production runs.
    """
    if not self.is_world_process_zero():
        return

    print(f"***** {split} metrics *****")
    metrics_formatted = metrics_format(metrics)
    k_width = max(len(str(x)) for x in metrics_formatted)
    v_width = max(len(str(x)) for x in metrics_formatted.values())
    for key in sorted(metrics_formatted.keys()):
        print(f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}")


def save_metrics(self, split, metrics, combined=True):
    """
    Save metrics into a json file for that split, e.g. `train_results.json`.

    Under distributed environment this is done only for a process with rank 0.

    Args:
        split (`str`):
            Mode/split name: one of `train`, `eval`, `test`, `all`
        metrics (`dict[str, float]`):
            The metrics returned from train/evaluate/predict
        combined (`bool`, *optional*, defaults to `True`):
            Creates combined metrics by updating `all_results.json` with metrics of this call

    To understand the metrics please read the docstring of [`~Trainer.log_metrics`]. The only difference is that raw
    unformatted numbers are saved in the current method.

    """
    if not self.is_world_process_zero():
        return

    path = os.path.join(self.args.output_dir, f"{split}_results.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)

    if combined:
        path = os.path.join(self.args.output_dir, "all_results.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}

        all_metrics.update(metrics)
        with open(path, "w") as f:
            json.dump(all_metrics, f, indent=4, sort_keys=True)


def save_state(self):
    """
    Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model.

    Under distributed environment this is done only for a process with rank 0.
    """
    if not self.is_world_process_zero():
        return

    path = os.path.join(self.args.output_dir, "trainer_state.json")
    self.state.save_to_json(path)


def get_num_trainable_parameters(self) -> int:
    pass


def get_learning_rates(self) -> list[float]:
    pass


def get_optimizer_group(self, param: str | torch.nn.parameter.Parameter | None = None):
    pass


def get_model_param_count(model, trainable_only=False):
    pass


def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    forbidden_layer_patterns = (
        [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
    )
    result = []
    for name, child in model.named_children():
        child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
        result += [
            f"{name}.{n}"
            for n in child_params
            if not isinstance(child, tuple(forbidden_layer_types))
            and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
        ]
    result += [
        k for k in model._parameters if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
    ]

    return result


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


def remove_dummy_checkpoint(is_main_process, output_dir, filenames):
    if is_main_process:
        for filename in filenames:
            file = os.path.join(output_dir, filename)
            if os.path.isfile(file):
                os.remove(file)


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    @smp.step()
    def smp_forward_backward(model, inputs, gradient_accumulation_steps=1):
        pass

    @smp.step()
    def smp_forward_only(model, inputs):
        return model(**inputs)

    def smp_gather(tensor):
        pass

    def smp_nested_concat(tensor):
        if isinstance(tensor, (list, tuple)):
            return type(tensor)(smp_nested_concat(t) for t in tensor)
        elif isinstance(tensor, dict):
            return type(tensor)({k: smp_nested_concat(v) for k, v in tensor.items()})
        return tensor.detach().concat().cpu()


@dataclass
class AcceleratorConfig:

    split_batches: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the accelerator should split the batches yielded by the dataloaders across the devices. If"
            " `True` the actual batch size used will be the same on any kind of distributed processes, but it must be a"
            " round multiple of the `num_processes` you are using. If `False`, actual batch size used will be the one set"
            " in your script multiplied by the number of processes."
        },
    )
    dispatch_batches: bool | None = field(
        default=None,
        metadata={
            "help": "If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process"
            " and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose"
            " underlying dataset is an `IterableDataslet`, `False` otherwise."
        },
    )
    even_batches: bool = field(
        default=True,
        metadata={
            "help": "If set to `True`, in cases where the total batch size across all processes does not exactly divide the"
            " dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among"
            " all workers."
        },
    )
    use_seedable_sampler: bool = field(
        default=True,
        metadata={
            "help": "Whether or not use a fully seedable random sampler ([`accelerate.data_loader.SeedableRandomSampler`])."
            "Ensures training results are fully reproducible using a different sampling technique. "
            "While seed-to-seed results may differ, on average the differences are negligible when using"
            "multiple different seeds to compare. Should also be ran with [`~utils.set_seed`] for the best results."
        },
    )

    non_blocking: bool = field(
        default=False,
        metadata={
            "help": "Whether to use non-blocking CUDA calls to help minimize synchronization during "
            "distributed training with prepared `DataLoader` inputs being moved to device. "
            "Best if used with `pin_memory=True` in the `TrainingArguments`. Requires accelerate "
            "v0.30.0."
        },
    )

    gradient_accumulation_kwargs: dict | None = field(
        default=None,
        metadata={
            "help": "Additional kwargs to configure gradient accumulation, see [`accelerate.utils.GradientAccumulationPlugin`]. "
            "Any of the following (optional) keys are acceptable: "
            "  num_steps (`int`): Will take precedence over [`~.TrainingArguments.gradient_accumulation_steps`] if "
            "    the latter is set to 1, otherwise an exception will be raised. "
            "  sync_each_batch (`bool`): Whether to synchronize the gradients at each data batch. "
            "    The [`accelerate.utils.GradientAccumulationPlugin`] default is `False`."
        },
    )
    use_configured_state: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use a pre-configured `AcceleratorState` or `PartialState` defined before calling `TrainingArguments`."
            "If `True`, an `Accelerator` or `PartialState` must be initialized. May lead to issues using sweeps or hyperparameter tuning."
        },
    )

    @classmethod
    def from_json_file(cls, json_file):
        open_file = io.open if os.path.exists(json_file) else open
        with open_file(json_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        extra_keys = sorted(key for key in config_dict if key not in cls.__dataclass_fields__)
        if len(extra_keys) > 0:
            raise ValueError(
                f"The config file at {json_file} had unknown keys ({extra_keys}), please try upgrading your `transformers`"
                " version or fix (and potentially remove these keys) from your config file."
            )
        return cls(**config_dict)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)

    def pop(self, key, default=None):
        return self.__dict__.pop(key, default)


class LayerWiseDummyOptimizer(torch.optim.Optimizer):

    def __init__(self, optimizer_dict=None, **kwargs):
        dummy_tensor = torch.randn(1, 1)
        self.optimizer_dict = optimizer_dict
        super().__init__([dummy_tensor], {"lr": kwargs.get("lr", 1e-03)})

    def zero_grad(self, set_to_none: bool = True) -> None:
        pass

    def step(self, closure=None) -> float | None:
        pass


class LayerWiseDummyScheduler(LRScheduler):

    def __init__(self, *args, **kwargs):
        self.default_lr = kwargs["lr"]
        optimizer = LayerWiseDummyOptimizer(**kwargs)
        last_epoch = -1
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        pass

    def _get_closed_form_lr(self):
        pass


def set_rng_state_for_device(device_name, device_module, checkpoint_rng_state, is_distributed):
    pass


def safe_globals():
    pass
