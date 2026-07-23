
from collections.abc import Callable
from functools import partial
from typing import Any, TypeVar, overload

import torch
import torch.nn as nn
import torch.types


def add(m1: torch.Tensor, m2: torch.Tensor, inplace: bool) -> torch.Tensor:
    if not inplace:
        m1 = m1 + m2
    else:
        m1 += m2

    return m1


def permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int) -> torch.Tensor:
    return t.reshape(t.shape[:-no_dims] + (-1,))


def masked_mean(mask: torch.Tensor, value: torch.Tensor, dim: int, eps: float = 1e-4) -> torch.Tensor:
    pass


def pts_to_distogram(
    pts: torch.Tensor, min_bin: torch.types.Number = 2.3125, max_bin: torch.types.Number = 21.6875, no_bins: int = 64
) -> torch.Tensor:
    pass


def dict_multimap(fn: Callable[[list], Any], dicts: list[dict]) -> dict:
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def one_hot(x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


def batched_gather(data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0) -> torch.Tensor:
    pass


T = TypeVar("T")


def dict_map(
    fn: Callable[[T], Any], dic: dict[Any, dict | list | tuple | T], leaf_type: type[T]
) -> dict[Any, dict | list | tuple | Any]:
    pass


@overload
def tree_map(fn: Callable[[T], Any], tree: T, leaf_type: type[T]) -> Any: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: dict, leaf_type: type[T]) -> dict: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: list, leaf_type: type[T]) -> list: ...


@overload
def tree_map(fn: Callable[[T], Any], tree: tuple, leaf_type: type[T]) -> tuple: ...


def tree_map(fn, tree, leaf_type):
    pass


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)
