import logging
import math
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import Any

import torch

from .tensor_utils import tensor_tree_map, tree_map


def _fetch_dims(tree: dict | list | tuple | torch.Tensor) -> list[tuple[int, ...]]:
    shapes = []
    if isinstance(tree, dict):
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    elif isinstance(tree, (list, tuple)):
        for t in tree:
            shapes.extend(_fetch_dims(t))
    elif isinstance(tree, torch.Tensor):
        shapes.append(tree.shape)
    else:
        raise TypeError("Not supported")

    return shapes


@torch.jit.ignore
def _flat_idx_to_idx(flat_idx: int, dims: tuple[int, ...]) -> tuple[int, ...]:
    pass


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: Sequence[int],
    start_edges: Sequence[bool] | None = None,
    end_edges: Sequence[bool] | None = None,
) -> list[tuple[slice, ...]]:
    """
    Produces an ordered sequence of tensor slices that, when used in sequence on a tensor with shape dims, yields
    tensors that contain every leaf in the contiguous range [start, end]. Care is taken to yield a short sequence of
    slices, and perhaps even the shortest possible (I'm pretty sure it's the latter).

    end is INCLUSIVE.
    """

    def reduce_edge_list(l: list[bool]) -> None:
        tally = True
        for i in range(len(l)):
            reversed_idx = -1 * (i + 1)
            l[reversed_idx] &= tally
            tally = l[reversed_idx]

    if start_edges is None:
        start_edges = [s == 0 for s in start]
        reduce_edge_list(start_edges)
    if end_edges is None:
        end_edges = [e == (d - 1) for e, d in zip(end, dims)]
        reduce_edge_list(end_edges)

    if len(start) == 0:
        return [()]
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]

    slices: list[tuple[slice, ...]] = []
    path_list: list[slice] = []

    for s, e in zip(start, end):
        if s == e:
            path_list.append(slice(s, s + 1))
        else:
            break

    path: tuple[slice, ...] = tuple(path_list)
    divergence_idx = len(path)

    if divergence_idx == len(dims):
        return [path]

    def upper() -> tuple[tuple[slice, ...], ...]:
        assert start_edges is not None
        assert end_edges is not None

        sdi = start[divergence_idx]
        return tuple(
            path + (slice(sdi, sdi + 1),) + s
            for s in _get_minimal_slice_set(
                start[divergence_idx + 1 :],
                [d - 1 for d in dims[divergence_idx + 1 :]],
                dims[divergence_idx + 1 :],
                start_edges=start_edges[divergence_idx + 1 :],
                end_edges=[True for _ in end_edges[divergence_idx + 1 :]],
            )
        )

    def lower() -> tuple[tuple[slice, ...], ...]:
        assert start_edges is not None
        assert end_edges is not None

        edi = end[divergence_idx]
        return tuple(
            path + (slice(edi, edi + 1),) + s
            for s in _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1 :]],
                end[divergence_idx + 1 :],
                dims[divergence_idx + 1 :],
                start_edges=[True for _ in start_edges[divergence_idx + 1 :]],
                end_edges=end_edges[divergence_idx + 1 :],
            )
        )

    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),))
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx]),))
        slices.extend(lower())

    return slices


@torch.jit.ignore
def _chunk_slice(t: torch.Tensor, flat_start: int, flat_end: int, no_batch_dims: int) -> torch.Tensor:
    pass


def chunk_layer(
    layer: Callable,
    inputs: dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
) -> Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and inputs are assumed to be simple "pytrees," consisting only of (arbitrarily nested) lists, tuples,
    and dicts with torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded inputs. All leaves must be tensors and must share the same batch
            dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch dimensions are specified, a "sub-batch" is defined
            as a single indexing of all batch dimensions simultaneously (s.t. the number of sub-batches is the product
            of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary in most cases, and is ever so slightly
            slower than the default setting.
    Returns:
        The reassembled output of the layer on the inputs.
    """
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple(max(s) for s in zip(*initial_dims))

    def _prep_inputs(t: torch.Tensor) -> torch.Tensor:
        pass

    prepped_inputs: dict[str, Any] = tensor_tree_map(_prep_inputs, inputs)
    prepped_outputs = None
    if _out is not None:
        prepped_outputs = tensor_tree_map(lambda t: t.view([-1] + list(t.shape[no_batch_dims:])), _out)

    flat_batch_dim = 1
    for d in orig_batch_dims:
        flat_batch_dim *= d

    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)

    def _select_chunk(t: torch.Tensor) -> torch.Tensor:
        pass

    i = 0
    out = prepped_outputs
    for _ in range(no_chunks):
        if not low_mem:
            select_chunk = _select_chunk
        else:
            select_chunk = partial(
                _chunk_slice,
                flat_start=i,
                flat_end=min(flat_batch_dim, i + chunk_size),
                no_batch_dims=len(orig_batch_dims),
            )

        chunks: dict[str, Any] = tensor_tree_map(select_chunk, prepped_inputs)

        output_chunk = layer(**chunks)

        if out is None:
            out = tensor_tree_map(lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:]), output_chunk)

        if isinstance(output_chunk, dict):

            def assign(d1: dict, d2: dict) -> None:
                for k, v in d1.items():
                    if isinstance(v, dict):
                        assign(v, d2[k])
                    else:
                        if _add_into_out:
                            v[i : i + chunk_size] += d2[k]
                        else:
                            v[i : i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif isinstance(output_chunk, tuple):
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i : i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif isinstance(output_chunk, torch.Tensor):
            if _add_into_out:
                out[i : i + chunk_size] += output_chunk
            else:
                out[i : i + chunk_size] = output_chunk
        else:
            raise TypeError("Not supported")

        i += chunk_size

    out = tensor_tree_map(lambda t: t.view(orig_batch_dims + t.shape[1:]), out)

    return out


class ChunkSizeTuner:
    def __init__(
        self,
        max_chunk_size: int = 512,
    ):
        self.max_chunk_size = max_chunk_size
        self.cached_chunk_size: int | None = None
        self.cached_arg_data: tuple | None = None

    def _determine_favorable_chunk_size(self, fn: Callable, args: tuple, min_chunk_size: int) -> int:
        pass

    def _compare_arg_caches(self, ac1: Iterable, ac2: Iterable) -> bool:
        pass

    def tune_chunk_size(
        self,
        representative_fn: Callable,
        args: tuple,
        min_chunk_size: int,
    ) -> int:
        pass
