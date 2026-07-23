from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..utils import is_torch_available


if TYPE_CHECKING:
    import torch
    from torch.distributed.tensor import DTensor

if is_torch_available():
    import torch
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
    from torch.distributed.tensor.placement_types import Shard

    if not hasattr(Shard, "local_shard_size_and_offset") and hasattr(Shard, "_local_shard_size_and_offset"):
        Shard.local_shard_size_and_offset = Shard._local_shard_size_and_offset


class DtensorShardOperation:

    def __init__(self, param: DTensor):
        self.device_mesh = param.device_mesh
        self.placements = tuple(param.placements)
        self.param_ndim = param.ndim
        local_shape, offsets = compute_local_shape_and_global_offset(param.shape, self.device_mesh, self.placements)
        self._axis0_offset = offsets[0]
        self._axis0_local_size = local_shape[0]

    def shard_tensor(
        self, source: torch.Tensor, tensor_idx: int | None = None, device=None, dtype=None
    ) -> torch.Tensor | None:
        """Return this rank's local shard of a checkpoint tensor.

        Two layouts (example param shape [N, in, out]):

        - tensor_idx is None: one stacked [N, in, out] tensor;
          slice every sharded dim (including axis 0).
        - tensor_idx given: one [in, out] tensor per expert;
          return None if this rank does not own that expert, else slice
          inner dims only. Surviving pieces are stacked by MergeModulelist
          into this rank's local [n_local, in, out] shard.
        """
        source_shape = list(source.shape) if isinstance(source, torch.Tensor) else source.get_shape()
        dim_placements = [
            (mesh_dim, placement) for mesh_dim, placement in enumerate(self.placements) if hasattr(placement, "dim")
        ]

        if tensor_idx is None:
            if not dim_placements:
                return source[...].to(device=device, dtype=dtype)

            planned_ops_by_dim = [[] for _ in source_shape]
            for mesh_dim, placement in dim_placements:
                sub_mesh = self._get_sub_mesh(mesh_dim)
                rank, world_size = sub_mesh.get_local_rank(), sub_mesh.size()
                dim_idx = self._normalize_param_dim(placement.dim)
                planned_ops_by_dim[dim_idx].append((placement, rank, world_size))

            intervals_by_dim = [[(0, size)] for size in source_shape]
            for dim_idx, planned_ops in enumerate(planned_ops_by_dim):
                intervals = intervals_by_dim[dim_idx]
                for placement, rank, world_size in planned_ops:
                    if placement.is_shard():
                        intervals = self._compute_contiguous_slice(intervals, rank, world_size)
                    else:
                        intervals = self._compute_strided_slice(intervals, rank, world_size, placement.split_factor)
                intervals_by_dim[dim_idx] = intervals

            has_strided_shard = any(not placement.is_shard() for _, placement in dim_placements)
            if has_strided_shard:
                return self._slice_and_cat(source, intervals_by_dim, device, dtype)
            else:
                slice_parts = []
                for intervals in intervals_by_dim:
                    start, end = intervals[0] if len(intervals) > 0 else (0, 0)
                    slice_parts.append(slice(start, end))

                return source[tuple(slice_parts)].to(device=device, dtype=dtype)

        normalized_dim_placements = [
            (mesh_dim, placement, self._normalize_param_dim(placement.dim)) for mesh_dim, placement in dim_placements
        ]

        has_axis0_shard = any(param_dim == 0 for _, _, param_dim in normalized_dim_placements)
        owns_tensor_idx = self._axis0_offset <= tensor_idx < self._axis0_offset + self._axis0_local_size
        if has_axis0_shard and not owns_tensor_idx:
            return None

        planned_ops_by_source_dim = [[] for _ in source_shape]
        for mesh_dim, _, param_dim in normalized_dim_placements:
            if param_dim > 0:
                source_dim = param_dim - 1
                sub_mesh = self._get_sub_mesh(mesh_dim)
                rank, world_size = sub_mesh.get_local_rank(), sub_mesh.size()
                planned_ops_by_source_dim[source_dim].append((rank, world_size))

        intervals_by_source_dim = [[(0, size)] for size in source_shape]
        for source_dim, planned_ops in enumerate(planned_ops_by_source_dim):
            intervals = intervals_by_source_dim[source_dim]
            for rank, world_size in planned_ops:
                intervals = self._compute_contiguous_slice(intervals, rank, world_size)
            intervals_by_source_dim[source_dim] = intervals

        slice_parts = []
        for intervals in intervals_by_source_dim:
            start, end = intervals[0] if intervals else (0, 0)
            slice_parts.append(slice(start, end))

        return source[tuple(slice_parts)].to(device=device, dtype=dtype)

    def _compute_strided_slice(
        self, intervals: list[tuple[int, int]], rank: int, world_size: int, split_factor: int
    ) -> list[tuple[int, int]]:
        local_intervals = []

        for interval_start, interval_end in intervals:
            group_width = math.ceil((interval_end - interval_start) / split_factor)

            for group_idx in range(split_factor):
                group_start = interval_start + group_idx * group_width
                group_end = min(group_start + group_width, interval_end)
                group_len = group_end - group_start

                if group_len > 0:
                    local_shard_size, local_shard_offset = Shard.local_shard_size_and_offset(
                        group_len, world_size, rank
                    )
                    if local_shard_size > 0:
                        shard_start = group_start + local_shard_offset
                        local_intervals.append((shard_start, shard_start + local_shard_size))

        return local_intervals

    def _compute_contiguous_slice(
        self, intervals: list[tuple[int, int]], rank: int, world_size: int
    ) -> list[tuple[int, int]]:

        flat_total_len = sum(end - start for start, end in intervals)
        local_flat_len, local_flat_start = Shard.local_shard_size_and_offset(flat_total_len, world_size, rank)
        local_flat_end = local_flat_start + local_flat_len

        if local_flat_len == 0:
            return []

        if len(intervals) == 1:
            source_start, _ = intervals[0]
            return [(source_start + local_flat_start, source_start + local_flat_end)]

        flat_segments = []  # (interval_flat_start, interval_flat_end, source_start)
        idx = 0
        for source_start, source_end in intervals:
            interval_len = source_end - source_start
            if interval_len > 0:
                flat_segments.append((idx, idx + interval_len, source_start))
                idx += interval_len

        local_intervals = []
        for interval_flat_start, interval_flat_end, source_start in flat_segments:
            overlap_flat_start = max(interval_flat_start, local_flat_start)
            overlap_flat_end = min(interval_flat_end, local_flat_end)
            if overlap_flat_start < overlap_flat_end:
                source_overlap_start = source_start + (overlap_flat_start - interval_flat_start)
                source_overlap_end = source_start + (overlap_flat_end - interval_flat_start)
                local_intervals.append((source_overlap_start, source_overlap_end))

        return local_intervals

    def _slice_and_cat(
        self,
        source: torch.Tensor,
        intervals: list[list[tuple[int, int]]],
        device: torch.device | str | int | None,
        dtype: torch.dtype | None,
    ) -> torch.Tensor:
        multi_interval_dims = [dim_idx for dim_idx, dim_intervals in enumerate(intervals) if len(dim_intervals) > 1]
        if len(multi_interval_dims) > 1:
            raise ValueError("Current shard-on-read only supports disjoint ranges on a single checkpoint dimension.")
        concat_dim = multi_interval_dims[0] if multi_interval_dims else None

        base_slices = []
        for dim_idx, dim_intervals in enumerate(intervals):
            if dim_idx == concat_dim:
                base_slices.append(slice(None))
            else:
                start, end = dim_intervals[0]
                base_slices.append(slice(start, end))

        if concat_dim is None:
            return source[tuple(base_slices)].to(device=device, dtype=dtype)

        base_slices_tuple = tuple(base_slices)
        interval_tensors = []
        for interval_start, interval_end in intervals[concat_dim]:
            interval_slices = (
                *base_slices_tuple[:concat_dim],
                slice(interval_start, interval_end),
                *base_slices_tuple[concat_dim + 1 :],
            )
            interval_tensors.append(source[interval_slices])

        return torch.cat(interval_tensors, dim=concat_dim).to(device=device, dtype=dtype)

    def _get_sub_mesh(self, mesh_dim: int):
        if self.device_mesh.ndim == 1:
            return self.device_mesh
        return self.device_mesh[self.device_mesh.mesh_dim_names[mesh_dim]]

    def _normalize_param_dim(self, dim: int) -> int:
        return dim if dim >= 0 else self.param_ndim + dim


def _dtensor_from_local_like(local_tensor: torch.Tensor, ref: DTensor) -> DTensor:
    """Wrap `local_tensor` as a DTensor that mirrors `ref`'s mesh, placements,
    global shape, and stride."""
    return DTensor.from_local(
        local_tensor.contiguous(),
        ref.device_mesh,
        ref.placements,
        run_check=False,
        shape=ref.shape,
        stride=tuple(ref.stride()),
    )
