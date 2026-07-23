
from __future__ import annotations

import math
import os
import re
import traceback
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from itertools import chain
from typing import TYPE_CHECKING, Any

import torch

from .distributed.sharding_utils import DtensorShardOperation, _dtensor_from_local_like
from .integrations.accelerate import get_device, offload_weight
from .integrations.tensor_parallel import ALL_PARALLEL_STYLES
from .utils import is_env_variable_true
from .utils.loading_report import LoadStateDictInfo
from .utils.logging import get_logger, tqdm


_torch_distributed_available = torch.distributed.is_available()
if _torch_distributed_available:
    from torch.distributed.tensor import DTensor

if TYPE_CHECKING:
    from .integrations.tensor_parallel import TensorParallelLayer
    from .modeling_utils import LoadStateDictConfig, PreTrainedModel
    from .quantizers import HfQuantizer


logger = get_logger(__name__)


def build_glob_alternation(
    globs: list[WeightRenaming | WeightConverter | str],
) -> tuple[re.Pattern, dict[str, str], dict[str, str]]:
    """
    Build a single alternation regex with one named group per glob.
    """
    src_group_to_glob: dict[str, str] = {}
    tgt_group_to_glob: dict[str, str] = {}
    branches: list[str] = []
    i = 0
    for glob in globs:
        if isinstance(glob, (WeightRenaming, WeightConverter)):
            for src in glob.source_patterns:
                group_name = f"g{i}"
                src_group_to_glob[group_name] = src
                i += 1
                body = src.replace("*", r".*")
                branches.append(f"(?P<{group_name}>{body})")
                tgt_group_to_glob[group_name] = glob.target_patterns[0]  # we index with the first target
        else:
            group_name = f"g{i}"
            src_group_to_glob[group_name] = glob
            i += 1
            body = glob
            body = body.replace("*", r".*")
            branches.append(f"(?P<{group_name}>{body})")
            tgt_group_to_glob[group_name] = glob

    alternation = re.compile("|".join(branches))
    return alternation, src_group_to_glob, tgt_group_to_glob


class ConversionOps(ABC):

    def __repr__(self):
        if hasattr(self, "dim"):
            return f"{self.__class__.__name__}(dim={self.dim})"
        else:
            return f"{self.__class__.__name__}"

    @abstractmethod
    def convert(
        self, input_dict: dict[str, Any], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, list[torch.Tensor]]:
        raise NotImplementedError

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError


class _IdentityOp(ConversionOps):

    def convert(self, input_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        return input_dict


class Chunk(ConversionOps):

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        targets = target_patterns
        sizes = len(targets)
        chunks = tuple(chunk.contiguous() for chunk in torch.chunk(tensor, sizes, dim=self.dim))
        if len(input_dict) > 1 or len(target_patterns) == 1 or len(chunks) != len(target_patterns):
            raise ValueError(f"Failed to convert {kwargs.get('full_layer_name')}")
        return dict(zip(targets, chunks))

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class Concatenate(ConversionOps):

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = self.get_target_pattern(target_patterns)
        all_tensors = []
        for source_pattern in source_patterns:
            if source_pattern not in input_dict:
                continue
            tensors = input_dict.pop(source_pattern)
            if isinstance(tensors, list):
                all_tensors.extend(tensors)
            else:
                all_tensors.append(tensors)
        return {target_pattern: torch.cat(all_tensors, dim=self.dim)}

    def get_target_pattern(self, target_patterns: list[str]) -> str:
        if len(target_patterns) > 1:
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class Interleave(ConversionOps):

    def __init__(self, dim: int = 0, inverse: bool = False):
        self.dim = dim
        self.inverse = inverse

    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        tensor = next(iter(input_dict.values()))
        tensor = tensor[0] if isinstance(tensor, list) else tensor

        shape = list(tensor.shape)
        if self.inverse:
            shape[self.dim : self.dim + 1] = [2, shape[self.dim] // 2]
        else:
            shape[self.dim : self.dim + 1] = [shape[self.dim] // 2, 2]

        tensor = tensor.reshape(shape).transpose(self.dim, self.dim + 1).reshape(tensor.shape).contiguous()
        return {target_patterns[0]: tensor}

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class MergeModulelist(ConversionOps):

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        input_size = len(input_dict)
        merged: dict[str, torch.Tensor] = {}
        for source_pattern in list(input_dict.keys()):
            tensors = input_dict.pop(source_pattern)
            target_pattern = self.get_target_pattern(input_size, source_pattern, target_patterns)
            if isinstance(tensors, torch.Tensor):
                merged[target_pattern] = tensors
            else:
                merged[target_pattern] = torch.stack(tensors, dim=self.dim)
        return merged

    def get_target_pattern(self, input_size: int, source_pattern: str, target_patterns: list[str]) -> str:
        if input_size == 1:
            if len(target_patterns) == 1:
                return target_patterns[0]
            else:
                raise ValueError("Undefined Operation encountered!")
        else:
            return source_pattern

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class SplitModulelist(ConversionOps):

    def __init__(self, dim: int = 0):
        self.dim = dim

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        all_tensors = {}
        for source_pattern, tensors in input_dict.items():
            tensor = tensors[0] if isinstance(tensors, list) else tensors
            sizes = tensor.size(self.dim)
            targets = self.get_target_patterns(input_dict, source_pattern, target_patterns, sizes)
            chunks = torch.chunk(tensor, sizes, dim=self.dim)
            all_tensors.update({target: chunk.squeeze() for target, chunk in zip(targets, chunks)})
        return all_tensors

    def get_target_patterns(
        self, input_dict: dict, source_pattern: str, target_patterns: list[str], sizes: int
    ) -> list[str]:
        if len(input_dict) == 1:
            if len(target_patterns) == 1:
                return [target_patterns[0].replace("*", f"{i}") for i in range(sizes)]
            else:
                raise ValueError("Undefined Operation encountered!")
        else:
            return [source_pattern.replace("*", f"{i}") for i in range(sizes)]

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class Transpose(ConversionOps):

    def __init__(self, dim0: int = 0, dim1: int = 1, check_dims: bool = False):
        self.dim0 = dim0
        self.dim1 = dim1
        self.check_dims = check_dims

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        target_pattern = self.get_target_pattern(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        if not self.check_dims:
            return {target_pattern: torch.transpose(tensor, dim0=self.dim0, dim1=self.dim1).contiguous()}
        else:
            expected_shape = kwargs["model"].get_parameter(kwargs["full_layer_name"]).shape
            if tensor.shape == expected_shape:
                return {target_pattern: tensor}
            else:
                return {target_pattern: torch.transpose(tensor, dim0=self.dim0, dim1=self.dim1).contiguous()}

    def get_target_pattern(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str]
    ) -> str:
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            else:
                raise ValueError("Undefined Operation encountered!")
        else:
            return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class Conv3dToLinear(ConversionOps):

    def __init__(self, in_channels: int, kernel_size: tuple[int, int, int]):
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    @staticmethod
    def _get_target_pattern(
        input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str]
    ) -> str:
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            else:
                raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        target_pattern = self._get_target_pattern(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors

        if tensor.ndim == 5:
            tensor = tensor.reshape(tensor.shape[0], -1).contiguous()
        elif tensor.ndim != 2:
            raise ValueError(f"Conv3dToLinear expects a 5D or 2D tensor, got {tensor.ndim}D")

        return {target_pattern: tensor}

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class LinearToConv3d(ConversionOps):

    def __init__(self, in_channels: int, kernel_size: tuple[int, int, int]):
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    @torch.no_grad
    def convert(
        self, input_dict: dict[str, torch.Tensor], source_patterns: list[str], target_patterns: list[str], **kwargs
    ) -> dict[str, torch.Tensor]:
        target_pattern = Conv3dToLinear._get_target_pattern(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors

        target_shape = (tensor.shape[0], self.in_channels, *self.kernel_size)
        if tensor.numel() != math.prod(target_shape):
            raise ValueError(f"Cannot reshape tensor with shape {tensor.shape} into {target_shape}")

        return {target_pattern: tensor.reshape(target_shape).contiguous()}

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class PermuteForRope(ConversionOps):

    def __init__(self):
        pass

    def _apply(self, tensor: torch.Tensor) -> torch.Tensor:
        dim1, dim2 = tensor.shape
        n_heads = self.config.getattr("num_attention_heads", 1)

        tensor = tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
        tensor = tensor.transpose(1, 2).reshape(dim1, dim2)
        return tensor

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        self.config = config
        output: dict[str, list[torch.Tensor]] = {}
        for key, tensors in input_dict.items():
            if len(tensors) != 1:
                raise ValueError("PermuteForRope expects a single tensor per key.")
            output[key] = [self._apply(tensors[0])]
        return output

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class VisionFuseAndPermuteForRope(ConversionOps):

    def __init__(self, dim: int = 0, permute_layer_names: list[str] | None = None):
        self.dim = dim
        self.permute_layer_names = permute_layer_names or []

    def _apply_permutation(self, tensor: torch.Tensor) -> torch.Tensor:
        dim0 = tensor.shape[0]
        n_heads = getattr(self.config.vision_config, "num_attention_heads", 1)
        half_head = dim0 // n_heads // 2

        if tensor.ndim == 2:
            tensor = tensor.view(n_heads, 2, half_head, tensor.shape[1])
            tensor = tensor.transpose(1, 2).reshape(dim0, tensor.shape[-1])
        elif tensor.ndim == 1:
            tensor = tensor.view(n_heads, 2, half_head)
            tensor = tensor.transpose(1, 2).reshape(dim0)
        return tensor

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        self.config = config
        target_pattern = self.get_target_pattern(target_patterns)

        all_tensors = []
        for source_pattern in source_patterns:
            tensors = input_dict[source_pattern][0]
            if any(name in source_pattern for name in self.permute_layer_names) and tensors.ndim == 2:
                tensors = self._apply_permutation(tensors)
            all_tensors.append(tensors)

        return {target_pattern: torch.cat(all_tensors, dim=self.dim)}

    def get_target_pattern(self, target_patterns: list[str]) -> str:
        if len(target_patterns) > 1:
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class VisionUnfuseAndPermuteForRope(ConversionOps):

    def __init__(self, dim: int = 0, permute_layer_names: list[str] | None = None):
        self.dim = dim
        self.permute_layer_names = permute_layer_names or []

    def _apply_permutation(self, tensor: torch.Tensor) -> torch.Tensor:
        dim0 = tensor.shape[0]
        n_heads = getattr(self.config.vision_config, "num_attention_heads", 1)
        half_head = dim0 // n_heads // 2

        if tensor.ndim == 2:
            tensor = tensor.view(n_heads, half_head, 2, tensor.shape[1])
            tensor = tensor.transpose(1, 2).reshape(dim0, tensor.shape[-1])
        elif tensor.ndim == 1:
            tensor = tensor.view(n_heads, half_head, 2)
            tensor = tensor.transpose(1, 2).reshape(dim0)
        return tensor

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        self.config = config

        tensor = next(iter(input_dict.values()))[0]
        targets = self.get_target_patterns(input_dict, target_patterns)
        chunks = torch.chunk(tensor, len(targets), dim=self.dim)

        output: dict[str, torch.Tensor] = dict(zip(targets, chunks))
        for key, value in output.items():
            if any(name in key for name in self.permute_layer_names):
                output[key] = self._apply_permutation(value)
        return output

    def get_target_patterns(self, input_dict: dict, target_patterns: list[str]) -> list[str]:
        if len(input_dict) > 1 or len(target_patterns) == 1:
            raise ValueError("Undefined Operation encountered!")
        return target_patterns

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class ErnieFuseAndSplitTextVisionExperts(ConversionOps):

    def __init__(self, stack_dim: int = 0, concat_dim: int = 1):
        self.stack_dim = stack_dim
        self.concat_dim = concat_dim

    def split_list_into_chunks(self, tensor_list: list[torch.Tensor], chunks: int = 2):
        split_size = math.ceil(len(tensor_list) / chunks)  # best effort split size
        return [tensor_list[i * split_size : (i + 1) * split_size] for i in range(chunks)]

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        valid_keys = input_dict.keys()
        split_and_fused = defaultdict(list)
        for key in source_patterns:
            if key not in valid_keys:
                raise ValueError(
                    f"Expected pattern {key} in collected tensors but only found tensors for: {valid_keys}"
                )

            tensors = input_dict.get(key, [])
            split_tensor_lists = self.split_list_into_chunks(tensors, chunks=len(target_patterns))
            stacked_tensors = (torch.stack(tensor_group, dim=self.stack_dim) for tensor_group in split_tensor_lists)
            for idx, tensor_group in enumerate(stacked_tensors):
                split_and_fused[target_patterns[idx]].append(tensor_group)

        for k, v in split_and_fused.items():
            split_and_fused[k] = torch.cat(v, dim=self.concat_dim)

        return split_and_fused

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class ErnieSplitAndDecoupleTextVisionExperts(ConversionOps):

    def __init__(self, stack_dim: int = 0, concat_dim: int = 1):
        self.stack_dim = stack_dim
        self.concat_dim = concat_dim

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        fused_modules = len(target_patterns)
        valid_keys = input_dict.keys()
        split_tensors = []
        for key in source_patterns:
            if key not in valid_keys:
                raise ValueError(
                    f"Expected pattern {key} in collected tensors but only found tensors for: {valid_keys}"
                )

            split_tensors.append(input_dict[key][0].chunk(fused_modules, dim=self.concat_dim))

        decoupled = {}
        for idx, key in enumerate(target_patterns):
            tensor_groups = [
                list(torch.unbind(tensor_group[idx], dim=self.stack_dim)) for tensor_group in split_tensors
            ]
            tensor_list = list(chain.from_iterable(tensor_groups))
            targets = [key.replace("*", f"{i}") for i in range(len(tensor_list))]
            decoupled |= dict(zip(targets, tensor_list))

        return decoupled

    @property
    def reverse_op(self) -> ConversionOps:
        pass


def process_target_pattern(pattern: str) -> tuple[str, str | None]:
    """
    Process a target pattern for reverse mapping (when targets become sources).

    This handles several edge cases in checkpoint conversion mappings:
    - Removes `^` prefix and `$` suffix (start/end of string anchors)
    - Removes negative lookahead/lookbehind assertions
    - Detects capturing groups and replaces them with `\\1` backreference

    Args:
        pattern: The target pattern to process for reverse mapping.

    Returns:
        A tuple of (processed_pattern, captured_group) where captured_group is
        the original capturing group found (e.g., "(encoder|decoder)") or None.
    """
    pattern = pattern.removeprefix("^")
    pattern = pattern.removesuffix("$")
    pattern = re.sub(r"\(\?.+?\)?\)", "", pattern)
    pattern = pattern.replace(r"\.", ".")
    capturing_group_match = re.search(r"\(.+?\)", pattern)
    captured_group = None
    if capturing_group_match:
        captured_group = capturing_group_match.group(0)
        pattern = pattern.replace(captured_group, r"\1", 1)
    return pattern, captured_group


def process_source_pattern(source_pattern: str, target_pattern: str) -> str:
    """
    Process a source pattern for reverse mapping (when sources become targets).
    This is useful because usually if the original source (so now the target in reverse mode) had a `^` or `$`
    to restrict to start/end of string, we should do the same in reverse mode. This is why this method in conditioned
    on the target pattern, we want to do it only for pairs (source, target) when the original source (so the current target
    in reverse mode) had it.
    """
    if target_pattern.startswith("^"):
        source_pattern = f"^{source_pattern}" if not source_pattern.startswith("^") else source_pattern
    if target_pattern.endswith("$"):
        source_pattern = f"{source_pattern}$" if not source_pattern.endswith("$") else source_pattern

    return source_pattern


class WeightTransform:
    __slots__ = (
        "source_patterns",
        "target_patterns",
        "compiled_sources",
        "distributed_operation",
        "quantization_operation",
        "collected_tensors",
        "layer_targets",
        "_original_source_patterns",
        "_original_target_patterns",
        "_was_used",
        "scope_prefix",
        "base_model_prefix",
    )

    def __init__(self, source_patterns: str | list[str], target_patterns: str | list[str]):
        self.source_patterns: list[str] = source_patterns
        self.target_patterns: list[str] = target_patterns
        self._original_source_patterns = self.source_patterns.copy()
        self._original_target_patterns = self.target_patterns.copy()

        self.distributed_operation: Any = None
        self.quantization_operation: ConversionOps | None = None
        self.collected_tensors: dict[str, list[Future]] = defaultdict(list)
        self.layer_targets: dict[str, set[str]] = defaultdict(set)

        self._was_used = False

        self.scope_prefix: str | None = None
        self.base_model_prefix: str | None = None


        target_capturing_groups: list[str] = []
        for i, pattern in enumerate(self.target_patterns):
            self.target_patterns[i], captured_group = process_target_pattern(pattern)
            if captured_group is not None:
                target_capturing_groups.append(captured_group)

        unique_capturing_groups = set(target_capturing_groups)
        if len(unique_capturing_groups) > 1:
            raise ValueError(
                f"Multiple different capturing groups found in target_patterns: {unique_capturing_groups}. "
                f"All target patterns must use the same capturing group pattern."
            )
        unique_capturing_group = unique_capturing_groups.pop() if unique_capturing_groups else None

        for i, pattern in enumerate(self.source_patterns):
            if r"\1" in pattern:
                if unique_capturing_group is None:
                    raise ValueError(
                        f"Source pattern '{pattern}' contains \\1 backreference, but no capturing groups "
                        f"found in target_patterns."
                    )
                pattern = pattern.replace(r"\1", unique_capturing_group, 1)
            if len(self.source_patterns) == len(self.target_patterns):
                pattern = process_source_pattern(pattern, self._original_target_patterns[i])
            self.source_patterns[i] = pattern

        branches = []
        for i, source_pattern in enumerate(self.source_patterns):
            group_name = f"g{i}"
            pattern = source_pattern.replace(".*.", r"\..*\.")
            branches.append(f"(?P<{group_name}>{pattern})")
        self.compiled_sources = re.compile("|".join(branches))

    def __repr__(self):
        return f"{self.__class__.__name__}(source_patterns={self.source_patterns}, target_patterns={self.target_patterns})"

    def __setattr__(self, name, value):
        if name in ("source_patterns", "target_patterns"):
            if hasattr(self, name):
                raise ValueError(f"Cannot assign to field {name}, you should create a new instance")
            elif isinstance(value, str):
                value = [value]
        object.__setattr__(self, name, value)

    def add_tensor(self, target_key: str, source_key: str, source_pattern: str, future: Future):
        self.collected_tensors[source_pattern].append(future)
        self.layer_targets[target_key].add(source_key)

    def _scoped_match(self, source_key: str) -> tuple[str | None, str, re.Match[str]] | None:
        """
        Strip `scope_prefix` (if any) from `source_key`, then match `compiled_sources` against the
        remaining suffix.

        Returns `(prefix_dot, key_to_match, match_object)` on match, else `None`. `prefix_dot` is
        the prefix consumed from `source_key`: either `f"{scope_prefix}."` or that same string with
        one `base_model_prefix` level stripped or prepended when the former didn't match.
        `None` when `scope_prefix` is unset.
        """
        key_to_match = source_key
        prefix = None
        if self.scope_prefix is not None:
            scope_prefix = f"{self.scope_prefix}." if self.scope_prefix != "" else ""
            base_model_prefix = f"{self.base_model_prefix}." if self.base_model_prefix != "" else ""
            if source_key.startswith(base_model_prefix + scope_prefix):
                prefix = base_model_prefix + scope_prefix
            elif source_key.startswith(scope_prefix):
                prefix = scope_prefix
            else:
                return None
            key_to_match = source_key.removeprefix(prefix)

        match_object = self.compiled_sources.search(key_to_match)
        if match_object is None:
            return None
        return (prefix, key_to_match, match_object)

    def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
        """
        Return a tuple (renamed_key, source_pattern_producing_the_match).
        Try renaming `source_key` according to the source and target patterns of the current WeightTransform.
        In case of a one-to-many transform, i.e. we have several target patterns, the matching source pattern
        will be replaced by the first of all the target patterns (they are then correctly expanded in the Operations).
        """
        matched = self._scoped_match(source_key)
        if matched is None:
            return source_key, None

        prefix_dot, key_to_match, match_object = matched

        self._was_used = True

        matching_group_name = next(name for name, val in match_object.groupdict().items() if val is not None)
        source_pattern_that_matched = self.source_patterns[int(matching_group_name[1:])]
        replacement = self.target_patterns[0]
        if re.search(r"\\\d", replacement):
            group_start = self.compiled_sources.groupindex[matching_group_name]
            replacement = re.sub(
                r"\\(\d+)",
                lambda m: match_object.group(group_start + int(m.group(1))),
                replacement,
            )
        renamed_key = key_to_match.replace(match_object.group(0), replacement, 1)
        if prefix_dot is not None:
            renamed_key = prefix_dot + renamed_key
        return renamed_key, source_pattern_that_matched

    def reverse_transform(self) -> WeightTransform:
        """Reverse the current `WeightTransform` instance, to be able to save with the opposite weight transformations."""
        if self.quantization_operation is not None:
            raise ValueError("Cannot reverse the transform with TP or quantization")

        kwargs = {}
        if hasattr(self, "operations"):
            kwargs["operations"] = [op.reverse_op for op in self.operations[::-1]]

        reverse_transform = self.__class__(
            source_patterns=self._original_target_patterns, target_patterns=self._original_source_patterns, **kwargs
        )
        reverse_transform.scope_prefix = self.scope_prefix
        reverse_transform.base_model_prefix = self.base_model_prefix
        return reverse_transform

    def materialize_tensors(self) -> dict[str, list[torch.Tensor]]:
        """
        Materialize all the tensors that were saved in `self.collected_tensors`. This function removes them from the
        internal attribute to avoid keeping them in memory during the different `self.convert` operations, and return
        a new dictionary (otherwise we use more memory than needed during loading).

        We basically have 3 cases here:
        - async loading (default): the tensors are Future instances that we need to wait for
        - sync loading: the tensors are Callable, we need to call the Callable to actually load them from disk
        - saving: the tensors are already torch.Tensor instances (the existing model weights)
        """
        collected_tensors = {}
        for key in list(self.collected_tensors.keys()):
            tensors = self.collected_tensors.pop(key)
            if isinstance(tensors[0], Future):
                tensors = [future.result() for future in tensors if future.result() is not None]
            elif callable(tensors[0]):
                tensors = [func() for func in tensors]
                tensors = [tensor for tensor in tensors if tensor is not None]
            collected_tensors[key] = tensors

        return collected_tensors

    def was_used(self) -> bool:
        """
        Return whether the current Transform matched any weights during loading/saving. This is needed as some
        weight renaming transforms are not bijective, i.e. if we drop/add full parts of a name with PrefixChange, we
        lose some information that we cannot get back if we don't know if the Transform was used before already (say we
        have a prefix to drop, we need to know whether the checkpoints we loaded before contained the said prefix or not
        before adding it back, or not, during saving).
        """
        return self._was_used


class WeightRenaming(WeightTransform):

    __slots__ = ()

    def convert(
        self,
        layer_name: str,
        model=None,
        config=None,
        hf_quantizer=None,
        loading_info: LoadStateDictInfo | None = None,
    ):
        collected_tensors = self.materialize_tensors()

        target_key = self.target_patterns[0]
        collected_tensors = {target_key: collected_tensors[self.source_patterns[0]]}

        if hf_quantizer is not None and self.quantization_operation is not None:
            with log_conversion_errors(
                layer_name, loading_info, (len(collected_tensors), layer_name), self.quantization_operation
            ):
                collected_tensors = self.quantization_operation.convert(
                    collected_tensors,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    full_layer_name=target_key,
                    model=model,
                    config=config,
                    missing_keys=loading_info.missing_keys if loading_info else None,
                )

        return collected_tensors


class GroupWeightRename(WeightRenaming):

    __slots__ = ("_active",)

    def __init__(self, source_patterns: list[str], target_patterns: list[str]):
        if len(source_patterns) != len(target_patterns):
            raise ValueError(
                "GroupWeightRename requires N:N length matching, but found "
                f"len(source_patterns)={len(source_patterns)} != len(target_patterns)={len(target_patterns)}"
            )
        super().__init__(source_patterns=source_patterns, target_patterns=target_patterns)
        self._active = None  # None = undecided; True = guard was seen

    def rename_source_key(self, source_key: str) -> tuple[str, str | None]:
        matched = self._scoped_match(source_key)
        if matched is None:
            return source_key, None

        prefix_dot, key_to_match, match_object = matched
        matching_group_name = next(name for name, val in match_object.groupdict().items() if val is not None)
        group_index = int(matching_group_name[1:])

        if group_index == 0:
            self._active = True
        elif not self._active:
            return source_key, None

        self._was_used = True
        replacement = self.target_patterns[group_index]
        if re.search(r"\\\d", replacement):
            group_start = self.compiled_sources.groupindex[matching_group_name]
            replacement = re.sub(
                r"\\(\d+)",
                lambda m: match_object.group(group_start + int(m.group(1))),
                replacement,
            )
        renamed_key = key_to_match.replace(match_object.group(0), replacement, 1)
        if prefix_dot is not None:
            renamed_key = prefix_dot + renamed_key
        return renamed_key, self.source_patterns[group_index]

    def reverse_transform(self) -> GroupWeightRename:
        pairs = sorted(zip(self._original_target_patterns, self._original_source_patterns))
        reversed_srcs = [p[0] for p in pairs]
        reversed_tgts = [p[1] for p in pairs]
        result = GroupWeightRename(source_patterns=reversed_srcs, target_patterns=reversed_tgts)
        result.scope_prefix = self.scope_prefix
        result.base_model_prefix = self.base_model_prefix
        return result


class PrefixChange(WeightRenaming):

    __slots__ = (
        "prefix_to_add",
        "prefix_to_remove",
        "model_prefix",
    )

    def __init__(
        self, prefix_to_add: str | None = None, prefix_to_remove: str | None = None, model_prefix: str | None = None
    ):
        if (prefix_to_add is None) ^ (prefix_to_remove is not None):
            raise ValueError("You must provide only one of `prefix_to_add` and `prefix_to_remove`")

        self.prefix_to_add = prefix_to_add
        self.prefix_to_remove = prefix_to_remove
        self.model_prefix = "" if model_prefix is None else model_prefix
        prefix = rf"{self.model_prefix}\." if self.model_prefix != "" else ""

        if prefix_to_add is not None:
            super().__init__(
                source_patterns=rf"^{prefix}(?:(?!{prefix_to_add}\.))(.+)$",
                target_patterns=rf"{prefix}{prefix_to_add}\.\1",
            )
        else:
            super().__init__(source_patterns=rf"^{prefix}{prefix_to_remove}\.(.+)$", target_patterns=rf"{prefix}\1")

    def reverse_transform(self) -> WeightTransform:
        """Reverse the current `WeightTransform` instance, to be able to save with the opposite weight transformations."""
        if self.quantization_operation is not None:
            raise ValueError("Cannot reverse the transform with TP or quantization")

        result = PrefixChange(
            prefix_to_add=self.prefix_to_remove, prefix_to_remove=self.prefix_to_add, model_prefix=self.model_prefix
        )
        result.scope_prefix = self.scope_prefix
        result.base_model_prefix = self.base_model_prefix
        return result


_INTERNAL_MANY_TO_MANY_CONVERSIONS = (
    ErnieFuseAndSplitTextVisionExperts,
    ErnieSplitAndDecoupleTextVisionExperts,
)


class WeightConverter(WeightTransform):
    __slots__ = ("operations",)

    def __init__(
        self, source_patterns: str | list[str], target_patterns: str | list[str], operations: list[ConversionOps]
    ):
        super().__init__(source_patterns, target_patterns)
        self.operations: list[ConversionOps] = operations

        if bool(len(self.source_patterns) - 1) + bool(len(self.target_patterns) - 1) >= 2:
            if not any(isinstance(op, _INTERNAL_MANY_TO_MANY_CONVERSIONS) for op in self.operations):
                raise ValueError(
                    f"source keys={self.source_patterns}, target_patterns={self.target_patterns} but you can only have one to many, one to one or many to one."
                )
        if not self.operations:
            raise ValueError("WeightConverter requires at least one operation.")

    def convert(
        self,
        layer_name: str,
        model=None,
        config=None,
        hf_quantizer=None,
        loading_info: LoadStateDictInfo | None = None,
    ):
        collected_tensors = self.materialize_tensors()

        for op in self.operations:
            with log_conversion_errors(layer_name, loading_info, (len(collected_tensors), layer_name), op):
                collected_tensors = op.convert(
                    collected_tensors,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    full_layer_name=layer_name,
                    model=model,
                    config=config,
                    missing_keys=loading_info.missing_keys if loading_info else None,
                )

        full_name = layer_name
        if ".*." in layer_name:
            full_name = layer_name.replace(".*.", ".0.")

        try:
            prefix, _, suffix = next(full_name.partition(k) for k in collected_tensors.keys() if k in full_name)
            collected_tensors = {prefix + k + suffix: v for k, v in collected_tensors.items()}
        except StopIteration:
            pass

        if hf_quantizer is not None and self.quantization_operation is not None:
            with log_conversion_errors(
                layer_name, loading_info, (len(collected_tensors), layer_name), self.quantization_operation
            ):
                collected_tensors = self.quantization_operation.convert(
                    collected_tensors,
                    source_patterns=self.source_patterns,
                    target_patterns=self.target_patterns,
                    full_layer_name=layer_name,
                    config=config,
                    model=model,
                    missing_keys=loading_info.missing_keys if loading_info else None,
                )
        return collected_tensors


GLOBAL_WORKERS = min(4, os.cpu_count() or 4)


def _materialize_copy(tensor: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
    tensor = tensor[...]
    if dtype is not None or device is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor


def spawn_materialize(
    thread_pool: ThreadPoolExecutor | None,
    tensor: torch.Tensor,
    device=None,
    dtype=None,
    sharding_op: DtensorShardOperation | None = None,
    tensor_idx: int | None = None,
) -> Future | Callable:
    """Materialize (and optionally shard) a tensor, asynchronously if a thread pool is provided.

    When ``sharding_op`` is given the tensor is sharded (DTensor placement or legacy TP plan);
    otherwise it is simply copied to *device*/*dtype*. Without a thread pool a deferred
    callable is returned instead of a Future.
    """

    def _job():
        pass

    if thread_pool is not None:
        return thread_pool.submit(_job)
    else:
        return _job


def dot_natural_key(s: str):
    """Sort key for state-dict names: split on `"."` and sort digits numerically
    and strings alphabetically. We emit a tuple at each point to sort ints
    first and strings second to avoid int-string comparison failures.
    """
    parts = []
    for part in s.split("."):
        if part.isdigit():
            parts.append((0, int(part)))
        else:
            parts.append((1, part))
    return parts


@contextmanager
def log_conversion_errors(
    first_target_key: str,
    loading_info: LoadStateDictInfo | None,
    extras: Any = None,
    op: list[ConversionOps] | ConversionOps | None = None,
):
    """Catch all exceptions during `convert` calls, and log the errors for later. Re-raise a `SkipParameters` exception
    that will be caught later to skip the parameters that raised the original Exception."""
    try:
        yield
    except Exception as e:
        if loading_info is None:
            raise e

        def _format_op_name(curr_op: list[ConversionOps] | ConversionOps | None) -> str | None:
            if curr_op is None:
                return None
            if isinstance(curr_op, (list, tuple, set)):
                names = [o.__class__.__name__ for o in curr_op if o is not None]
                if not names:
                    return None
                return ", ".join(names)
            return curr_op.__class__.__name__

        op_name = _format_op_name(op)

        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        if isinstance(extras, tuple) and len(extras) == 2:
            length, target_keys = extras
            descriptor = f"{op_name} " if op_name else ""
            loading_info.conversion_errors[first_target_key] = (
                f"{tb_str}{e}\nError: {descriptor}on tensors destined for {target_keys}. Ckpt contains: {length}"
            )
        elif isinstance(extras, str):
            suffix = f" via {op_name}" if op_name else ""
            loading_info.conversion_errors[first_target_key] = (
                f"{tb_str}{e}\nError{suffix} when processing parameter {extras}"
            )
        elif extras is None and op_name:
            loading_info.conversion_errors[first_target_key] = f"{op_name}: {e}"
        else:
            loading_info.conversion_errors[first_target_key] = f"{extras} |Error: {e}"

        raise SkipParameters()


@torch.no_grad()
def set_param_for_module(
    model: PreTrainedModel,
    target_name: str,
    param_value: torch.Tensor,
    loading_info: LoadStateDictInfo,
    distributed_operation: TensorParallelLayer | None,
    hf_quantizer: HfQuantizer,
):
    module_path, _, param_name = target_name.rpartition(".")
    module_obj = model.get_submodule(module_path) if module_path else model

    if param_name == torch.nn.modules.module._EXTRA_STATE_KEY_SUFFIX:
        module_obj.set_extra_state(param_value)
        loading_info.missing_keys.discard(target_name)
        return

    ref = getattr(module_obj, param_name)
    if ref is None:
        loading_info.unexpected_keys.add(target_name)
    else:
        if not isinstance(param_value, torch.nn.Parameter) and not isinstance(ref, DTensor):
            if param_name not in module_obj._buffers:
                param_value = torch.nn.Parameter(param_value, requires_grad=param_value.is_floating_point())

        loading_info.missing_keys.discard(target_name)

        if distributed_operation is not None:
            expected_shape = torch.Size(distributed_operation.get_expected_sharded_shape(ref.shape))
        elif isinstance(ref, DTensor):
            expected_shape = ref._local_tensor.shape
        else:
            expected_shape = ref.shape

        if ref is not None and param_value.shape != expected_shape and hf_quantizer is None:
            loading_info.mismatched_keys.add((target_name, param_value.shape, expected_shape))
        else:
            if isinstance(ref, DTensor):
                local_param = param_value.detach() if isinstance(param_value, torch.nn.Parameter) else param_value
                dtensor_param = _dtensor_from_local_like(local_param, ref)
                param_value = torch.nn.Parameter(dtensor_param, requires_grad=ref.requires_grad)
            param_value._is_hf_initialized = True
            setattr(module_obj, param_name, param_value)
            if distributed_operation is not None:
                distributed_operation.update_module_attributes(module_obj)


def offload_and_maybe_resave_param(
    target_name: str,
    param: torch.Tensor,
    loading_info: LoadStateDictInfo,
    disk_offload_folder: str,
    disk_offload_index: dict,
    applied_ops: WeightConverter | WeightRenaming,
) -> dict:
    """Takes care of correctly offloading `param`. If it's not already present in the `disk_offload_index`, or if any
    WeightConverter operations have been applied, it will resave the new parameter. Otherwise, it will use the original
    `disk_offload_index` for this given param."""
    loading_info.missing_keys.discard(target_name)
    if target_name not in disk_offload_index or isinstance(applied_ops, WeightConverter):
        disk_offload_index = offload_weight(param, target_name, disk_offload_folder, disk_offload_index)
    return disk_offload_index


class SkipParameters(Exception):

    pass


def rename_source_key(
    source_key: str,
    weight_renamings: list[WeightRenaming],
    weight_converters: list[WeightConverter],
    base_model_prefix: str | None = None,
    meta_state_dict: dict | None = None,
) -> tuple[str, str | None]:
    """
    Rename a checkpoint key by first applying all `WeightRenaming`s, then at most one `WeightConverter`.

    A renaming and a converter may act on the same key in that order: the renaming normalises the
    key into the namespace the converter expects. The reverse holds on the save path (converter
    first, then renaming). There is no need for a converter-then-rename order because converters
    act only on specific leaf patterns; no subsequent renamings should ever target their output.

    Args:
        source_key (`str`):
            The original checkpoint key to rename.
        weight_renamings (`list[WeightRenaming]`):
            Applied in order; every matching renaming fires (they may chain).
        weight_converters (`list[WeightConverter]`):
            Applied after all renamings; at most one may match. Subsequent converters are skipped.
        base_model_prefix (`str`, *optional*):
            Base-model prefix to add or strip when both `base_model_prefix` and `meta_state_dict` are given.
        meta_state_dict (`dict`, *optional*):
            Meta state dict used to decide whether `base_model_prefix` should be added or stripped.

    Returns:
        `tuple[str, str | None]`: The renamed key and the matched converter's source pattern
        (or `None` if no converter matched).
    """
    renamed_key = source_key
    for renaming in weight_renamings:
        renamed_key, _ = renaming.rename_source_key(renamed_key)

    source_pattern = None
    for converter in weight_converters:
        renamed_key, source_pattern = converter.rename_source_key(renamed_key)
        if source_pattern is not None:
            break

    if base_model_prefix is not None and meta_state_dict is not None:
        if (
            renamed_key.startswith(base_model_prefix)
            and meta_state_dict.get(re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)) is not None
        ):
            renamed_key = re.sub(f"^{base_model_prefix}.", "", renamed_key, count=1)
        elif meta_state_dict.get(f"{base_model_prefix}.{renamed_key}") is not None:
            renamed_key = f"{base_model_prefix}.{renamed_key}"

    return renamed_key, source_pattern


def convert_and_load_state_dict_in_model(
    model: PreTrainedModel,
    state_dict: dict[str, Any],
    load_config: LoadStateDictConfig,
    tp_plan: dict[str, str] | None,
    disk_offload_index: dict | None = None,
):
    r"""
    We build a mapping from the keys obtained by renaming each of the checkpoint keys according to the weight_mapping rules.
    Then we load the tensors into the model, applying any conversion operations as needed.

    The `param_name_to_load` will look like this:
    {
        "model.layers.0.attention.q.weight": # Notice here there is only the first key of the target keys
            WeightConverter(
                source_patterns=["qkv"],
                target_patterns=["q", "k","v"],
                operations=[Chunk(dim=0, chunks=3)]),
                collected_tensors={
                    "qkv": [Future]},
                layer_targets={
                    "model.layers.0.attention.q.weight": {"model.layers.0.attention.qkv.weight"},
                    "model.layers.0.attention.k.weight": {"model.layers.0.attention.qkv.weight"},
                    "model.layers.0.attention.v.weight": {"model.layers.0.attention.qkv.weight"},
                }
            ),
        ...
    }

    We make sure that the keys are the full keys. The only "nit" here is that 1 key can map to multiple target keys (e.g. qkv -> q, k, v).
    In that case the weight converter will take care of doing the appropriate renaming.

    For example for:
    ```python
    WeightConverter(
        source_patterns=["mlp.experts.*.gate_proj.weight","mlp.experts.*.up_proj.weight"],
        target_patterns="mlp.experts.gate_up_proj",
        operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
    )
    ```
    we would have the following collected tensors:
    ```python
    collected_tensors = {
        "mlp.experts.*.gate_proj.weight": [Future, Future, Future, Future, Future, Future, Future, Future],
        "mlp.experts.*.up_proj.weight": [Future, Future, Future, Future, Future, Future, Future, Future],
    }
    ```
    The first op, `MergeModulelist`, would stack the 8 tensors of each source but will not "rename" them into the fused target name.
    The second op, `Concatenate`, would then rename the fused tensor into the final target name.

    If we want to split `qkv` we would have:
    ```python
    collected_tensors = {
        "attention.qkv.weight": [Future], # here its the full SOURCE keys.
    }
    ```
    The `Chunk` operation would then split the single tensor into 3 and rename them accordingly and update the collected tensors to:
    ```python
    realized_values = {
        "attention.q.weight": [Tensor],
        "attention.k.weight": [Tensor],
        "attention.v.weight": [Tensor],
    }
    ```

    Now that this is done, we can quantize / dequantize accordingly the collected_tensors.

    For some quantization methods, we need to gather different tensors:

    ```python
    # for "medmekk/llama-3.2-1b-float8-torchao"
    WeightConverter(
        source_patterns=[":qdata", ":scale"],
        target_patterns="",
        operations=[TorchaoDeserialize()],
    )
    ```
    This will collect all tensors that have the same prefix, but end with `:qdata` or `:scale`. This will give us:
    ```python
    all_weight_mapping = {
        "model.layers.13.self_attn.o_proj.weight": WeightConverter(
            source_patterns=[":qdata", ":scale"],
            target_patterns="",
            operations=[TorchaoDeserialize()],
            collected_tensors={
                ":qdata": [Future],
                ":scale": [Future],
            },
        ...
    }
    ```

    """
    base_model_prefix = model.base_model_prefix
    tp_plan = tp_plan or {}
    device_map = load_config.device_map or {"": "cpu"}
    hf_quantizer = load_config.hf_quantizer
    dtype = load_config.dtype
    device_mesh = load_config.device_mesh
    disk_offload_folder = load_config.disk_offload_folder
    offload_buffers = load_config.offload_buffers
    dtype_plan = load_config.dtype_plan or {}
    weight_mapping = load_config.weight_mapping or []
    meta_model_state_dict = model.state_dict()
    model_buffers = {k for k, _ in model.named_buffers()}

    loading_info = LoadStateDictInfo(
        missing_keys=set(meta_model_state_dict.keys()),
        unexpected_keys=set(),
        mismatched_keys=set(),
        conversion_errors={},
        error_msgs=[],
    )

    has_on_the_fly_quantization = hf_quantizer is not None and not hf_quantizer.pre_quantized
    if (
        is_env_variable_true("HF_DEACTIVATE_ASYNC_LOAD")
        or "disk" in device_map.values()
        or has_on_the_fly_quantization
    ):
        thread_pool = None
    else:
        thread_pool = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)

    renamings = [entry for entry in weight_mapping if isinstance(entry, WeightRenaming)]
    converters = [entry for entry in weight_mapping if isinstance(entry, WeightConverter)]
    param_name_to_load: dict[str, WeightRenaming | WeightConverter] = {}

    if tp_plan != {}:
        tp_plan_alt, tp_plan_by_group_name, _ = build_glob_alternation(list(tp_plan.keys()))
    if dtype_plan != {}:
        dtype_policy_alt, dtype_policy_by_group_name, _ = build_glob_alternation(list(dtype_plan.keys()))

    pattern_to_converter = {k: converter for converter in converters for k in converter.source_patterns}

    state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    for original_key, tensor in state_dict:
        renamed_key, source_pattern = rename_source_key(
            original_key, renamings, converters, base_model_prefix, meta_model_state_dict
        )
        if renamed_key not in meta_model_state_dict and original_key in meta_model_state_dict:
            renamed_key, source_pattern = rename_source_key(
                original_key, [], [], base_model_prefix=base_model_prefix, meta_state_dict=meta_model_state_dict
            )

        if renamed_key in meta_model_state_dict:
            empty_param = meta_model_state_dict.get(renamed_key)
            if source_pattern is not None:
                new_converter = deepcopy(pattern_to_converter[source_pattern])
                mapping = param_name_to_load.setdefault(renamed_key, new_converter)
            else:
                mapping = param_name_to_load.setdefault(renamed_key, WeightRenaming(original_key, renamed_key))
                source_pattern = original_key

            needs_quantization = (
                hf_quantizer
                and not hf_quantizer.pre_quantized
                and hf_quantizer.param_needs_quantization(model, renamed_key)
            )
            if needs_quantization:
                mapping.quantization_operation = hf_quantizer.get_quantize_ops()

            _dtype = dtype
            if (
                hf_quantizer
                and hf_quantizer.pre_quantized
                and (
                    original_key != renamed_key
                    or not (
                        tensor.get_dtype().startswith(("F", "BF"))
                        if hasattr(tensor, "get_dtype")
                        else tensor.is_floating_point()
                    )
                )
            ):
                _dtype = None
            elif dtype_plan != {} and dtype_policy_alt.search(renamed_key):
                matched_dtype_pattern = dtype_policy_alt.search(renamed_key)
                if matched_dtype_pattern is not None:
                    _dtype = dtype_plan[dtype_policy_by_group_name[matched_dtype_pattern.lastgroup]]
            elif empty_param is not None and empty_param.dtype != _dtype:
                _dtype = empty_param.dtype  # usually correct when initializing

            tensor_idx = (
                len(mapping.collected_tensors.get(source_pattern, []))
                if isinstance(mapping, WeightConverter)
                and any(isinstance(op, MergeModulelist) for op in mapping.operations)
                else None
            )

            param_device = get_device(device_map, renamed_key, valid_torch_device=True)
            sharding_op = None
            materialize_device = param_device

            if isinstance(empty_param, DTensor):
                sharding_op = DtensorShardOperation(empty_param)
            elif device_mesh and tp_plan:
                if matched_tp_pattern := tp_plan_alt.search(renamed_key):
                    matched_tp_pattern = tp_plan_by_group_name[matched_tp_pattern.lastgroup]
                    if getattr(mapping, "distributed_operation", None) is None:
                        tp_layer = ALL_PARALLEL_STYLES[model.tp_plan[matched_tp_pattern]].__class__
                        mapping.distributed_operation = tp_layer(
                            device_mesh=device_mesh, rank=device_mesh.get_local_rank(), empty_param=empty_param.clone()
                        )
                    sharding_op = mapping.distributed_operation
                    materialize_device = device_map[""]

            future_or_tensor = spawn_materialize(
                thread_pool,
                tensor,
                materialize_device,
                _dtype,
                sharding_op=sharding_op,
                tensor_idx=tensor_idx,
            )

            mapping.add_tensor(renamed_key, original_key, source_pattern, future_or_tensor)
        elif source_pattern is not None:  # add all target keys as unexpected
            mapping = pattern_to_converter[source_pattern]
            for k in mapping.target_patterns:
                loading_info.unexpected_keys.add(renamed_key.replace(mapping.target_patterns[0], k))
        else:
            loading_info.unexpected_keys.add(renamed_key)

    try:
        for first_param_name, mapping in tqdm(param_name_to_load.items(), desc="Loading weights"):
            try:
                realized_value = mapping.convert(
                    first_param_name,
                    model=model,
                    config=model.config,
                    hf_quantizer=hf_quantizer,
                    loading_info=loading_info,
                )
                for target_name, param in realized_value.items():
                    param = param[0] if isinstance(param, list) else param
                    param_device = get_device(device_map, target_name)
                    if param_device == "disk" and (target_name not in model_buffers or offload_buffers):
                        disk_offload_index = offload_and_maybe_resave_param(
                            target_name, param, loading_info, disk_offload_folder, disk_offload_index, mapping
                        )
                    else:
                        set_param_for_module(
                            model,
                            target_name,
                            param,
                            loading_info,
                            mapping.distributed_operation,
                            hf_quantizer,
                        )

                del realized_value

            except SkipParameters:
                continue

    finally:
        if thread_pool is not None:
            thread_pool.shutdown(wait=False, cancel_futures=True)

    model_specific_conversions = [conversion for conversion in weight_mapping if conversion.was_used()]
    model._weight_conversions = model_specific_conversions

    return loading_info, disk_offload_index


def revert_weight_conversion(model: PreTrainedModel, state_dict: dict[str, torch.Tensor]):
    """
    Revert the conversion mapping that was used to load the model with `from_pretrained`, or the default one
    if the model was created in another way and is part of the default mappings.
    """
    weight_conversions = getattr(model, "_weight_conversions", None)
    if weight_conversions is None:
        from .conversion_mapping import get_model_conversion_mapping

        weight_conversions = get_model_conversion_mapping(model, add_legacy=False)
        weight_conversions = [x for x in weight_conversions if not isinstance(x, PrefixChange)]
        weight_conversions = weight_conversions if len(weight_conversions) > 0 else None

    if weight_conversions is None:
        return state_dict

    weight_conversions = weight_conversions[::-1]

    inverted_transforms = [transform.reverse_transform() for transform in weight_conversions]
    inverted_converters = [transform for transform in inverted_transforms if isinstance(transform, WeightConverter)]
    inverted_renamings = [transform for transform in inverted_transforms if not isinstance(transform, WeightConverter)]
    pattern_to_converter = {
        pattern: converter for converter in inverted_converters for pattern in converter.source_patterns
    }

    conversion_mapping: dict[str, WeightTransform] = {}
    state_dict = sorted(state_dict.items(), key=lambda kv: dot_natural_key(kv[0]))
    for original_key, tensor in state_dict:
        converter_key, matched_pattern = rename_source_key(original_key, [], inverted_converters)
        checkpoint_key, _ = rename_source_key(converter_key, inverted_renamings, [])

        if matched_pattern is not None:
            mapping = conversion_mapping.setdefault(converter_key, deepcopy(pattern_to_converter[matched_pattern]))
        else:
            mapping = conversion_mapping.setdefault(checkpoint_key, WeightRenaming(original_key, checkpoint_key))
            matched_pattern = original_key

        mapping.add_tensor(checkpoint_key, original_key, matched_pattern, tensor)

    new_state_dict = {}
    for layer_name, mapping in conversion_mapping.items():
        realized = mapping.convert(layer_name, model=model, config=model.config)
        for target_name, param in realized.items():
            param = param[0] if isinstance(param, list) else param
            if isinstance(mapping, WeightConverter):
                target_name, _ = rename_source_key(target_name, inverted_renamings, [])
            new_state_dict[target_name] = param

    return new_state_dict
