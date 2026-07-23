
from __future__ import annotations

import importlib.metadata
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from packaging import version
from torch import nn

from .optimization import Adafactor
from .trainer_pt_utils import LayerWiseDummyOptimizer
from .trainer_utils import check_target_module_exists
from .training_args import OptimizerNames, ParallelMode
from .utils import (
    is_apollo_torch_available,
    is_bitsandbytes_available,
    is_galore_torch_available,
    is_grokadamw_available,
    is_lomo_available,
    is_schedulefree_available,
    is_torch_optimi_available,
    is_torchao_available,
    strtobool,
)


if TYPE_CHECKING:
    from .modeling_utils import PreTrainedModel
    from .training_args import TrainingArguments

logger = logging.getLogger(__name__)


@dataclass
class OptimizerContext:

    args: TrainingArguments
    model: PreTrainedModel | None
    optimizer_kwargs: dict[str, Any]
    adam_kwargs: dict[str, Any]
    optim_args: dict[str, str]


def _parse_optim_args(optim_args_str: str | None) -> dict[str, str]:
    """Parse optimizer arguments from a comma-separated string."""
    if not optim_args_str:
        return {}
    optim_args = {}
    for mapping in optim_args_str.replace(" ", "").split(","):
        key, value = mapping.split("=")
        optim_args[key] = value
    return optim_args


OptimizerHandler = Callable[[OptimizerContext], tuple[Any, dict[str, Any]]]


def is_optimizer_factory(optimizer_cls_or_factory: Any) -> bool:
    """
    Check if the returned value from a handler is a factory rather than an Optimizer class.

    Factory callables are used for complex optimizers like Muon or Dion that need to:
    - Split parameters between multiple internal optimizers
    - Handle complex sharding logic
    - Access the full model structure for parameter grouping

    Args:
        optimizer_cls_or_factory: The first element returned by an optimizer handler.

    Returns:
        `bool`: True if it's not an Optimizer class (i.e., likely a factory), False if it's an Optimizer class.
    """
    if isinstance(optimizer_cls_or_factory, type) and issubclass(optimizer_cls_or_factory, torch.optim.Optimizer):
        return False
    return True


def _setup_low_rank_optimizer(
    args: TrainingArguments,
    model: PreTrainedModel,
    optimizer_name: str,
    optimizer_mapping: dict[str, Any],
    optim_kwargs: dict[str, Any],
    optimizer_kwargs: dict[str, Any],
    is_layerwise_supported: bool = True,
) -> tuple[Any, dict[str, Any]]:
    pass




def _get_adafactor(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_adamw_torch(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_adamw_torch_xla(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_adamw_torch_npu_fused(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_bitsandbytes_optimizer(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_adamw_anyprecision(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_sgd(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_adagrad(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_rmsprop(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_galore_optimizer(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_apollo_optimizer(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_lomo_optimizer(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_grokadamw(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_torchao_optimizer(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_schedule_free_optimizer(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass


def _get_stable_adamw(ctx: OptimizerContext) -> tuple[Any, dict[str, Any]]:
    pass



_BITSANDBYTES_OPTIMIZERS = [
    OptimizerNames.ADAMW_BNB,
    OptimizerNames.ADAMW_8BIT,
    OptimizerNames.PAGED_ADAMW,
    OptimizerNames.PAGED_ADAMW_8BIT,
    OptimizerNames.ADEMAMIX,
    OptimizerNames.ADEMAMIX_8BIT,
    OptimizerNames.PAGED_ADEMAMIX,
    OptimizerNames.PAGED_ADEMAMIX_8BIT,
    OptimizerNames.LION,
    OptimizerNames.LION_8BIT,
    OptimizerNames.PAGED_LION,
    OptimizerNames.PAGED_LION_8BIT,
    OptimizerNames.RMSPROP_BNB,
    OptimizerNames.RMSPROP_8BIT,
    OptimizerNames.RMSPROP_32BIT,
]

_GALORE_OPTIMIZERS = [
    OptimizerNames.GALORE_ADAMW,
    OptimizerNames.GALORE_ADAMW_8BIT,
    OptimizerNames.GALORE_ADAFACTOR,
    OptimizerNames.GALORE_ADAMW_LAYERWISE,
    OptimizerNames.GALORE_ADAMW_8BIT_LAYERWISE,
    OptimizerNames.GALORE_ADAFACTOR_LAYERWISE,
]

_APOLLO_OPTIMIZERS = [
    OptimizerNames.APOLLO_ADAMW,
    OptimizerNames.APOLLO_ADAMW_LAYERWISE,
]

_TORCHAO_OPTIMIZERS = [
    OptimizerNames.ADAMW_TORCH_4BIT,
    OptimizerNames.ADAMW_TORCH_8BIT,
]

_SCHEDULE_FREE_OPTIMIZERS = [
    OptimizerNames.SCHEDULE_FREE_RADAM,
    OptimizerNames.SCHEDULE_FREE_ADAMW,
    OptimizerNames.SCHEDULE_FREE_SGD,
]


_OPTIMIZER_HANDLERS: dict[str, OptimizerHandler] = {
    OptimizerNames.ADAFACTOR: _get_adafactor,
    OptimizerNames.ADAMW_TORCH: _get_adamw_torch,
    OptimizerNames.ADAMW_TORCH_FUSED: _get_adamw_torch,
    OptimizerNames.ADAMW_TORCH_XLA: _get_adamw_torch_xla,
    OptimizerNames.ADAMW_TORCH_NPU_FUSED: _get_adamw_torch_npu_fused,
    OptimizerNames.ADAMW_ANYPRECISION: _get_adamw_anyprecision,
    OptimizerNames.SGD: _get_sgd,
    OptimizerNames.ADAGRAD: _get_adagrad,
    OptimizerNames.RMSPROP: _get_rmsprop,
    OptimizerNames.GROKADAMW: _get_grokadamw,
    OptimizerNames.STABLE_ADAMW: _get_stable_adamw,
    OptimizerNames.LOMO: _get_lomo_optimizer,
    OptimizerNames.ADALOMO: _get_lomo_optimizer,
    **dict.fromkeys(_BITSANDBYTES_OPTIMIZERS, _get_bitsandbytes_optimizer),
    **dict.fromkeys(_GALORE_OPTIMIZERS, _get_galore_optimizer),
    **dict.fromkeys(_APOLLO_OPTIMIZERS, _get_apollo_optimizer),
    **dict.fromkeys(_TORCHAO_OPTIMIZERS, _get_torchao_optimizer),
    **dict.fromkeys(_SCHEDULE_FREE_OPTIMIZERS, _get_schedule_free_optimizer),
}
