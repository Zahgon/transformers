from __future__ import annotations

import inspect
import os
from typing import TYPE_CHECKING, Any

from ..integrations.tensor_parallel import replace_layer_number_by_wildcard
from ..utils import is_torch_available, is_torch_greater_or_equal, logging, strtobool
from ..utils.quantization_config import QuantizationMethod


if TYPE_CHECKING:
    import torch.nn as nn

    from .configuration_utils import DistributedConfig

if is_torch_available():
    import torch

if is_torch_available() and is_torch_greater_or_equal("2.6"):
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

logger = logging.get_logger(__name__)


def is_fsdp_enabled() -> bool:
    """Check if FSDP is active via Accelerate (env var based) — covers FSDP1 only."""
    if not is_torch_available():
        return False

    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and strtobool(os.environ.get("ACCELERATE_USE_FSDP", "False")) == 1
        and strtobool(os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING", "False")) == 1
    )


def is_fsdp_managed_module(module: nn.Module) -> bool:
    """Check if a module is managed by FSDP (1 or 2)."""
    if not is_torch_available():
        return False
    if not torch.distributed.is_available():
        return False

    if getattr(module, "_is_fsdp_managed_module", False):
        return True
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel
    except ImportError:
        return False
    return isinstance(module, FullyShardedDataParallel)


def _get_fsdp_policy_kwargs(distributed_config: DistributedConfig | None) -> dict[str, Any]:
    """Build ``fully_shard`` policy kwargs from ``DistributedConfig`` runtime flags."""
    if distributed_config is None:
        return {}

    fsdp_policy_kwargs = {}
    if distributed_config.fsdp_cpu_offload:
        fsdp_policy_kwargs["offload_policy"] = CPUOffloadPolicy()
    if distributed_config.fsdp_mixed_precision:
        fsdp_policy_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
        )
    return fsdp_policy_kwargs


def _get_input_output_embeddings(model: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    input_embed = None
    output_head = None
    if hasattr(model, "get_input_embeddings"):
        input_embed = model.get_input_embeddings()
    if hasattr(model, "get_output_embeddings"):
        output_head = model.get_output_embeddings()
    return input_embed, output_head


def is_norm_and_head_pair(no_reshard_targets: list[tuple[str, nn.Module]], model: nn.Module) -> bool:
    if len(no_reshard_targets) != 2:
        return False
    input_embed, output_head = _get_input_output_embeddings(model)
    head_modules = {module for module in (input_embed, output_head) if module is not None}

    names, modules = [], []
    for name, module in no_reshard_targets:
        names.append(name)
        modules.append(module)

    has_final_norm = any(name == "norm" or name.endswith(".norm") for name in names)
    has_output_head = any(module in head_modules for module in modules)
    return has_final_norm and has_output_head


def _resolve_tied_embed_lm_head_plan(
    fsdp_plan: dict[str, str],
    model: nn.Module,
) -> dict[str, str]:
    """
    Rewrite the plan so tied embed/lm_head weights are wrapped once.
    Example:
        {"model.embed_tokens": "free_full_weight",
        "model.layers.*": "free_full_weight",
        "model.norm": "keep_full_weight",
        "lm_head": "keep_full_weight"}
    ->
        {"model.layers.*": "free_full_weight",
        "model.norm": "keep_full_weight",
        "model.embed_tokens": "keep_full_weight"}
    """
    tied_keys = getattr(model, "all_tied_weights_keys", None) or {}
    if not tied_keys:
        return fsdp_plan

    input_embed, output_head = _get_input_output_embeddings(model)
    name_by_module = {module: name for name, module in model.named_modules()}
    embed_module = name_by_module.get(input_embed)
    head_module = name_by_module.get(output_head)

    if embed_module is None or head_module is None:
        return fsdp_plan

    adapted_plan = fsdp_plan.copy()
    adapted_plan.pop(embed_module, None)

    if fsdp_plan.get(head_module) == "keep_full_weight":
        adapted_plan.pop(head_module, None)
        adapted_plan[embed_module] = "keep_full_weight"

    return adapted_plan


def expand_fsdp_plan(
    model: nn.Module,
    fsdp_plan: dict[str, str],
) -> tuple[list[tuple[str, nn.Module]], list[tuple[str, nn.Module]]]:
    """Expand plan keys into reshard and no-reshard ``(module_name, module)`` shard targets."""
    reshard_targets: list[tuple[str, nn.Module]] = []
    no_reshard_targets: list[tuple[str, nn.Module]] = []

    for module_name, module in model.named_modules():
        plan_key = module_name if module_name in fsdp_plan else replace_layer_number_by_wildcard(module_name)
        if plan_key in fsdp_plan:
            if fsdp_plan[plan_key] == "keep_full_weight":
                no_reshard_targets.append((module_name, module))
            else:
                reshard_targets.append((module_name, module))

    return reshard_targets, no_reshard_targets


def verify_fsdp_plan(module_names: list[str], fsdp_plan: dict[str, str] | None) -> None:
    pass


def apply_fully_sharded_data_parallelism(
    model: nn.Module, fsdp_mesh: torch.distributed.device_mesh.DeviceMesh
) -> nn.Module:
    """
    Apply FSDP2 (fully_shard) to a model.
    """
    if not is_torch_available():
        raise ImportError("PyTorch is required for FSDP support")

    if not is_torch_greater_or_equal("2.6"):
        raise OSError("FSDP2 requires torch>=2.6")

    fsdp_plan = dict(getattr(model, "_fsdp_plan", None) or {})
    if not fsdp_plan:
        raise ValueError(
            f"{type(model).__name__} does not have a FSDP2 plan declared. Set "
            "`base_model_fsdp_plan` on the config and `_fsdp_plan` on the head class."
        )

    distributed_config = getattr(model.config, "distributed_config", None)
    fsdp_policy_kwargs = _get_fsdp_policy_kwargs(distributed_config)

    adapted_fsdp_plan = _resolve_tied_embed_lm_head_plan(fsdp_plan, model)
    reshard_targets, no_reshard_targets = expand_fsdp_plan(model, adapted_fsdp_plan)

    for module_name, module in reshard_targets:
        fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=True, **fsdp_policy_kwargs)
        logger.debug(f"Applied fully_shard to {module_name} (reshard=True)")

    if is_norm_and_head_pair(no_reshard_targets, model):
        names, modules = [], []
        for name, module in no_reshard_targets:
            names.append(name)
            modules.append(module)
        fully_shard(modules, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_policy_kwargs)
        logger.debug(f"Grouped tail {names} (reshard=False)")
    else:
        for name, module in no_reshard_targets:
            fully_shard(module, mesh=fsdp_mesh, reshard_after_forward=False, **fsdp_policy_kwargs)
            logger.debug(f"Applied fully_shard to {name} (reshard=False)")

    fully_shard(model, mesh=fsdp_mesh, **fsdp_policy_kwargs)

    logger.info(f"FSDP2 applied to model via _fsdp_plan: {len(fsdp_plan)} entries")

    model._is_fsdp_managed_module = True


    return model


def get_fsdp_ckpt_kwargs():
    """
    Returns checkpoint kwargs for FSDP model saving.

    Checks if the `adapter_only` parameter is supported by `save_fsdp_model` from accelerate
    and returns the appropriate kwargs.
    """
    from accelerate.utils import save_fsdp_model

    if "adapter_only" in list(inspect.signature(save_fsdp_model).parameters):
        return {"adapter_only": True}
    else:
        return {}


def update_fsdp_plugin_peft(model, accelerator):
    pass
