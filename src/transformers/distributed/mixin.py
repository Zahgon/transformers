from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

from ..integrations.tensor_parallel import (
    ALL_PARALLEL_STYLES,
    apply_tensor_parallelism,
    initialize_tensor_parallelism,
)
from ..utils import is_torch_available
from .configuration_utils import DistributedConfig
from .fsdp import apply_fully_sharded_data_parallelism
from .utils import initialize_fully_sharded_data_parallelism


if TYPE_CHECKING:
    import torch.nn as nn

if is_torch_available():
    import torch

    _torch_distributed_available = torch.distributed.is_available()
else:
    _torch_distributed_available = False


class DistributedMixin:

    _device_mesh = None
    _tp_plan: dict[str, str] | None = None
    _ep_plan: dict[str, str] | None = None
    _tp_size = None
    _pp_plan: dict[str, tuple[str, str]] | None = None
    _fsdp_plan: dict[str, str] | None = None

    def init_parallel_plans(self) -> None:
        """Copy class-level plans onto the instance and merge config/children contributions."""
        model_cls = type(self)
        self._tp_plan = dict(getattr(model_cls, "_tp_plan", None) or {})
        self._ep_plan = dict(getattr(model_cls, "_ep_plan", None) or {})
        self._pp_plan = dict(getattr(model_cls, "_pp_plan", None) or {})
        self._fsdp_plan = dict(getattr(model_cls, "_fsdp_plan", None) or {})

        if self.base_model is self:
            self._pp_plan.update(self.config.base_model_pp_plan or {})
            self._tp_plan.update(self.config.base_model_tp_plan or {})
            self._ep_plan.update(self.config.base_model_ep_plan or {})
            self._fsdp_plan.update(self.config.base_model_fsdp_plan or {})

        for name, module in self.named_children():
            if plan := getattr(module, "_ep_plan", None):
                self._ep_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_tp_plan", None):
                self._tp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_pp_plan", None):
                self._pp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})
            if plan := getattr(module, "_fsdp_plan", None):
                self._fsdp_plan.update({f"{name}.{k}": v for k, v in plan.copy().items()})

    @property
    def tp_plan(self) -> dict[str, str]:
        pass

    @property
    def fsdp_plan(self) -> dict[str, str]:
        pass

    @property
    def pp_plan(self) -> dict[str, tuple[str, str]]:
        pass

    @tp_plan.setter
    def tp_plan(self, plan: dict[str, str] | None):
        pass

    @pp_plan.setter
    def pp_plan(self, plan: dict[str, tuple[str, str]] | None):
        pass

    @classmethod
    def prepare_distribute_model(
        cls,
        distributed_config: DistributedConfig | dict | None,
        *,
        device_mesh=None,
        device_map=None,
    ) -> tuple[DistributedConfig | None, object, object]:
        """Parse ``distributed_config``, init TP/FSDP mesh, and validate."""
        if distributed_config is None:
            return None, device_map, device_mesh

        if isinstance(distributed_config, dict):
            distributed_config = DistributedConfig.from_dict(distributed_config)

        if distributed_config.tp_size > 1:
            if distributed_config.tp_plan is None:
                distributed_config.tp_plan = "auto"
            device_map, device_mesh = initialize_tensor_parallelism(
                distributed_config.tp_plan,
                tp_size=distributed_config.tp_size,
                device_mesh=device_mesh,
                device_map=device_map,
            )
        elif distributed_config.fsdp_size > 1:
            device_map, device_mesh = initialize_fully_sharded_data_parallelism(distributed_config)

        distributed_config.validate()
        return distributed_config, device_map, device_mesh

    @classmethod
    def maybe_distribute_model(
        cls,
        model: nn.Module,
        distributed_config: DistributedConfig | None,
        device_mesh,
    ):
        """Apply TP or FSDP2 after model init, before weight loading."""
        if _torch_distributed_available and device_mesh is not None:
            model.config.distributed_config = distributed_config
            model._device_mesh = device_mesh

            if distributed_config.tp_size > 1:
                model = apply_tensor_parallelism(
                    model,
                    distributed_config.tp_plan,
                    distributed_config,
                    device_mesh,
                )
            elif distributed_config.fsdp_size > 1:
                fsdp_mesh = device_mesh["fsdp"] if device_mesh.ndim > 1 else device_mesh
                model = apply_fully_sharded_data_parallelism(model, fsdp_mesh)
        return model
