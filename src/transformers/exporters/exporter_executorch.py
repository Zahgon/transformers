
from __future__ import annotations

import math
import operator
from collections.abc import MutableMapping
from typing import Any

from ..utils import logging
from ..utils.import_utils import is_executorch_available, is_torch_available
from .configs import ExecutorchConfig
from .exporter_dynamo import DynamoExporter
from .utils import (
    apply_fx_node_fixes,
    apply_fx_program_fixes,
    apply_patches,
    module_device,
    module_dtype,
    register_fx_node_fix,
    register_fx_program_fix,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.fx.experimental.symbolic_shapes import guard_or_true
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils._sympy.numbers import IntInfinity
    from torch.utils._sympy.value_ranges import ValueRanges

    from .. import masking_utils
    from ..modeling_utils import PreTrainedModel


if is_executorch_available():
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir.capture._config import EdgeCompileConfig
    from executorch.exir.passes.executorch_prim_ops_registry import _PYTHON_SYM_OPS_TO_EXECUTORCH_SYM_OPS
    from executorch.exir.program import EdgeProgramManager, ExecutorchProgramManager, to_edge_transform_and_lower

    if torch.cuda.is_available():
        from executorch.backends.cuda.cuda_backend import CudaBackend
        from executorch.backends.cuda.cuda_partitioner import CudaPartitioner


logger = logging.get_logger(__name__)


class ExecutorchExporter(DynamoExporter):

    required_packages = ["torch", "executorch"]
    tested_versions = {"torch": "2.12.0", "executorch": "1.3.1"}

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: ExecutorchConfig | dict[str, Any],
    ) -> ExecutorchProgramManager:
        pass


def _get_edge_compile_config() -> EdgeCompileConfig:
    pass




def prepare_for_xnnpack(model: PreTrainedModel, sample_inputs: dict[str, Any]):
    pass


def prepare_for_cuda(model: PreTrainedModel, sample_inputs: dict[str, Any]):
    pass


_BACKEND_PREPARE = {
    "xnnpack": prepare_for_xnnpack,
    "cuda": prepare_for_cuda,
}




@register_patch("executorch", "torch.split", "torch.Tensor.split")
def _patch_split(original):
    pass


@register_patch("executorch", "torch.chunk", "torch.Tensor.chunk")
def _patch_chunk(original):
    pass


@register_patch("executorch", "torch.topk", "torch.Tensor.topk")
def _patch_topk(original):
    pass


@register_patch("executorch", "torch.detach", "torch.Tensor.detach")
def _patch_detach(_original):
    pass


@register_patch("executorch", "torch.nn.functional.avg_pool2d")
def _patch_avg_pool2d(original):
    pass


@register_patch("executorch", "transformers.masking_utils._vmap_expansion_sdpa")
def _patch_broadcast_mask_expansion(_original):
    pass


@register_patch("executorch", "torch.nn.functional.scaled_dot_product_attention")
def _patch_scaled_dot_product_attention(original):
    pass


@register_patch("executorch", "torch.bernoulli", "torch.Tensor.bernoulli")
def _patch_bernoulli(_original):
    pass


@register_patch("executorch", "torch.Tensor.expand")
def _patch_expand(original):
    pass




@register_patch(
    "executorch",
    "executorch.exir.sym_util.eval_upper_bound",
    "executorch.exir.passes.sym_shape_eval_pass.eval_upper_bound",
)
def _patch_eval_upper_bound(original):
    pass


@register_patch(
    "executorch", "executorch.exir.passes.prune_empty_tensors_pass.PruneEmptyTensorsPass.remove_empty_tensors_from_cat"
)
def _patch_remove_empty_tensors_from_cat(_original):
    pass


@register_patch("executorch", "executorch.exir.verification.verifier._check_tensor_args_matching_op_allowed_dtype")
def _patch_check_tensor_args_dtype(original):
    pass


@register_patch(
    "executorch",
    "executorch.exir.tensor.dim_order_from_stride",
    "executorch.exir.tensor_layout.dim_order_from_stride",
    "executorch.exir.emit._emitter.dim_order_from_stride",
    "executorch.exir.passes.replace_view_copy_with_view_pass.dim_order_from_stride",
)
def _patch_dim_order_from_stride(_original):
    pass


@register_patch("executorch", "executorch.exir.passes.spec_prop_pass.SpecPropPass.update_placeholder_tensor_specs")
def _patch_update_placeholder_tensor_specs(_original):
    pass


@register_patch(
    "executorch",
    "executorch.exir.passes.executorch_prim_ops_registry._EXECUTORCH_SYM_OPS",
    "executorch.exir.verification.verifier._EXECUTORCH_SYM_OPS",
)
def _extend_sym_ops_allowlist(original):
    pass


def _make_squeeze_define_node(original):
    pass


@register_patch("executorch", "executorch.backends.xnnpack.operators.node_visitor._node_visitor_dict")
def _patch_squeeze_node_visitors(original):
    pass



_MAX_DIM_MULTIPLIER = 4
_MAX_DIM_FLOOR = 1024


def _as_int(x, default: int = 0) -> int:
    pass


@register_fx_program_fix("executorch")
def _fix_range_constraints(exported_program: ExportedProgram) -> None:
    pass


@register_fx_program_fix("executorch")
def _fix_missing_placeholder_vals(exported_program: ExportedProgram) -> None:
    pass




@register_fx_node_fix("executorch")
def _fix_amax_dim(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("executorch")
def _fix_python_sym_op(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("executorch")
def _fix_clone_memory_format(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("executorch")
def _fix_sym_pow_as_mul(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass
