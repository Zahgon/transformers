

from __future__ import annotations

import copy
import functools
import operator
from collections.abc import MutableMapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np

from ..utils import logging
from ..utils.import_utils import is_onnxscript_available, is_torch_available
from .configs import OnnxConfig
from .exporter_dynamo import DynamoExporter
from .utils import (
    apply_fx_node_fixes,
    apply_patches,
    duplicate_leaf_tensors,
    get_leaf_tensors,
    register_fx_node_fix,
    register_patch,
)


if is_torch_available():
    import torch
    from torch.export import ExportedProgram
    from torch.onnx import ONNXProgram

    from .. import masking_utils


if is_onnxscript_available():
    import onnx_ir
    from onnxscript.function_libs.torch_lib.ops.core import aten_index_put
    from onnxscript.onnx_opset import opset18 as op

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

    if is_onnxscript_available():
        from onnxscript.function_libs.torch_lib.ops.core import BOOL, INT64, TReal


logger = logging.get_logger(__file__)


class OnnxExporter(DynamoExporter):

    required_packages = ["torch", "onnx", "onnxscript"]
    tested_versions = {"torch": "2.12.0", "onnx": "1.21.0", "onnxscript": "0.7.0"}

    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, Any],
        config: OnnxConfig | dict[str, Any],
    ) -> ONNXProgram:
        pass




@contextmanager
def patch_model_outputs(model):
    pass


def disambiguate_io_names(inputs_names: list[str], outputs_names: list[str]) -> tuple[list[str], list[str]]:
    pass




@register_patch("onnx", "torch.where")
def _patch_where(original):
    pass


@register_patch("onnx", "torch.unsqueeze", "torch.Tensor.unsqueeze")
def _patch_unsqueeze(original):
    pass


@register_patch("onnx", "transformers.masking_utils._vmap_expansion_sdpa")
def _patch_broadcast_mask_expansion(_original):
    pass


@register_patch("onnx", "torch.nn.RMSNorm.forward")
def _patch_rms_norm_forward(original):
    pass


@register_patch("onnx", "torch.randperm")
def _patch_randperm(original):
    pass


@register_patch("onnx", "torch.histc")
def _patch_histc(original):
    pass


@register_patch("onnx", "onnxscript.onnx_opset._impl.opset13.Opset13.Constant")
def _patch_opset13_constant(original):
    pass


def _patch_cummax_or_cummin(original, *, mode: str):
    pass


@register_patch("onnx", "torch.cummax", "torch.Tensor.cummax")
def _patch_cummax(original):
    pass


@register_patch("onnx", "torch.cummin", "torch.Tensor.cummin")
def _patch_cummin(original):
    pass


@register_patch("onnx", "torch.exp", "torch.Tensor.exp")
def _patch_exp(original):
    pass


@register_patch("onnx", "torch.fft.irfft")
def _patch_irfft(original):
    pass


@register_patch("onnx", "torch.bucketize")
def _patch_bucketize(original):
    pass


@register_patch("onnx", "torch.searchsorted")
def _patch_searchsorted(original):
    pass


@register_patch("onnx", "torch.full")
def _patch_full(original):
    pass


@register_patch("onnx", "torch.masked.mean")
def _patch_masked_mean(original):
    pass


@register_patch("onnx", "torch.masked.var")
def _patch_masked_var(original):
    pass


@register_patch("onnx", "torch.Tensor.masked_scatter")
def _patch_masked_scatter(original):
    pass


@register_patch("onnx", "torch.roll")
def _patch_roll(original):
    pass




@register_patch("onnx", "torch.onnx._internal.exporter._core._prepare_exported_program_for_export")
def _patch_prepare_for_export(original):
    pass




_COMPARISON_OPS = frozenset({operator.le, operator.lt, operator.ge, operator.gt, operator.eq, operator.ne})


@register_fx_node_fix("onnx")
def _fix_dead_comparison(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("onnx")
def _fix_alias(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("onnx")
def _fix_detach_inplace(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("onnx")
def _fix_index_put_inplace(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


_ASSERTION_OPS = set()
if is_torch_available():
    _ASSERTION_OPS.update(
        {
            torch.ops.aten._assert_async.default,
            torch.ops.aten._assert_async.msg,
            torch.ops.aten._assert_scalar.default,
            torch.ops.aten._assert_tensor_metadata.default,
            torch.ops.aten.sym_constrain_range_for_size.default,
        }
    )


@register_fx_node_fix("onnx")
def _fix_assertion(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("onnx")
def _fix_fill_diagonal_inplace(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("onnx")
def _fix_triu_inplace(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("onnx")
def _fix_sort_stable(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass


@register_fx_node_fix("onnx")
def _fix_remainder_scalar(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    pass




def _values_broadcast_to_self(values: TReal, self: TReal) -> bool:
    pass


def _aten_index_put(
    self: TReal,
    indices: Sequence[INT64 | BOOL | None],
    values: TReal,
    accumulate: bool = False,
) -> TReal:
    pass


def _aten_bincount(self: INT64, weights=None, minlength: int = 0) -> INT64:
    pass


def _aten_grouped_mm(mat_a: TReal, mat_b: TReal, offs: INT64, bias=None, out_dtype=None) -> TReal:
    pass


def _aten_repeat_interleave_self_int(self, repeats, dim=None, output_size=None):
    pass


def _operator_floordiv(self, other):
    pass


def _aten_masked_fill(self, mask, value):
    pass


_ONNX_TRANSLATION_TABLE: dict[Any, Any] = {}
if is_onnxscript_available():
    _ONNX_TRANSLATION_TABLE.update(
        {
            torch.ops.aten.bincount.default: _aten_bincount,
            torch.ops.aten.index_put.default: _aten_index_put,
            torch.ops.aten._grouped_mm.default: _aten_grouped_mm,
            torch.ops.transformers.grouped_mm_fallback.default: _aten_grouped_mm,
            torch.ops.aten.repeat_interleave.self_int: _aten_repeat_interleave_self_int,
            torch.ops.aten.masked_fill.Scalar: _aten_masked_fill,
            torch.ops.aten.masked_fill.Tensor: _aten_masked_fill,
            operator.floordiv: _operator_floordiv,
        }
    )




def _fix_ir_topk_sorted(graph_like: onnx_ir.Graph) -> None:
    pass


_IR_FIXES = [
    _fix_ir_topk_sorted,
]


def apply_onnx_ir_fixes(onnx_program: ONNXProgram) -> None:
    pass
