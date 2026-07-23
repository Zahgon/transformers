

from __future__ import annotations

import contextlib
import copy
import enum
import functools
import inspect
import sys
from collections.abc import MutableMapping
from typing import Any

from ..utils import logging
from ..utils.generic import get_max_seqlen
from ..utils.import_utils import is_torch_available


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

    from ..modeling_utils import PreTrainedModel
    from ..vision_utils import (
        get_vision_attention_seqlens,
        get_vision_bilinear_indices_and_weights,
        get_vision_merged_shape,
        get_vision_nearest_position_ids,
        get_vision_position_ids,
        get_vision_window_index,
    )



_PATCHES: dict[str, list[tuple[Any, str, callable]]] = {}
_FX_NODE_FIXES: dict[str, list[callable]] = {}
_FX_PROGRAM_FIXES: dict[str, list[callable]] = {}


@contextlib.contextmanager
def patch_attribute(obj: Any, attribute: str, factory: Any):
    """Swap `obj.<attribute>` with `factory(original)` for the duration of the block."""
    original = getattr(obj, attribute)
    setattr(obj, attribute, factory(original))
    try:
        yield
    finally:
        setattr(obj, attribute, original)


@contextlib.contextmanager
def patch_attributes(patches: list[tuple[Any, str, callable]]):
    """Install `(obj, attribute, factory)` patches for the duration of the block.

    Plural form of `patch_attribute` — each `factory(original)` returns the replacement
    callable. Originals are restored on exit, even if the body raises.
    """
    with contextlib.ExitStack() as stack:
        for obj, attribute, factory in patches:
            stack.enter_context(patch_attribute(obj, attribute, factory))
        yield


@contextlib.contextmanager
def apply_patches(backend: str):
    """Install `_PATCHES[backend]` for the duration of the block."""
    with patch_attributes(_PATCHES.get(backend, [])):
        yield


def register_fx_node_fix(backend: str):
    """Append the decorated `(gm, node) -> bool` fix to `_FX_NODE_FIXES[backend]`."""

    def decorator(fn):
        pass

    return decorator


def register_fx_program_fix(backend: str):
    """Append the decorated `(exported_program) -> None` fix to `_FX_PROGRAM_FIXES[backend]`.

    Use this for fixes that need program-level context (range_constraints, graph_signature,
    state_dict) — the per-node `_FX_NODE_FIXES` shape only sees one node at a time.
    """

    def decorator(fn):
        pass

    return decorator


def apply_fx_program_fixes(backend: str, exported_program) -> None:
    pass


def register_patch(backend: str, *paths: str):
    """Append the decorated `factory(original)` to `_PATCHES[backend]`, once per `path`.

    Each `path` is a dotted Python path like `"torch.where"`, `"torch.Tensor.unsqueeze"`,
    or `"transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeTop2Router._cast_classifier"`.
    The rightmost segment is the attribute to swap; the rest is the object that owns it.
    Paths are resolved at decoration time — submodules are imported as needed, falling
    back to `getattr` for class attributes. A path that fails to resolve (e.g. the backend
    isn't installed) is silently skipped so the module still imports.

    Passing multiple paths registers the SAME factory against each — useful for swapping
    the same method or torch op across several call sites (e.g. ``torch.unsqueeze`` +
    ``torch.Tensor.unsqueeze``, or one vision-attention forward across N model classes).
    """

    def decorator(fn):
        pass

    return decorator


def _resolve_dotted_path(path: str):
    """Resolve a dotted Python path to the actual object — importing submodules where
    possible, falling back to `getattr` for class attributes (e.g. `torch.Tensor`).
    Returns `None` if the path can't be resolved (e.g. the backend isn't installed)."""
    import importlib

    parts = path.split(".")
    try:
        obj = importlib.import_module(parts[0])
        for part in parts[1:]:
            try:
                obj = importlib.import_module(f"{obj.__name__}.{part}")
            except (ImportError, AttributeError):
                obj = getattr(obj, part)
        return obj
    except (ImportError, AttributeError):
        return None


def apply_fx_node_fixes(backend: str, graph_module) -> None:
    """Walk every call_function node and apply the first matching `_FX_NODE_FIXES[backend]`
    fix, then DCE.

    Each fix has signature `(gm, node) -> bool`. Returning `True` means the fix consumed
    the node — no further fixes run against it. Fixes are expected to be disjoint by
    `node.target`; if multiple could apply, list order decides.

    After the walk, `Graph.eliminate_dead_code` runs on every sub-GraphModule and
    `gm.recompile()` is called once. PyTorch DCE occasionally raises `SystemError` /
    `KeyError` from `erase_node._update_args_kwargs` on orphaned symbolic-size nodes —
    we swallow both; any survivors are handled by the downstream backend optimizer.
    """
    fixes = _FX_NODE_FIXES.get(backend, [])
    for gm in graph_module.modules():
        if not isinstance(gm, torch.fx.GraphModule):
            continue
        for node in list(gm.graph.nodes):
            if node.op != "call_function":
                continue
            for fix in fixes:
                if fix(gm, node):
                    break
        try:
            gm.graph.eliminate_dead_code()
            gm.recompile()
        except (SystemError, KeyError):
            pass



_LEAF_SKIP_TYPES: tuple[type, ...] = (type,)
if is_torch_available():
    _LEAF_SKIP_TYPES += (enum.Enum, torch.SymInt, torch.SymFloat, torch.SymBool)


def _map_leaf_tensors(obj: Any, fn: callable) -> Any:
    pass


def _iter_leaf_tensors(obj: Any, prefix: str = ""):
    """Yield `(dotted_path, tensor)` for every tensor in a nested structure."""
    if isinstance(obj, _LEAF_SKIP_TYPES):
        return
    if isinstance(obj, torch.Tensor):
        yield prefix or "output", obj
    elif isinstance(obj, (list, tuple, set)):
        for index, item in enumerate(obj):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _iter_leaf_tensors(item, path)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from _iter_leaf_tensors(value, path)
    elif hasattr(obj, "__dict__"):
        yield from _iter_leaf_tensors(vars(obj), prefix)




def get_leaf_tensors(obj: Any) -> dict[str, torch.Tensor]:
    """Recursively retrieve all leaf tensors from a potentially nested structure.

    Args:
        obj (`Any`):
            A tensor, dataclass, dict, list, tuple, or any nesting thereof.

    Returns:
        `dict[str, torch.Tensor]`: Flat mapping from dotted path strings to tensors.
    """
    return dict(_iter_leaf_tensors(obj))


def duplicate_leaf_tensors(obj: Any) -> Any:
    pass


def cast_leaf_tensors(obj: Any, dtype: torch.dtype, device: torch.device) -> Any:
    pass


def module_device(model: PreTrainedModel | torch.nn.Module) -> torch.device | None:
    pass


def module_dtype(model: PreTrainedModel | torch.nn.Module) -> torch.dtype | None:
    pass


_OUTPUT_FLAGS = ("use_cache", "output_attentions", "output_hidden_states", "return_dict", "return_loss")


def prepare_for_export(
    model: PreTrainedModel | torch.nn.Module, inputs: MutableMapping[str, Any]
) -> tuple[PreTrainedModel | torch.nn.Module, MutableMapping[str, Any], dict[str, Any]]:
    pass




def _find_submodule_attr(model: torch.nn.Module, name: str) -> Any | None:
    pass


_EXPORT_INPUT_PREPARERS: dict[tuple[str, ...], callable] = {}


def register_export_input_preparer(*markers: str):
    """Register `fn(model, inputs) -> None`. Dispatched when every `marker` is a key in
    `inputs` with a non-`None` value — no model_type list to maintain. Use multiple
    markers to narrow the match when a single kwarg is too ambiguous (e.g.
    `("input_features", "feature_lens")` for omni audio encoders)."""

    def decorator(fn):
        pass

    return decorator


@register_export_input_preparer("grid_thw")
def _prepare_grid_thw_vision_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    pass


@register_export_input_preparer("target_sizes")
def _prepare_navit_vision_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    pass


@register_export_input_preparer("input_features", "feature_lens")
def _prepare_omni_audio_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    pass


@register_export_input_preparer("input_features", "input_features_mask")
def _prepare_qwen3_asr_audio_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    pass


def precompute_export_inputs(model: torch.nn.Module, inputs: dict[str, Any]) -> None:
    pass




@contextlib.contextmanager
def _capture_forward(module: torch.nn.Module):
    """Capture forward call kwargs into a list (one dict per call).

    Positional args are normalised to kwargs via `inspect.signature` so the
    captured dicts can be passed directly as `kwargs=inputs` to `torch.export`.
    """

    calls: list[dict] = []
    original = module.forward
    sig = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        captured = {}
        bound = sig.bind(*args, **kwargs)
        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                captured.update(copy.deepcopy(value))
            elif param.kind != inspect.Parameter.VAR_POSITIONAL:
                captured[name] = copy.deepcopy(value)
        calls.append(captured)
        return original(*args, **kwargs)

    module.forward = wrapper
    try:
        yield calls
    finally:
        module.forward = original


def decompose_prefill_decode(
    model: PreTrainedModel,
    inputs: dict[str, Any],
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Run `model.generate()` for 2 tokens and capture prefill and decode inputs.

    Reuses the full generation machinery so every architecture (decoder-only, SSM,
    encoder-decoder, multi-modal, …) gets correct inputs without reimplementing the loop.

    Returns:
        `dict[str, tuple[torch.nn.Module, dict]]`:
        `{"prefill": (model, prefill_inputs), "decode": (model, decode_inputs)}`
    """
    try:
        with _capture_forward(model) as calls:
            model.generate(**copy.deepcopy(inputs), max_new_tokens=2, min_new_tokens=2)
    except Exception as e:
        raise RuntimeError(
            f"decompose_prefill_decode failed for {type(model).__name__}. "
            f"Inputs passed: {list(inputs.keys())}. "
            f"Make sure the inputs are compatible with model.generate()."
        ) from e

    if len(calls) < 2:
        raise RuntimeError(
            f"decompose_prefill_decode expected at least 2 calls to {type(model).__name__}.forward() "
            f"during generate(max_new_tokens=2), but captured {len(calls)}. This likely means "
            "generate() bypasses the top-level forward() (e.g. delegates to an inner model), "
            "so prefill/decode decomposition is not supported for this architecture."
        )

    return {
        "prefill": (copy.copy(model), calls[0]),
        "decode": (copy.copy(model), calls[1]),
    }


_MULTIMODAL_PROJECTOR_NAMES = ("multi_modal_projector", "connector", "embed_vision", "embed_audio")
_MULTIMODAL_LM_HEAD_NAMES = ("lm_head",)


def _find_multimodal_submodules(model: PreTrainedModel) -> dict[str, torch.nn.Module]:
    """Return `{attr_name: module}` for multi-modal submodules found on `model`.

    Uses the canonical `PreTrainedModel.get_encoder("image"/"audio")` and `get_decoder()`
    accessors for encoders and the language model. Projectors and `lm_head` are looked
    up by name on `model` and its `base_model` (e.g. `LlavaModel` under `LlavaForConditionalGeneration`).

    Only returns results when at least one modal encoder AND a language model are found —
    otherwise the model is not multi-modal and should be exported as a single unit.
    """
    found: dict[str, torch.nn.Module] = {}

    has_encoder = False
    for modality in ("image", "audio"):
        encoder = model.get_encoder(modality=modality)
        if encoder is not None and encoder is not model:
            found[f"{modality}_encoder"] = encoder
            has_encoder = True

    decoder = model.get_decoder()
    if decoder is not None and decoder is not model:
        found["language_model"] = decoder

    for root in {model, model.base_model}:
        for name in _MULTIMODAL_PROJECTOR_NAMES + _MULTIMODAL_LM_HEAD_NAMES:
            if name not in found and getattr(root, name, None) is not None:
                found[name] = getattr(root, name)

    if not has_encoder or "language_model" not in found:
        return {}

    return found


def is_multimodal(model: PreTrainedModel) -> bool:
    """Returns `True` if the model is multi-modal with modal encoders and a language model."""
    return bool(_find_multimodal_submodules(model))


def decompose_multimodal(model: PreTrainedModel, inputs: dict[str, Any]) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Capture inputs to each multi-modal submodule via a single forward pass.

    Detects all known multi-modal submodules by attribute name (vision tower, projector,
    language model, lm_head, …) and captures their forward kwargs during one
    `model(**inputs)` call.

    Each submodule is returned as a separate `name: (module, inputs)` entry for
    independent export. The token-merge step (e.g. `masked_scatter` for multi-modal models)
    is intentionally left outside the exported graphs — it is the caller's responsibility
    to assemble `inputs_embeds` from the encoder outputs before running the decoder.

    Returns:
        `dict[str, tuple[torch.nn.Module, dict]]`: One `name: (module, inputs)`
        entry per detected submodule (image/audio encoder, projector, language model, lm_head).

    Raises:
        `ValueError`: if no known multi-modal submodules are found on the model.
    """
    submodules = _find_multimodal_submodules(model)
    if not submodules:
        raise ValueError(
            f"decompose_multimodal found no multi-modal submodules on {type(model).__name__}. "
            f"Expected an image/audio encoder + language model, found neither."
        )

    try:
        with contextlib.ExitStack() as stack, torch.no_grad():
            submodule_inputs = {
                name: stack.enter_context(_capture_forward(module)) for name, module in submodules.items()
            }
            model(**copy.deepcopy(inputs))
    except Exception as e:
        raise RuntimeError(
            f"decompose_multimodal failed for {type(model).__name__}. Inputs passed: {list(inputs.keys())}."
        ) from e

    return {
        name: (module, submodule_inputs[name][-1])
        for name, module in submodules.items()
        if submodule_inputs[name]  # skip submodules not called (e.g. lm_head on base models)
    }


def decompose_for_generation(
    model: PreTrainedModel, inputs: dict[str, Any]
) -> dict[str, tuple[torch.nn.Module, dict]]:
    """Decompose a generative model into independently exportable `(model, forward_inputs)` pairs.

    Runs `decompose_prefill_decode` to capture prefill and decode forward kwargs from a real
    `model.generate(**inputs, max_new_tokens=2)`. If the prefill is multi-modal (per `is_multimodal`),
    further splits it into one entry per submodule (vision/audio encoder, projector, language model,
    `lm_head`) via `decompose_multimodal`.

    Args:
        model: Generative model. Must support `model.generate(**inputs)`.
        inputs: **Generate** kwargs — what you'd pass to `model.generate(**inputs)`.

    Returns:
        `{component_name: (submodel, forward_inputs)}`. Keys are `"prefill"` / `"decode"` for
        plain generative models and `"<modality>_encoder"` / `"multi_modal_projector"` /
        `"language_model"` / `"lm_head"` / `"decode"` for multi-modal generative models.
    """
    stages = decompose_prefill_decode(model, inputs)
    prefill_model, prefill_inputs = stages["prefill"]

    if not is_multimodal(prefill_model):
        return stages

    components = decompose_multimodal(prefill_model, prefill_inputs)
    components["decode"] = stages["decode"]
    return components
