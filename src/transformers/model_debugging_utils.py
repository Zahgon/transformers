
import functools
import json
import os
import re
from contextlib import contextmanager, redirect_stdout
from io import StringIO

from .utils import logging
from .utils.import_utils import is_torch_available, requires


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    _torch_distributed_available = False
    if torch.distributed.is_available():
        import torch.distributed.tensor

        _torch_distributed_available = True
else:
    _torch_distributed_available = False


logger = logging.get_logger(__name__)


def _is_rank_zero():
    """Return True if rank=0 or we aren't running distributed."""
    if not (_torch_distributed_available and torch.distributed.is_initialized()):
        return True
    return torch.distributed.get_rank() == 0


MEMORY_ADDRESS_REGEX = re.compile(r"object at 0x[0-9A-Fa-f]+")


def _sanitize_repr_for_diff(x_str: str) -> str:
    """
    Replace memory addresses in an object's repr with a stable placeholder
    so that beautiful JSON diffs won't be ruined by ephemeral addresses.
    """
    return MEMORY_ADDRESS_REGEX.sub("object at 0xXXXXXXXX", x_str)


def _dtensor_repr(x):
    pass


def _serialize_tensor_like_io(
    value, debug_path: str | None = None, use_repr: bool = True, path_to_value: str | None = None
):
    """
    Converts Tensors and DTensors to a JSON-serializable dictionary representation.

    Args:
        value: Any Python object, often including torch Tensors, lists, dicts, etc.
        debug_path (`str`, *optional*, defaults to `None`): Directory to dump debug JSON and SafeTensors files.
        use_repr (bool, *optional*, defaults to `True`): Whether to save a `repr()`-ized version of the tensor as the
            `value` property in the asscoiated FULL_TENSORS.json file, or to store the full tensors in separate
            SafeTensors file and store the relative path to that file in the `value` property in the dictionary.
        path_to_value (`str`, *optional*, defaults to `None`): The file name for the SafeTensors file holding the full
            tensor value if `use_repr=False`.

    Returns:
        A nested Python structure (list, dict, or sanitized string) that is safe to json.dump.
    """
    torch.set_printoptions(sci_mode=True)

    if use_repr:
        value_out = _repr_to_list(value)
    elif path_to_value:
        if not path_to_value.endswith(".safetensors"):
            path_to_value += ".safetensors"

        filepath = os.path.join(debug_path, path_to_value) if debug_path else path_to_value
        save_file({"data": value.contiguous().detach().cpu()}, filepath)
        value_out = f"./{path_to_value}"
    else:
        raise ValueError(f"{use_repr=} and {path_to_value=} cannot both be falsy.")

    out = {
        "shape": repr(value.shape),
        "dtype": repr(value.dtype),
        "value": value_out,
    }
    if value.dtype in {torch.float16, torch.float32, torch.bfloat16}:
        out.update(
            {
                "mean": _sanitize_repr_for_diff(repr(value.mean())),
                "std": _sanitize_repr_for_diff(repr(value.std())),
                "min": _sanitize_repr_for_diff(repr(value.min())),
                "max": _sanitize_repr_for_diff(repr(value.max())),
            }
        )
    return out


def _serialize_io(value, debug_path: str | None = None, use_repr: bool = True, path_to_value: str | None = None):
    """
    Recursively build a JSON-serializable Python structure from `value`.
    Tensors and DTensors become either sanitized repr strings, or are saved to disk as SafeTensors files and their
    relative paths are recorded in the returned Python structure.
    Lists/tuples/dicts are recursed into.
    All memory addresses are replaced with a stable placeholder.

    Args:
        value: Any Python object, often including torch Tensors, lists, dicts, etc.
        debug_path (`str`, *optional*, defaults to `None`): Directory to dump debug JSON and SafeTensors files.
        use_repr (bool, *optional*, defaults to `True`): Whether to save a `repr()`-ized version of the tensors as the
            `value` property in the asscoiated FULL_TENSORS.json file, or to store full tensors in separate SafeTensors
            files and store the relative path to that file in the `value` property.
        path_to_value (`str`, *optional*, defaults to `None`): The file name for the SafeTensors file holding the full
            tensor value if `use_repr=False`.

    Returns:
        A nested Python structure (list, dict, or sanitized string) that is safe to json.dump.
    """
    if isinstance(value, (list, tuple)):
        return [
            _serialize_io(v, debug_path=debug_path, use_repr=use_repr, path_to_value=f"{path_to_value}_{i}")
            for i, v in enumerate(value)
        ]

    if isinstance(value, dict):
        return {
            k: _serialize_io(v, debug_path=debug_path, use_repr=use_repr, path_to_value=f"{path_to_value}_{k}")
            for k, v in value.items()
        }

    if hasattr(value, "_local_tensor"):
        return _serialize_tensor_like_io(
            value._local_tensor, debug_path=debug_path, use_repr=use_repr, path_to_value=path_to_value
        )

    if isinstance(value, torch.Tensor):
        return _serialize_tensor_like_io(value, debug_path=debug_path, use_repr=use_repr, path_to_value=path_to_value)

    return _sanitize_repr_for_diff(repr(value))


def _repr_to_list(value: torch.Tensor):
    """
    Converts a tensor into a sanitized multi-line string representation.

    Args:
        value (`torch.Tensor`): The tensor to represent.

    Returns:
        `list[str]`: List of string lines representing the tensor.
    """
    torch.set_printoptions(sci_mode=True, linewidth=120)
    with StringIO() as buf, redirect_stdout(buf):
        print(value)  # to redirected stdout to avoid line splits
        raw = buf.getvalue()
    return _sanitize_repr_for_diff(raw).splitlines()


def prune_outputs_if_children(node):
    if node.get("children"):
        node.pop("outputs", None)
        for child in node["children"]:
            prune_outputs_if_children(child)


LAYER_SUFFIX_RE = re.compile(r"(.*)\.(\d+)$")  # should be generic enough, ends with a number


def is_layer_block(node):
    """
    Checks whether a node represents a layer block with submodules.

    Args:
        node (`dict`): A node from the call tree.

    Returns:
        `bool`: Whether the node is a layer block.
    """
    match = LAYER_SUFFIX_RE.match(node.get("module_path", ""))
    if not match or not node.get("children"):
        return False
    number = match.group(2)
    return any(f".{number}." in child.get("module_path", "") for child in node["children"])


def prune_intermediate_layers(node):
    """
    Recursively removes intermediate layers from the tree to improve readability.
    Keeps at least the first and last layers if many consecutive layers are present.

    Args:
        node (`dict`): The root or subnode to prune recursively.
    """
    if not node.get("children"):
        return
    layer_blocks = [(i, child) for i, child in enumerate(node["children"]) if is_layer_block(child)]

    if len(layer_blocks) > 2:
        to_remove = [i for i, _ in layer_blocks[1:-1]]
        node["children"] = [child for i, child in enumerate(node["children"]) if i not in to_remove]

    for child in node["children"]:
        prune_intermediate_layers(child)


def log_model_debug_trace(debug_path: str | None, model):
    if debug_path:
        try:
            os.makedirs(debug_path, exist_ok=True)
            base = os.path.join(debug_path, model._debugger_module_dump_name + "_debug_tree")
        except Exception as e:
            raise ValueError(f"Unexpected or existing debug_path={debug_path}.") from e
    else:
        base = model._debugger_module_dump_name + "_debug_tree"

    logger.info(f"Writing model trace at {base}.json")
    full_path = base + "_FULL_TENSORS.json"
    summary_path = base + "_SUMMARY.json"

    prune_outputs_if_children(model._call_tree)

    with open(full_path, "w") as f:
        json.dump(model._call_tree, f, indent=2)

    def strip_values(node):
        def clean(val):
            if isinstance(val, dict):
                val.pop("value", None)
                for v in val.values():
                    clean(v)
            elif isinstance(val, list):
                for item in val:
                    clean(item)

        clean(node.get("inputs", {}))
        clean(node.get("outputs", {}))

        for child in node.get("children", []):
            strip_values(child)

    tree_copy = json.loads(json.dumps(model._call_tree))  # deep copy
    strip_values(tree_copy)

    with open(summary_path, "w") as f:
        json.dump(tree_copy, f, indent=2)


def _attach_debugger_logic(
    model,
    debug_path: str = ".",
    do_prune_layers: bool = True,
    use_repr: bool = True,
):
    """
    Attaches a debugging wrapper to every module in the model.

    This records structured inputs and outputs during the forward pass into a call tree.

    Args:
        model (`PreTrainedModel`, `nn.Module`): Model to wrap.
        debug_path (`str`): Optional directory to dump debug JSON files.
        do_prune_layers (`bool`, *optional*, defaults to `True`): Whether to prune intermediate layers.
        use_repr (bool, *optional*, defaults to `True`): Whether to save a `repr()`-ized version of the tensors as the
            `value` property in the associated FULL_TENSORS.json file, or to store full tensors in separate SafeTensors
            files and store the relative path to that file in the `value` property.
    """
    class_name = model.__class__.__name__

    model._call_tree = {"module_path": class_name, "inputs": None, "outputs": None, "children": []}
    model._debugger_model_call_stack = []
    model._debugger_module_dump_name = class_name  # used for final JSON filename

    if debug_path:
        try:
            os.makedirs(debug_path, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Unexpected or existing debug_path={debug_path}.") from e

    def wrap_forward(module, full_path):
        orig_forward = module.forward

        @functools.wraps(orig_forward)
        def wrapped_forward(*inps, **kws):
            pass

        module.forward = wrapped_forward

    for name, submodule in model.named_modules():
        if name == "":
            continue
        wrap_forward(submodule, f"{class_name}.{name}")

    real_top_forward = model.forward

    @functools.wraps(real_top_forward)
    def top_wrapped_forward(*inps, **kws):
        pass

    model.forward = top_wrapped_forward


@requires(backends=("torch",))
@contextmanager
def model_addition_debugger_context(
    model,
    debug_path: str | None = None,
    do_prune_layers: bool = True,
    use_repr: bool = True,
):
    """
    # Model addition debugger - context manager for model adders
    This context manager is a power user tool intended for model adders.

    It tracks all forward calls within a model forward and logs a slice of each input and output on a nested JSON file.
    If `use_repr=True` (the default), the JSON file will record a `repr()`-ized version of the tensors as a list of
    strings. If `use_repr=False`, the full tensors will be stored in separate SafeTensors files and the JSON file will
    provide a relative path to that file.

    To note, this context manager enforces `torch.no_grad()`.

    ## Usage

    add the context manager to a model to debug

    ```python
    import torch

    from PIL import Image
    from transformers import LlavaProcessor, LlavaForConditionalGeneration, model_addition_debugger_context

    torch.random.manual_seed(673)

    # load pretrained model and processor
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = LlavaProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(model_id)

    # create random image input
    random_image = Image.fromarray(torch.randint(0, 256, (224, 224, 3), dtype=torch.uint8).numpy())

    # prompt
    prompt = "<image>Describe this image."

    # process inputs
    inputs = processor(text=prompt, images=random_image, return_tensors="pt")

    # call forward method (not .generate!)
    with model_addition_debugger_context(model, debug_path="Your_debug_path", do_prune_layers=False):
        output = model.forward(**inputs)
    ```

    """
    orig_forwards = {m: m.forward for _, m in model.named_modules()}
    orig_forwards[model] = model.forward
    _attach_debugger_logic(model, debug_path, do_prune_layers, use_repr)
    try:
        yield model
    finally:
        for module_instance, forward_method in orig_forwards.items():
            module_instance.forward = forward_method
