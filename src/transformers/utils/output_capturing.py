
from __future__ import annotations

import threading
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING

from .import_utils import is_torchdynamo_compiling, requires


if TYPE_CHECKING:
    from torch import nn

    from ..modeling_utils import PreTrainedModel


_CAN_RECORD_REGISTRY = {}


@dataclass
@requires(backends=("torch",))
class OutputRecorder:

    target_class: type[nn.Module]
    index: int = 0
    layer_name: str | None = None
    class_name: str | None = None
    capture_initial_hidden_state: bool = True


class CompileableContextVar:

    def __init__(self, name):
        self.context_var = ContextVar(name, default=None)
        self.global_var = None
        self.compiling = False

    def get(self):
        if self.compiling:
            return self.global_var
        else:
            return self.context_var.get()

    def set(self, value):
        pass

    def reset(self, token):
        if self.compiling or token is None:
            self.global_var = None
            self.compiling = False
        else:
            self.context_var.reset(token)


_active_collector = CompileableContextVar("output_collector")


def install_output_capuring_hook(
    module: nn.Module, key: str, index: int, capture_initial_hidden_state: bool = True
) -> None:
    """Install the forward hook needed to capture the output described by `key` and `index` in `module`."""

    def output_capturing_hook(module, args, output):
        pass

    module.register_forward_hook(output_capturing_hook)


def recursively_install_hooks(
    parent_module: nn.Module, module_name: str, capture_tasks: list[tuple[str, OutputRecorder]]
) -> None:
    """
    Recursively install all output capturing hooks on all submodules of `parent_module`.
    Note that we need to use this recursive approach instead of simply iterating over all modules, because we want
    to respect the `capture_tasks` of all individual submodels (`PreTrainedModel` instances) in the graph. That is, once
    we reach a submodel in the graph, its children should use this submodel's `capture_tasks`, but other parts of the graph
    should not.
    """
    from ..modeling_utils import PreTrainedModel

    for name, module in parent_module.named_children():
        if not isinstance(module, PreTrainedModel):
            recursively_install_hooks(module, f"{module_name}.{name}", capture_tasks)
        else:
            install_all_output_capturing_hooks(module, prefix=f"{module_name}.{name}")

    for key, specs in capture_tasks:
        match_target_class = specs.target_class is not None and isinstance(parent_module, specs.target_class)
        match_class_name = specs.class_name is not None and module_name.endswith(specs.class_name)

        if match_target_class or match_class_name:
            if specs.layer_name is not None:
                target_layer_name = specs.layer_name.strip(".")
                target_layer_name = "." + target_layer_name + "."
                matches = target_layer_name in module_name + "."
                if not matches:
                    continue

            install_output_capuring_hook(parent_module, key, specs.index, specs.capture_initial_hidden_state)


def install_all_output_capturing_hooks(model: PreTrainedModel, prefix: str | None = None) -> None:
    """
    Install the output recording hooks on all the modules in `model`. This will take care of correctly dispatching
    the `_can_record_outputs` property of each individual submodels in case of composite models.
    """
    capture_flags = _CAN_RECORD_REGISTRY.get(str(model.__class__)) or {}  # there is a weak ref for executorch

    capture_tasks = []
    for key, layer_specs in capture_flags.items():
        if not isinstance(layer_specs, list):
            layer_specs = [layer_specs]
        for specs in layer_specs:
            if not isinstance(specs, OutputRecorder):
                index = 0 if "hidden_states" in key else 1
                class_name = None if not isinstance(specs, str) else specs
                target_class = specs if not isinstance(specs, str) else None
                specs = OutputRecorder(target_class=target_class, index=index, class_name=class_name)
            capture_tasks.append((key, specs))

    prefix = prefix if prefix is not None else ""
    recursively_install_hooks(model, prefix, capture_tasks)
    setattr(model, "_output_capturing_hooks_installed", True)


_hook_installation_lock = threading.Lock()


def maybe_install_capturing_hooks(model: PreTrainedModel) -> None:
    """
    Check if the model already has output capturing hooks installed, and install them if it is not already the
    case.
    Note that this is thread-safe, in case 2 (or more) threads want to install them concurrently.
    """
    if getattr(model, "_output_capturing_hooks_installed", False):
        return

    with _hook_installation_lock:
        if getattr(model, "_output_capturing_hooks_installed", False):
            return
        install_all_output_capturing_hooks(model)


def capture_outputs(func=None, *, tie_last_hidden_states=True):
    """
    Decorator to intercept specific layer outputs through hooks. The hooks are installed only once and lazily,
    the first time output capture is requested with the `output_xxx` kwargs/config.
    The implementation is fully context/thread safe, except when using `torch.compile`, as dynamo is unable to trace
    through `ContextVar` methods.

    Args:
        tie_last_hidden_states (`bool`, *optional*, defaults to `True`):
            Whether to overwrite `out.hidden_states[-1]` with the `out.last_hidden_state`.
            This is true for all language models and should be toggled off only if
            `out.hidden_states[-1]` has to be the hidden state before last layer norm, which
            is needed for some vision models (e.g. CLIP, SigLIP)
    """

    def wrapped_fn(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return_dict = kwargs.pop("return_dict", getattr(self.config, "return_dict", True))

            capturable_flags = _CAN_RECORD_REGISTRY.get(str(self.__class__)) or {}
            recordable_keys = {
                f"output_{k}": kwargs.get(f"output_{k}", getattr(self.config, f"output_{k}", False))
                for k in capturable_flags
            }
            if "cross_attentions" in capturable_flags:
                recordable_keys["output_cross_attentions"] = kwargs.get(
                    "output_attentions", getattr(self.config, "output_attentions", False)
                )
            if "mask_decoder_attentions" in capturable_flags:
                recordable_keys["output_mask_decoder_attentions"] = kwargs.get(
                    "output_attentions", getattr(self.config, "output_attentions", False)
                )

            collected_outputs = {k.replace("output_", ""): [] for k, v in recordable_keys.items() if v}
            if len(collected_outputs) > 0:
                maybe_install_capturing_hooks(self)
            output_token = _active_collector.set(collected_outputs)

            try:
                outputs = func(self, *args, **kwargs)
            finally:
                _active_collector.reset(output_token)

            for key in collected_outputs:
                if key == "hidden_states":
                    if not tie_last_hidden_states:
                        pass
                    elif hasattr(outputs, "vision_hidden_states"):
                        collected_outputs[key] = collected_outputs[key][:-1]
                        collected_outputs[key].append(outputs.vision_hidden_states)
                    elif hasattr(outputs, "last_hidden_state"):
                        collected_outputs[key] = collected_outputs[key][:-1]
                        collected_outputs[key].append(outputs.last_hidden_state)

                outputs[key] = tuple(collected_outputs[key])

            if return_dict is False:
                outputs = outputs.to_tuple()

            return outputs

        return wrapper

    if func is not None:
        return wrapped_fn(func)
    return wrapped_fn
