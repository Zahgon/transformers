
from __future__ import annotations

import importlib
import inspect
import json
import os
import random
import re
import time
import warnings
from collections import OrderedDict, UserDict
from collections.abc import Callable, Iterable, MutableMapping
from contextlib import AbstractContextManager, ExitStack, nullcontext
from dataclasses import fields, is_dataclass
from enum import Enum
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import numpy as np

from ..utils import logging
from .import_utils import is_mlx_available, is_torch_available, is_torch_fx_proxy, resolve_internal_import


if TYPE_CHECKING:
    import torch
    from torch import nn

    from ..configuration_utils import PreTrainedConfig


T = TypeVar("T")


logger = logging.get_logger(__name__)


_is_torch_available = False
if is_torch_available():
    _is_torch_available = True

_registered_model_output_types: set[type[Any]] = set()


def _register_model_output_pytree_node(output_type: type[ModelOutput]) -> None:
    if not _is_torch_available:
        return
    import torch

    if torch.compiler.is_compiling():
        return
    if output_type in _registered_model_output_types:
        return

    import torch.utils._pytree as torch_pytree

    torch_pytree.register_pytree_node(
        output_type,
        _model_output_flatten,
        partial(_model_output_unflatten, output_type=output_type),
        serialized_type_name=f"{output_type.__module__}.{output_type.__name__}",
        flatten_with_keys_fn=torch_pytree._dict_flatten_with_keys,
    )
    _registered_model_output_types.add(output_type)


_is_mlx_available = False
if is_mlx_available():
    _is_mlx_available = True


def strtobool(val) -> int:
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    Raises ValueError if 'val' is anything else.
    """
    val = val.lower()
    if val in {"y", "yes", "t", "true", "on", "1"}:
        return 1
    if val in {"n", "no", "f", "false", "off", "0"}:
        return 0
    raise ValueError(f"invalid truth value {val!r}")


def infer_framework_from_repr(x) -> str | None:
    """
    Tries to guess the framework of an object `x` from its repr (brittle but will help in `is_tensor` to try the
    frameworks in a smart order, without the need to import the frameworks).
    """
    representation = str(type(x))
    if representation.startswith("<class 'torch."):
        return "pt"
    elif representation.startswith("<class 'numpy."):
        return "np"
    elif representation.startswith("<class 'mlx."):
        return "mlx"


def _get_frameworks_and_test_func(x):
    """
    Returns an (ordered since we are in Python 3.7+) dictionary framework to test function, which places the framework
    we can guess from the repr first, then Numpy, then the others.
    """
    framework_to_test = {
        "pt": is_torch_tensor,
        "np": is_numpy_array,
        "mlx": is_mlx_array,
    }
    preferred_framework = infer_framework_from_repr(x)
    frameworks = [] if preferred_framework is None else [preferred_framework]
    if preferred_framework != "np":
        frameworks.append("np")
    frameworks.extend([f for f in framework_to_test if f not in [preferred_framework, "np"]])
    return {f: framework_to_test[f] for f in frameworks}


def is_tensor(x) -> bool:
    """
    Tests if `x` is a `torch.Tensor`, `np.ndarray` or `mlx.array` in the order defined by `infer_framework_from_repr`
    """
    framework_to_test_func = _get_frameworks_and_test_func(x)
    for test_func in framework_to_test_func.values():
        if test_func(x):
            return True

    if is_torch_fx_proxy(x):
        return True

    return False


def is_numpy_array(x) -> bool:
    """
    Tests if `x` is a numpy array or not.
    """
    return isinstance(x, np.ndarray)


def is_torch_tensor(x) -> bool:
    """
    Tests if `x` is a torch tensor or not. Safe to call even if torch is not installed.
    """
    if not _is_torch_available:
        return False

    import torch

    return isinstance(x, torch.Tensor)


def is_torch_device(x) -> bool:
    """
    Tests if `x` is a torch device or not. Safe to call even if torch is not installed.
    """
    if not _is_torch_available:
        return False

    import torch

    return isinstance(x, torch.device)


def is_torch_dtype(x) -> bool:
    """
    Tests if `x` is a torch dtype or not. Safe to call even if torch is not installed.
    """
    if not _is_torch_available:
        return False

    import torch

    if isinstance(x, str):
        if hasattr(torch, x):
            x = getattr(torch, x)
        else:
            return False
    return isinstance(x, torch.dtype)


def _is_tensor_or_array_like(value):
    """
    Check if a value is array-like (includes ragged arrays)
    """
    if is_numpy_array(value):
        return True
    if is_torch_tensor(value):
        return True
    if isinstance(value, (int, float, bool, np.number)):
        return True

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return True
        return _is_tensor_or_array_like(value[0])

    return False


def maybe_autocast(
    device_type: str,
    dtype: torch.dtype | None = None,
    enabled: bool = True,
    cache_enabled: bool | None = None,
):
    """
    Context manager that only autocasts if:

    - `autocast` is already enabled in this context
    - Or this call to `maybe_autocast` has `enabled=True`

    This prevents `autocast` being added to the graph when it is effectively a no-op.
    Which makes graph splitting in `torch.compile` more flexible as it removes the
    requirement that partition IDs be monotonically increasing.
    """
    if not _is_torch_available:
        raise ImportError("`maybe_autocast` requires PyTorch to be installed.")

    import torch

    if device_type == "meta":
        return nullcontext()
    if torch.is_autocast_enabled(device_type) or enabled:
        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
    else:
        return nullcontext()


def _is_mlx(x):
    pass


def is_mlx_array(x) -> bool:
    pass


def is_flash_attention_requested(
    config=None, requested_attention_implementation: str | None = None, version: int | list[int] | None = None
) -> bool:
    """
    Checks whether some flavor of flash attention is requested or not. Optionally, checks for specific versions of
    flash attention.

    This is checked against one of the two arguments, i.e. either the `config` or the directly passed value
    `requested_attention_implementation`. Otherwise, an error will be raised (ambiguity).

    The different versions of flash attention are usually
    - Implementations based on the original flash attention repo: https://github.com/Dao-AILab/flash-attention
    - Kernels implementations such as: https://huggingface.co/kernels-community/vllm-flash-attn3
    """
    if config is not None and requested_attention_implementation is not None:
        raise ValueError(
            "Requested attention implementation is ambiguous: "
            "Please pass either the config or the name of the attention implementation, not both."
        )

    if config is not None:
        checked_attention_implementation = config._attn_implementation
    else:
        checked_attention_implementation = requested_attention_implementation

    if checked_attention_implementation is None:
        return False

    if version is not None:
        if isinstance(version, int):
            version = [version]
        return any(re.match(r".*flash.*" + str(v), checked_attention_implementation) is not None for v in version)

    return "flash" in checked_attention_implementation


def get_max_seqlen(
    cu_seqlens: torch.Tensor,
    config: PreTrainedConfig,
    kwargs: dict | None = None,
    kwarg_name: str = "max_seqlen",
) -> int | None:
    """Get the maximum packed sequence length, or pop it from `kwargs` if precomputed.

    Args:
        cu_seqlens: `(num_sequences + 1,)` cumulative sequence boundaries.
        config: model configuration used to determine the attention implementation.
        kwargs: optional caller kwargs containing a precomputed maximum sequence length.
        kwarg_name: key used to pop the precomputed value from `kwargs`.

    Returns:
        Maximum packed sequence length as a Python integer, or `None` when Flash Attention is not requested
        and no precomputed value is provided.
    """
    if kwargs is not None and (max_seqlen := kwargs.pop(kwarg_name, None)) is not None:
        return max_seqlen
    if not is_flash_attention_requested(config):
        return None
    return (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()


def split_attention_implementation(implementation: str | None) -> tuple[bool, str | None]:
    """
    Split the optional `paged|` prefix from an attention implementation string.

    Note that `None` means using the default attention implementation, which is either torch's native `sdpa` or `eager` (if `sdpa` is not implemented for that model).
    """
    if implementation is None:
        return False, None

    is_paged = implementation.startswith("paged|")
    return is_paged, implementation.removeprefix("paged|")


def to_py_obj(obj):
    """
    Convert a PyTorch tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, (int, float)):
        return obj
    elif isinstance(obj, (dict, UserDict)):
        return {k: to_py_obj(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        if all(isinstance(x, (int, float, np.number)) for x in obj):
            return list(obj)

        return [to_py_obj(o) for o in obj]

    framework_to_py_obj = {
        "pt": lambda obj: obj.tolist(),
        "np": lambda obj: obj.tolist(),
    }

    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_py_obj[framework](obj)

    if isinstance(obj, np.number):
        return obj.tolist()
    else:
        return obj


def to_numpy(obj):
    """
    Convert a PyTorch tensor, Numpy array or python list to a Numpy array.
    """

    framework_to_numpy = {
        "pt": lambda obj: obj.detach().cpu().numpy(),
        "np": lambda obj: obj,
    }

    if isinstance(obj, (dict, UserDict)):
        return {k: to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return np.array(obj)

    framework_to_test_func = _get_frameworks_and_test_func(obj)
    for framework, test_func in framework_to_test_func.items():
        if test_func(obj):
            return framework_to_numpy[framework](obj)

    return obj


def safe_load_json_file(json_file: str):
    "A helper to load safe config files and raise a proper error message if it wasn't serialized correctly"
    try:
        with open(json_file, encoding="utf-8") as reader:
            text = reader.read()
        config_dict = json.loads(text)
    except json.JSONDecodeError:
        raise OSError(f"It looks like the config file at '{json_file}' is not a valid JSON file.")
    return config_dict


class ModelOutput(OrderedDict):

    def __init_subclass__(cls) -> None:
        """Register subclasses as pytree nodes.

        This is necessary to synchronize gradients when using `torch.nn.parallel.DistributedDataParallel` with
        `static_graph=True` with modules that output `ModelOutput` subclasses.
        """
        _register_model_output_pytree_node(cls)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _register_model_output_pytree_node(type(self))

        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclass."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )

    def __post_init__(self):
        """Check the ModelOutput dataclass.

        Only occurs if @dataclass decorator has been used.
        """
        _register_model_output_pytree_node(type(self))
        class_fields = fields(self)

        if not len(class_fields):
            raise ValueError(f"{self.__class__.__name__} has no fields.")
        if not all(field.default is None for field in class_fields[1:]):
            raise ValueError(f"{self.__class__.__name__} should not have more than one required field.")

        first_field = getattr(self, class_fields[0].name)
        other_fields_are_none = all(self.__dict__.get(field.name) is None for field in class_fields[1:])

        if other_fields_are_none and not is_tensor(first_field):
            if isinstance(first_field, dict):
                iterator = first_field.items()
                first_field_iterator = True
            else:
                try:
                    iterator = iter(first_field)
                    first_field_iterator = True
                except TypeError:
                    first_field_iterator = False

            if first_field_iterator:
                setattr(self, class_fields[0].name, None)
                super().__delitem__(class_fields[0].name)
                for idx, element in enumerate(iterator):
                    if not isinstance(element, (list, tuple)) or len(element) != 2 or not isinstance(element[0], str):
                        if idx == 0:
                            self[class_fields[0].name] = first_field
                        else:
                            raise ValueError(
                                f"Cannot set key/value for {element}. It needs to be a tuple (key, value)."
                            )
                        break
                    setattr(self, element[0], element[1])
                    if element[1] is not None:
                        self[element[0]] = element[1]
            elif first_field is not None:
                self[class_fields[0].name] = first_field
        else:
            for field in class_fields:
                v = self.__dict__.get(field.name)
                if v is not None:
                    self[field.name] = v

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        field_names = {field.name for field in fields(self)}
        if name in field_names and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> tuple:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())


def _model_output_flatten(output: ModelOutput) -> tuple[list[Any], list[str]]:
    pass


def _model_output_unflatten(
    values: Iterable[Any],
    context: list[str],
    output_type: type[ModelOutput] | None = None,
) -> ModelOutput:
    pass


class ExplicitEnum(str, Enum):

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


class TensorType(ExplicitEnum):

    PYTORCH = "pt"
    NUMPY = "np"
    MLX = "mlx"


class ContextManagers:

    def __init__(self, context_managers: list[AbstractContextManager]):
        self.context_managers = context_managers
        self.stack = ExitStack()

    def __enter__(self):
        for context_manager in self.context_managers:
            self.stack.enter_context(context_manager)

    def __exit__(self, *args, **kwargs):
        self.stack.__exit__(*args, **kwargs)


def can_return_loss(model_class):
    """
    Check if a given model can return loss.

    Args:
        model_class (`type`): The class of the model.
    """
    signature = inspect.signature(model_class.forward)

    for p in signature.parameters:
        if p == "return_loss" and signature.parameters[p].default is True:
            return True

    return False


def find_labels(model_class):
    """
    Find the labels used by a given model.

    Args:
        model_class (`type`): The class of the model.
    """
    model_name = model_class.__name__
    signature = inspect.signature(model_class.forward)

    if "QuestionAnswering" in model_name:
        return [p for p in signature.parameters if "label" in p or p in ("start_positions", "end_positions")]
    else:
        return [p for p in signature.parameters if "label" in p]


def flatten_dict(d: MutableMapping, parent_key: str = "", delimiter: str = "."):
    """Flatten a nested dict into a single level dict."""

    def _flatten_dict(d, parent_key="", delimiter="."):
        for k, v in d.items():
            key = str(parent_key) + delimiter + str(k) if parent_key else k
            if v and isinstance(v, MutableMapping):
                yield from flatten_dict(v, key, delimiter=delimiter).items()
            else:
                yield key, v

    return dict(_flatten_dict(d, parent_key, delimiter))


def transpose(array, axes=None):
    """
    Framework-agnostic version of transpose operation.
    """
    if is_numpy_array(array):
        return np.transpose(array, axes=axes)
    elif is_torch_tensor(array):
        return array.T if axes is None else array.permute(*axes)
    else:
        raise ValueError(f"Type not supported for transpose: {type(array)}.")


def reshape(array, newshape):
    """
    Framework-agnostic version of reshape operation.
    """
    if is_numpy_array(array):
        return np.reshape(array, newshape)
    elif is_torch_tensor(array):
        return array.reshape(*newshape)
    else:
        raise ValueError(f"Type not supported for reshape: {type(array)}.")


def squeeze(array, axis=None):
    """
    Framework-agnostic version of squeeze operation.
    """
    if is_numpy_array(array):
        return np.squeeze(array, axis=axis)
    elif is_torch_tensor(array):
        return array.squeeze() if axis is None else array.squeeze(dim=axis)
    else:
        raise ValueError(f"Type not supported for squeeze: {type(array)}.")


def expand_dims(array, axis):
    """
    Framework-agnostic version of expand_dims operation.
    """
    if is_numpy_array(array):
        return np.expand_dims(array, axis)
    elif is_torch_tensor(array):
        return array.unsqueeze(dim=axis)
    else:
        raise ValueError(f"Type not supported for expand_dims: {type(array)}.")


def tensor_size(array):
    pass


def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    if not _is_torch_available:
        return int(x)

    import torch

    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


def torch_float(x):
    """
    Casts an input to a torch float32 tensor if we are in a tracing context, otherwise to a Python float.
    """
    if not _is_torch_available:
        return int(x)

    import torch

    return x.to(torch.float32) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


def filter_out_non_signature_kwargs(extra: list | None = None):
    """
    Decorator to filter out named arguments that are not in the function signature.

    This decorator ensures that only the keyword arguments that match the function's signature, or are specified in the
    `extra` list, are passed to the function. Any additional keyword arguments are filtered out and a warning is issued.

    Parameters:
        extra (`Optional[list]`, *optional*):
            A list of extra keyword argument names that are allowed even if they are not in the function's signature.

    Returns:
        Callable:
            A decorator that wraps the function and filters out invalid keyword arguments.

    Example usage:

        ```python
        @filter_out_non_signature_kwargs(extra=["allowed_extra_arg"])
        def my_function(arg1, arg2, **kwargs):
            print(arg1, arg2, kwargs)

        my_function(arg1=1, arg2=2, allowed_extra_arg=3, invalid_arg=4)
        # This will print: 1 2 {"allowed_extra_arg": 3}
        # And issue a warning: "The following named arguments are not valid for `my_function` and were ignored: 'invalid_arg'"
        ```
    """
    extra = extra or []
    extra_params_to_pass = set(extra)

    def decorator(func):
        pass

    return decorator


class TransformersKwargs(TypedDict, total=False):

    num_items_in_batch: torch.Tensor | None
    output_hidden_states: bool | None
    output_attentions: bool | None
    output_router_logits: bool | None
    cu_seq_lens_q: torch.LongTensor | None
    cu_seq_lens_k: torch.LongTensor | None
    max_length_q: int | None
    max_length_k: int | None
    position_ids: torch.LongTensor | None
    is_causal: bool | None
    seq_idx: torch.IntTensor | None


def is_timm_config_dict(config_dict: dict[str, Any]) -> bool:
    """Checks whether a config dict is a timm config dict."""
    return "pretrained_cfg" in config_dict


def is_timm_local_checkpoint(pretrained_model_path: str) -> bool:
    """
    Checks whether a checkpoint is a timm model checkpoint.
    """
    if pretrained_model_path is None:
        return False

    pretrained_model_path = str(pretrained_model_path)

    is_file = os.path.isfile(pretrained_model_path)
    is_dir = os.path.isdir(pretrained_model_path)

    if is_file and pretrained_model_path.endswith(".json"):
        with open(pretrained_model_path) as f:
            config_dict = json.load(f)
        return is_timm_config_dict(config_dict)

    if is_dir and os.path.exists(os.path.join(pretrained_model_path, "config.json")):
        with open(os.path.join(pretrained_model_path, "config.json")) as f:
            config_dict = json.load(f)
        return is_timm_config_dict(config_dict)

    return False


def set_attribute_for_modules(module: nn.Module, key: str, value: Any):
    pass


def del_attribute_from_modules(module: nn.Module, key: str):
    pass


def can_return_tuple(func):
    """
    Decorator to wrap model method, to call output.to_tuple() if return_dict=False passed as a kwarg or
    return_dict=False is set in the config.

    Note:
        output.to_tuple() convert output to tuple skipping all `None` values.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        return_dict = self.config.return_dict if hasattr(self, "config") else True
        return_dict_passed = kwargs.pop("return_dict", return_dict)
        if return_dict_passed is not None:
            return_dict = return_dict_passed
        output = func(self, *args, **kwargs)
        if not return_dict and not isinstance(output, tuple):
            output = output.to_tuple()
        return output

    return wrapper


_KNOWN_MODALITIES = ("image", "video", "audio")


def accepts_precomputed_kwargs(modality: str):
    """
    Decorator for `get_<modality>_features` methods that:
      - strips the modality prefix from incoming kwargs whose stripped name isn't an existing
        parameter (e.g. `image_cu_seqlens` → `cu_seqlens`, forwarded via `**kwargs`);
      - drops kwargs prefixed with another known modality (e.g. `video_*` passed to an
        image method), so an outer `forward()` can blindly forward `**kwargs` to each
        modality method without leaking the wrong tensors into the wrong encoder;
      - leaves everything else untouched (including kwargs that match a named parameter).

    Used so multimodal models can accept arbitrary precomputed tensors (`image_cu_seqlens`,
    `video_position_ids`, …) without enumerating each one in every signature.

    NOTE: Apply this decorator **only once per modality**, on the innermost base model's
    `get_<modality>_features` (i.e. on `Model.get_image_features`, not on the outer
    `ForConditionalGeneration.get_image_features` wrapper). Stacking it at multiple layers
    causes premature prefix-stripping: the outer layer rewrites `image_foo` → `foo` based
    on its own (narrower) signature, hiding kwargs that the inner method declares as named
    parameters. Outer wrappers should just forward `**kwargs` through.

    TODO: these modality-prefixed kwargs (`image_cu_seqlens`, `video_position_ids`, …) are
    currently power-feature-only — they have no visible declaration in any public signature,
    so users have to discover them from helper functions or docs. We should find a way to
    surface them properly (e.g. in `TransformersKwargs`, in a dedicated `MultimodalKwargs`
    typed dict, or returned grouped from the processor as `BatchFeature.images_data={...}`)
    so the supported set is discoverable in one place.
    """
    prefix = f"{modality}_"
    other_prefixes = tuple(f"{m}_" for m in _KNOWN_MODALITIES if m != modality)

    def decorator(func):
        pass

    return decorator


def merge_with_config_defaults(func):
    pass




def check_model_inputs(func):
    pass


def no_inherit_decorator(obj: T) -> T:
    pass


class GeneralInterface(MutableMapping):

    _global_mapping = {}

    def __init__(self):
        self._local_mapping = {}

    def __getitem__(self, key):
        if key in self._local_mapping:
            return self._local_mapping[key]
        return self._global_mapping[key]

    def __setitem__(self, key, value):
        self._local_mapping.update({key: value})

    def __delitem__(self, key):
        del self._local_mapping[key]

    def __iter__(self):
        return iter({**self._global_mapping, **self._local_mapping})

    def __len__(self):
        return len(self._global_mapping.keys() | self._local_mapping.keys())

    @classmethod
    def register(cls, key: str, value: Callable):
        cls._global_mapping.update({key: value})

    def valid_keys(self) -> list[str]:
        return list(self.keys())


def retry(
    max_retries=5,
    initial_delay=1.0,
    max_delay=30.0,
    jitter=True,
    exceptions=(Exception,),
):
    """
    Decorator that retries a function call with exponential backoff.

    Args:
        max_retries (`int`, *optional*, defaults to 5):
            Maximum number of retry attempts.
        initial_delay (`float`, *optional*, defaults to 1.0):
            Initial delay in seconds before the first retry.
        max_delay (`float`, *optional*, defaults to 30.0):
            Maximum delay in seconds between retries.
        jitter (`bool`, *optional*, defaults to `True`):
            Whether to add random jitter to the delay.
        exceptions (`tuple`, *optional*, defaults to `(Exception,)`):
            Tuple of exception types to catch and retry on.
    """

    def decorator(func):
        pass

    return decorator


def maybe_replace_from_package(source_package: str, func_name: str):
    """
    This decorator will try to replace the decorated function with `func_name` imported from `package`, if it's available. If not,
    simply use the decorated function.
    Useful to define explicit torch fallback functions, while still using an optimized implementations from auxiliary package (e.g.
    `causal_conv1d`) if available.
    """

    def decorator(torch_func: Callable) -> Callable:
        pass

    return decorator
