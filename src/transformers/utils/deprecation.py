import inspect
import warnings
from functools import wraps

import packaging.version

from .. import __version__
from . import ExplicitEnum, is_torch_available, is_torchdynamo_compiling


if is_torch_available():
    import torch  # noqa: F401


class Action(ExplicitEnum):
    NONE = "none"
    NOTIFY = "notify"
    NOTIFY_ALWAYS = "notify_always"
    RAISE = "raise"


def deprecate_kwarg(
    old_name: str,
    version: str,
    new_name: str | None = None,
    warn_if_greater_or_equal_version: bool = False,
    raise_if_greater_or_equal_version: bool = False,
    raise_if_both_names: bool = False,
    additional_message: str | None = None,
):
    """
    Function or method decorator to notify users about deprecated keyword arguments, replacing them with a new name if specified.
    Note that is decorator is `torch.compile`-safe, i.e. it will not cause graph breaks (but no warning will be displayed if compiling).

    This decorator allows you to:
    - Notify users when a keyword argument is deprecated.
    - Automatically replace deprecated keyword arguments with new ones.
    - Raise an error if deprecated arguments are used, depending on the specified conditions.

    By default, the decorator notifies the user about the deprecated argument while the `transformers.__version__` < specified `version`
    in the decorator. To keep notifications with any version `warn_if_greater_or_equal_version=True` can be set.

    Parameters:
        old_name (`str`):
            Name of the deprecated keyword argument.
        version (`str`):
            The version in which the keyword argument was (or will be) deprecated.
        new_name (`Optional[str]`, *optional*):
            The new name for the deprecated keyword argument. If specified, the deprecated keyword argument will be replaced with this new name.
        warn_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to show warning if current `transformers` version is greater or equal to the deprecated version.
        raise_if_greater_or_equal_version (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if current `transformers` version is greater or equal to the deprecated version.
        raise_if_both_names (`bool`, *optional*, defaults to `False`):
            Whether to raise `ValueError` if both deprecated and new keyword arguments are set.
        additional_message (`Optional[str]`, *optional*):
            An additional message to append to the default deprecation message.

    Raises:
        ValueError:
            If raise_if_greater_or_equal_version is True and the current version is greater than or equal to the deprecated version, or if raise_if_both_names is True and both old and new keyword arguments are provided.

    Returns:
        Callable:
            A wrapped function that handles the deprecated keyword arguments according to the specified parameters.

    Example usage with renaming argument:

        ```python
        @deprecate_kwarg("reduce_labels", new_name="do_reduce_labels", version="6.0.0")
        def my_function(do_reduce_labels):
            print(do_reduce_labels)

        my_function(reduce_labels=True)  # Will show a deprecation warning and use do_reduce_labels=True
        ```

    Example usage without renaming argument:

        ```python
        @deprecate_kwarg("max_size", version="6.0.0")
        def my_function(max_size):
            print(max_size)

        my_function(max_size=1333)  # Will show a deprecation warning
        ```

    """

    deprecated_version = packaging.version.parse(version)
    current_version = packaging.version.parse(__version__)
    is_greater_or_equal_version = current_version >= deprecated_version

    if is_greater_or_equal_version:
        version_message = f"and removed starting from version {version}"
    else:
        version_message = f"and will be removed in version {version}"

    def wrapper(func):
        sig = inspect.signature(func)
        function_named_args = set(sig.parameters.keys())
        is_instance_method = "self" in function_named_args
        is_class_method = "cls" in function_named_args

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            pass

        return wrapped_func

    return wrapper
