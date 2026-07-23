
import functools
import logging
import os
import sys
import threading
from collections.abc import Callable
from logging import (
    CRITICAL,  # NOQA
    DEBUG,
    ERROR,
    FATAL,  # NOQA
    INFO,
    NOTSET,  # NOQA
    WARN,  # NOQA
    WARNING,
)
from logging import captureWarnings as _captureWarnings
from typing import Any

import huggingface_hub.utils as hf_hub_utils
from tqdm import auto as tqdm_lib

from .._typing import TransformersLogger


_lock = threading.Lock()
_default_handler: logging.Handler | None = None

log_levels = {
    "detail": logging.DEBUG,  # will also print filename and line number
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}

_default_log_level = logging.WARNING

_tqdm_active = not hf_hub_utils.are_progress_bars_disabled()
_tqdm_hook: Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any] | None = None


def _get_default_logging_level():
    """
    If TRANSFORMERS_VERBOSITY env var is set to one of the valid choices return that as the new default level. If it is
    not - fall back to `_default_log_level`
    """
    env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option TRANSFORMERS_VERBOSITY={env_level_str}, "
                f"has to be one of: {', '.join(log_levels.keys())}"
            )
    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        lib_name = _get_library_name()
        logging_format = f"[{lib_name}] %(message)s"

        if os.getenv("TRANSFORMERS_VERBOSITY", None) == "detail":
            logging_format = "%(levelname)s [%(name)s:%(lineno)s] %(asctime)s %(message)s"

        formatter = logging.Formatter(logging_format)
        _default_handler.setFormatter(formatter)

        ci = os.getenv("CI")
        is_ci = ci is not None and ci.upper() in {"1", "ON", "YES", "TRUE"}
        library_root_logger.propagate = is_ci


def _reset_library_root_logger() -> None:
    pass


def get_log_levels_dict():
    return log_levels


def captureWarnings(capture):
    pass


def get_logger(name: str | None = None) -> TransformersLogger:
    """
    Return a logger with the specified name.

    This function is not supposed to be directly accessed unless you are writing a custom transformers module.
    """

    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """
    Return the current level for the 🤗 Transformers's root logger as an int.

    Returns:
        `int`: The logging level.

    <Tip>

    🤗 Transformers has following logging levels:

    - 50: `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
    - 40: `transformers.logging.ERROR`
    - 30: `transformers.logging.WARNING` or `transformers.logging.WARN`
    - 20: `transformers.logging.INFO`
    - 10: `transformers.logging.DEBUG`

    </Tip>"""

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """
    Set the verbosity level for the 🤗 Transformers's root logger.

    Args:
        verbosity (`int`):
            Logging level, e.g., one of:

            - `transformers.logging.CRITICAL` or `transformers.logging.FATAL`
            - `transformers.logging.ERROR`
            - `transformers.logging.WARNING` or `transformers.logging.WARN`
            - `transformers.logging.INFO`
            - `transformers.logging.DEBUG`
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def set_verbosity_info():
    """Set the verbosity to the `INFO` level."""
    return set_verbosity(INFO)


def set_verbosity_warning():
    """Set the verbosity to the `WARNING` level."""
    return set_verbosity(WARNING)


def set_verbosity_debug():
    """Set the verbosity to the `DEBUG` level."""
    return set_verbosity(DEBUG)


def set_verbosity_error():
    pass


def disable_default_handler() -> None:
    pass


def enable_default_handler() -> None:
    """Enable the default handler of the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def add_handler(handler: logging.Handler) -> None:
    """adds a handler to the HuggingFace Transformers's root logger."""

    _configure_library_root_logger()

    assert handler is not None
    _get_library_root_logger().addHandler(handler)


def remove_handler(handler: logging.Handler) -> None:
    pass


def disable_propagation() -> None:
    pass


def enable_propagation() -> None:
    pass


def enable_explicit_format() -> None:
    """
    Enable explicit formatting for every HuggingFace Transformers's logger. The explicit formatter is as follows:
    ```
        [LEVELNAME|FILENAME|LINE NUMBER] TIME >> MESSAGE
    ```
    All handlers currently bound to the root logger are affected by this method.
    """
    handlers = _get_library_root_logger().handlers

    for handler in handlers:
        formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s >> %(message)s")
        handler.setFormatter(formatter)


def reset_format() -> None:
    pass


def warning_advice(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but if env var TRANSFORMERS_NO_ADVISORY_WARNINGS=1 is set, this
    warning will not be printed
    """
    no_advisory_warnings = os.getenv("TRANSFORMERS_NO_ADVISORY_WARNINGS")
    if no_advisory_warnings:
        return
    self.warning(*args, **kwargs)


logging.Logger.warning_advice = warning_advice  # type: ignore[unresolved-attribute]


@functools.lru_cache(None)
def warning_once(self, *args, **kwargs):
    """
    This method is identical to `logger.warning()`, but will emit the warning with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.warning(*args, **kwargs)


logging.Logger.warning_once = warning_once  # type: ignore[unresolved-attribute]


@functools.lru_cache(None)
def info_once(self, *args, **kwargs):
    """
    This method is identical to `logger.info()`, but will emit the info with the same message only once

    Note: The cache is for the function arguments, so 2 different callers using the same arguments will hit the cache.
    The assumption here is that all warning messages are unique across the code. If they aren't then need to switch to
    another type of cache that includes the caller frame information in the hashing function.
    """
    self.info(*args, **kwargs)


logging.Logger.info_once = info_once  # type: ignore[unresolved-attribute]


class EmptyTqdm:

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        self._iterator = args[0] if args else None

    def __iter__(self):
        return iter(self._iterator)

    def __getattr__(self, _):
        """Return empty function."""

        def empty_fn(*args, **kwargs):  # pylint: disable=unused-argument
            pass

        return empty_fn

    def __enter__(self):
        return self

    def __exit__(self, type_, value, traceback):
        return


class _tqdm_cls:
    def __call__(self, *args, **kwargs):
        factory = tqdm_lib.tqdm if _tqdm_active else EmptyTqdm
        if _tqdm_hook is not None:
            return _tqdm_hook(factory, args, kwargs)
        return factory(*args, **kwargs)

    def set_lock(self, *args, **kwargs):
        pass

    def get_lock(self):
        pass


tqdm = _tqdm_cls()


def is_progress_bar_enabled() -> bool:
    pass


def enable_progress_bar():
    """Enable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = True
    hf_hub_utils.enable_progress_bars()


def disable_progress_bar():
    """Disable tqdm progress bar."""
    global _tqdm_active
    _tqdm_active = False
    hf_hub_utils.disable_progress_bars()


def set_tqdm_hook(hook: Callable[[Callable[..., Any], tuple[Any, ...], dict[str, Any]], Any] | None):
    """
    Set a hook that customizes tqdm creation.

    The hook is called with the tqdm factory to use (either `tqdm.auto.tqdm` or an empty shim), along with the
    positional and keyword arguments that would have been passed to tqdm. The hook should return an object compatible
    with tqdm (i.e. implementing the methods your code relies on, such as `update`, `close`, context manager methods,
    etc.).

    Passing `None` clears the hook.

    Returns:
        The previous hook, which can be restored later.
    """
    global _tqdm_hook
    previous_hook = _tqdm_hook
    _tqdm_hook = hook
    return previous_hook
