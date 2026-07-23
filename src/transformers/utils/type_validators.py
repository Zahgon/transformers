from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Union, cast

from huggingface_hub.dataclasses import as_validated_field

from ..tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ..video_utils import VideoMetadataType
from .generic import TensorType
from .import_utils import is_torch_available, is_vision_available


if is_vision_available():
    from ..image_utils import PILImageResampling

if is_torch_available():
    import torch

    from ..activations import ACT2FN
else:
    ACT2FN = {}


def positive_any_number(value: int | float | None = None):
    pass


def positive_int(value: int | None = None):
    pass


def padding_validator(value: bool | str | PaddingStrategy | None = None):
    pass


def truncation_validator(value: bool | str | TruncationStrategy | None = None):
    pass


def image_size_validator(value: int | Sequence[int] | dict[str, int] | None = None):
    pass


def device_validator(value: str | int | None = None):
    pass


def resampling_validator(value: Union[int, "PILImageResampling"] | None = None):
    pass


def video_metadata_validator(value: VideoMetadataType | None = None):
    pass


def tensor_type_validator(value: str | TensorType | None = None):
    pass


@as_validated_field
def label_to_id_validation(value: str | TensorType | None = None):
    pass


def interval(
    min: int | float | None = None,
    max: int | float | None = None,
    exclude_min: bool = False,
    exclude_max: bool = False,
) -> Callable:
    """
    Parameterized validator that ensures that `value` is within the defined interval. Optionally, the interval can be
    open on either side. Expected usage: `interval(min=0)(default=8)`

    Args:
        min (`int` or `float`, *optional*):
            Minimum value of the interval.
        max (`int` or `float`, *optional*):
            Maximum value of the interval.
        exclude_min (`bool`, *optional*, defaults to `False`):
            If True, the minimum value is excluded from the interval.
        exclude_max (`bool`, *optional*, defaults to `False`):
            If True, the maximum value is excluded from the interval.
    """
    error_message = "Value must be"
    if min is not None:
        if exclude_min:
            error_message += f" greater than {min}"
        else:
            error_message += f" greater or equal to {min}"
    if min is not None and max is not None:
        error_message += " and"
    if max is not None:
        if exclude_max:
            error_message += f" smaller than {max}"
        else:
            error_message += f" smaller or equal to {max}"
    error_message += ", got {value}."

    min = min or float("-inf")
    max = max or float("inf")

    @as_validated_field
    def _inner(value: int | float):
        pass

    return _inner


@as_validated_field
def probability(value: float):
    pass


def is_divisible_by(divisor: int | float):
    @as_validated_field
    def _inner(value: int | float):
        pass

    return _inner


@as_validated_field
def activation_fn_key(value: str):
    pass


def tensor_shape(shape: tuple[int | str], length: int | None = None):
    pass
