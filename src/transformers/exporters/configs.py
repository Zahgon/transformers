import copy
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from typing import Any

from ..utils import logging


logger = logging.get_logger(__name__)


class ExportFormat(Enum):

    EXECUTORCH = "executorch"
    DYNAMO = "dynamo"
    ONNX = "onnx"


@dataclass
class ExportConfigMixin:

    export_format: ExportFormat

    @classmethod
    def from_dict(cls, config_dict):
        """
        Instantiates a [`ExportConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.

        Returns:
            [`ExportConfigMixin`]: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)
        return config

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        yield from self.__dict__.items()


@dataclass
class DynamoConfig(ExportConfigMixin):

    export_format: ExportFormat = ExportFormat.DYNAMO
    dynamic: bool = False

    strict: bool = False
    dynamic_shapes: dict[str, Any] | None = None
    prefer_deferred_runtime_asserts_over_guards: bool = False


@dataclass
class OnnxConfig(DynamoConfig):

    export_format: ExportFormat = ExportFormat.ONNX

    output_path: str | PathLike | None = None
    dynamic_shapes: dict[str, Any] | None = None
    opset_version: int | None = None
    external_data: bool = True
    optimize: bool = True
    export_params: bool = True
    keep_initializers_as_inputs: bool = False


@dataclass
class ExecutorchConfig(DynamoConfig):

    export_format: ExportFormat = ExportFormat.EXECUTORCH

    backend: str = "xnnpack"
