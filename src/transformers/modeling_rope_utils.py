
import math
import warnings
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Optional, TypedDict

from .utils import is_torch_available, logging


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from .configuration_utils import PreTrainedConfig


def dynamic_rope_update(rope_forward):
    pass


def _compute_linear_scaling_rope_parameters(
    config: Optional["PreTrainedConfig"] = None,
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    pass


def _compute_proportional_rope_parameters(
    config: Optional["PreTrainedConfig"] = None,
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
    head_dim_key: str = "head_dim",
) -> tuple["torch.Tensor", float]:
    pass


def _compute_dynamic_ntk_parameters(
    config: Optional["PreTrainedConfig"] = None,
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    pass


def _compute_yarn_parameters(
    config: "PreTrainedConfig",
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    pass


def _compute_longrope_parameters(
    config: "PreTrainedConfig",
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    pass


def _compute_llama3_parameters(
    config: "PreTrainedConfig",
    device: Optional["torch.device"] = None,
    seq_len: int | None = None,
    layer_type: str | None = None,
) -> tuple["torch.Tensor", float]:
    pass


ROPE_INIT_FUNCTIONS: dict[str, Callable[..., tuple["torch.Tensor", float]]] = {
    "linear": _compute_linear_scaling_rope_parameters,
    "dynamic": _compute_dynamic_ntk_parameters,
    "yarn": _compute_yarn_parameters,
    "longrope": _compute_longrope_parameters,
    "llama3": _compute_llama3_parameters,
    "proportional": _compute_proportional_rope_parameters,
}


class RopeParameters(TypedDict):

    rope_theta: float | None
    rope_type: str | None
    partial_rotary_factor: float | None
    factor: float | None
    original_max_position_embeddings: int | None
    attention_factor: float | None
    beta_fast: float | None
    beta_slow: float | None
    short_factor: list[float] | None
    long_factor: list[float] | None
    low_freq_factor: float | None
    high_freq_factor: float | None


class RotaryEmbeddingConfigMixin:

    default_theta = 10_000.0
    ignore_keys_at_rope_validation = set()

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        rope_theta = kwargs.pop("rope_theta", getattr(self, "rope_theta", self.default_theta))
        self.rope_parameters.setdefault("rope_theta", rope_theta)

        partial_rotary_factor = kwargs.get("partial_rotary_factor", getattr(self, "partial_rotary_factor", None))
        if partial_rotary_factor is not None:
            self.rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
            self.ignore_keys_at_rope_validation = set(self.ignore_keys_at_rope_validation or []) | {
                "partial_rotary_factor"
            }

        self.standardize_rope_params()
        return kwargs

    def standardize_rope_params(self):
        """
        Helper to standardize the config's rope params field by ensuring the params are defined for each
        later type. For old model the fn will duplicate a single rope param in each layer type (backward compatibility)
        """
        rope_theta = getattr(self, "rope_theta", None)
        partial_rotary_factor = getattr(self, "partial_rotary_factor", None)
        rope_parameters = getattr(self, "rope_parameters", None) or {}
        layer_types = getattr(self, "layer_types", None)

        if not (rope_parameters or rope_theta):
            logger.warning("`standardize_rope_params` was called but no RoPE parameters were found.")
            return
        elif layer_types is None or rope_parameters == {} or not set(rope_parameters.keys()).issubset(layer_types):
            rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
            rope_parameters.setdefault("rope_theta", rope_theta)
            if partial_rotary_factor is not None:
                rope_parameters["partial_rotary_factor"] = partial_rotary_factor

            if rope_parameters["rope_type"] in ["llama3", "yarn", "longrope"]:
                if hasattr(self, "original_max_position_embeddings"):
                    self.rope_parameters["original_max_position_embeddings"] = self.original_max_position_embeddings
                else:
                    self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)

        else:
            for layer_type in set(layer_types):
                rope_parameters[layer_type].setdefault("rope_type", rope_parameters[layer_type].get("type", "default"))
                rope_parameters[layer_type].setdefault("rope_theta", rope_theta)
                if partial_rotary_factor is not None:
                    rope_parameters[layer_type]["partial_rotary_factor"] = partial_rotary_factor

                if rope_parameters[layer_type]["rope_type"] in ["llama3", "yarn", "longrope"]:
                    self.rope_parameters[layer_type].setdefault(
                        "original_max_position_embeddings", self.max_position_embeddings
                    )

        self.rope_parameters = rope_parameters

    def validate_rope(self: "PreTrainedConfig"):
        """
        Validate the RoPE config arguments, given a `"PreTrainedConfig"` object
        """
        rope_parameters_dict = getattr(self, "rope_parameters", None)
        if not rope_parameters_dict:
            return

        if getattr(self, "layer_types", None) is not None and set(rope_parameters_dict.keys()).issubset(
            self.layer_types
        ):
            pass
        else:
            rope_parameters_dict = {"full_attention": rope_parameters_dict}

        for rope_parameters in rope_parameters_dict.values():
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            rope_parameters["rope_type"] = rope_type

            if validation_fn is not None:
                validation_fn(rope_parameters, ignore_keys=self.ignore_keys_at_rope_validation)
            else:
                logger.warning(
                    f"Missing validation function in 'RotaryEmbeddingConfigMixin' for 'rope_type'='{rope_type}'"
                )

    def _validate_default_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        pass

    def _validate_linear_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        pass

    def _validate_dynamic_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        pass

    def _validate_yarn_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        pass

    def _validate_longrope_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        pass

    def _validate_llama3_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        pass

    def _validate_proportional_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        pass

    @staticmethod
    def _check_received_keys(
        rope_type: str,
        received_keys: set,
        required_keys: set,
        optional_keys: set | None = None,
        ignore_keys: set | None = None,
    ):
        pass


def rope_config_validation(config: RotaryEmbeddingConfigMixin, ignore_keys: set | None = None):
    pass
