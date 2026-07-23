
from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from transformers.utils import logging


if TYPE_CHECKING:
    from transformers import PreTrainedConfig


logger = logging.get_logger(__name__)

_SENTINEL = object()


class AmbiguousGlobalPerLayerAttributeError(RuntimeError):
    pass


@dataclass
class _HeterogeneitySpec:
    per_layer_overrides: dict[int, dict[str, Any]]
    per_layer_attributes: set[str]
    explicit_per_layer_attributes: set[str]


def _normalize_layer_overrides(layer_overrides: dict[str, Any]) -> dict[str, Any]:
    pass


def _validate_layer_indices(config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]) -> None:
    pass


def _validate_sliding_window_and_attention_chunk_size(
    config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]
) -> None:
    pass


def _get_per_layer_attributes(per_layer_overrides: dict[int, dict[str, Any]]) -> set[str]:
    pass


def _modify_config_and_create_heterogeneity_spec(
    config: PreTrainedConfig, per_layer_overrides: dict[int, dict[str, Any]]
) -> _HeterogeneitySpec:
    pass


def _apply_heterogeneous_config(
    config: PreTrainedConfig,
    per_layer_config: dict[int | str, dict[str, Any]],
) -> None:
    pass


def _get_layer_config(
    config: PreTrainedConfig,
    layer_overrides: dict[str, Any],
) -> PreTrainedConfig:
    output_config = copy.copy(config)
    output_config.__dict__.pop("_heterogeneity_spec", None)

    output_config.skip = layer_overrides.get("skip", [])

    for attr, value in layer_overrides.items():
        if attr == "skip":
            continue
        setattr(output_config, attr, value)

    return output_config


class _PerLayerConfigView(Sequence["PreTrainedConfig"]):
    def __init__(self, config: PreTrainedConfig) -> None:
        self._config = config

    def __len__(self) -> int:
        return self._config.num_hidden_layers

    def __getitem__(self, layer_idx: int | slice) -> PreTrainedConfig | list[PreTrainedConfig]:
        if isinstance(layer_idx, slice):
            return [self[i] for i in range(*layer_idx.indices(len(self)))]

        if layer_idx < 0:
            layer_idx += len(self)
        if layer_idx < 0 or layer_idx >= len(self):
            raise IndexError("list index out of range")

        heterogeneity_spec = self._config._heterogeneity_spec
        return _get_layer_config(
            self._config,
            heterogeneity_spec.per_layer_overrides.get(layer_idx, {}),
        )


def _get_explicit_per_layer_overrides(config: PreTrainedConfig) -> dict[int, dict[str, Any]]:
    heterogeneity_spec = config._heterogeneity_spec
    explicit_per_layer_overrides = {}

    for layer_idx in range(config.num_hidden_layers):
        layer_overrides = copy.deepcopy(heterogeneity_spec.per_layer_overrides.get(layer_idx, {}))

        for attr in heterogeneity_spec.explicit_per_layer_attributes:
            if attr not in layer_overrides:
                layer_overrides[attr] = config._getattr_without_heterogeneous_validation(attr)

        if layer_overrides:
            explicit_per_layer_overrides[layer_idx] = layer_overrides

    return explicit_per_layer_overrides


class HeterogeneousConfigMixin:

    def __getattribute__(self, key: str) -> Any:
        heterogeneity_spec = super().__getattribute__("__dict__").get("_heterogeneity_spec")
        if heterogeneity_spec is not None:
            if key in heterogeneity_spec.per_layer_attributes:
                if not super().__getattribute__("allow_global_per_layer_attribute_access"):
                    raise AmbiguousGlobalPerLayerAttributeError(
                        f"'{key}' is a per-layer attribute and may vary across layers. Access it via the individual layer "
                        f"configs instead (e.g. config.per_layer_config[i].{key}). To read the global config value from "
                        f"config.{key} anyway, set `allow_global_per_layer_attribute_access` to `True` on the config. "
                        f"Warning: only do this if the caller can safely handle heterogeneous configs; code that assumes "
                        f"a homogeneous model may use the global value incorrectly."
                    )

                logger.warning_once(
                    f"Reading global config value for per-layer attribute `{key}` on a heterogeneous config. "
                    "Only do this if the caller can safely handle heterogeneous configs; code that assumes a homogeneous "
                    "model may use the global value incorrectly."
                )

        return super().__getattribute__(key)

    @property
    def is_heterogeneous(self) -> bool:
        pass

    @property
    def per_layer_config(self) -> Sequence[PreTrainedConfig] | None:
        pass

    @per_layer_config.setter
    def per_layer_config(self, per_layer_config: dict[int | str, dict[str, Any]] | None) -> None:
        pass

    @property
    def per_layer_attributes(self) -> set[str] | None:
        pass

    @property
    def allow_global_per_layer_attribute_access(self) -> bool:
        pass

    @allow_global_per_layer_attribute_access.setter
    def allow_global_per_layer_attribute_access(self, value: bool) -> None:
        pass

    @property
    def serialize_explicit_per_layer_config(self) -> bool:
        pass

    @serialize_explicit_per_layer_config.setter
    def serialize_explicit_per_layer_config(self, value: bool) -> None:
        pass

    def _iter_config_keys_with_heterogeneous_adjustment(self, keys: Iterable[str]) -> Iterable[str]:
        pass

    def _update_heterogeneous_to_dict_output(self, d: dict[str, Any]) -> None:
        if not self.is_heterogeneous:
            return

        if self.serialize_explicit_per_layer_config:
            per_layer_overrides = _get_explicit_per_layer_overrides(self)
        else:
            per_layer_overrides = self._heterogeneity_spec.per_layer_overrides

        if per_layer_overrides:
            max_digits = len(str(max(per_layer_overrides.keys())))
            d["per_layer_config"] = {
                str(layer_idx).zfill(max_digits): copy.deepcopy(layer_overrides)
                for layer_idx, layer_overrides in per_layer_overrides.items()
            }
        else:
            d["per_layer_config"] = {}

        d.pop("_heterogeneity_spec", None)

    def _getattr_without_heterogeneous_validation(self, key: str, default: Any = _SENTINEL) -> Any:
        if key != "attribute_map" and key in super().__getattribute__("attribute_map"):
            key = super().__getattribute__("attribute_map")[key]

        try:
            return super().__getattribute__(key)
        except AttributeError:
            if default is _SENTINEL:
                raise
            return default

    def _hasattr_without_heterogeneous_validation(self, key: str) -> bool:
        pass
