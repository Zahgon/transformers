
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from packaging import version

from ..utils import logging
from ..utils.import_utils import _is_package_available, is_torch_available
from .configs import ExportConfigMixin
from .utils import decompose_for_generation


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    if is_torch_available():
        import torch

        from ..cache_utils import Cache
        from ..modeling_utils import PreTrainedModel


class HfExporter(ABC):

    required_packages: list[str] = []
    min_versions: dict[str, str] = {}
    tested_versions: dict[str, str] = {}

    def __init__(self):
        self.validate_environment()

    def validate_environment(self, *args, **kwargs):
        """Check `required_packages` are installed and warn on version drift from `tested_versions`."""
        missing, drift = [], []
        for pkg in self.required_packages:
            exists, installed = _is_package_available(pkg, return_version=True)
            if not exists:
                missing.append(pkg)
                continue
            tested = self.tested_versions.get(pkg)
            if tested is not None and installed != "N/A":
                installed_base = installed.split("+", 1)[0]
                tested_base = tested.split("+", 1)[0]
                if installed_base != tested_base:
                    drift.append((pkg, installed_base, tested_base))

        if missing:
            specs = ", ".join(
                f"{pkg}=={self.tested_versions[pkg]}" if pkg in self.tested_versions else pkg for pkg in missing
            )
            raise ImportError(f"To use {type(self).__name__}, please install the following dependencies: {specs}")

        outdated = []
        for pkg, minimum in self.min_versions.items():
            _, installed = _is_package_available(pkg, return_version=True)
            if installed == "N/A" or version.parse(installed.split("+", 1)[0]) < version.parse(minimum):
                outdated.append(f"{pkg}>={minimum} (found {installed})")
        if outdated:
            raise ImportError(f"{type(self).__name__} requires newer versions of: {', '.join(outdated)}")

        if drift:
            details = ", ".join(f"{pkg}: installed {got}, tested {want}" for pkg, got, want in drift)
            logger.warning(
                f"{type(self).__name__} is experimental and patches many backend internals; "
                f"behaviour may differ from what was validated. Version drift detected — {details}. "
                f"If you hit issues, try the tested versions."
            )

    @abstractmethod
    def export(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin,
    ):
        """
        Export the model and return the backend-specific program object.

        Args:
            model ([`PreTrainedModel`]):
                The model to export.
            sample_inputs (`dict[str, torch.Tensor | Cache]`):
                **Forward** kwargs — what you'd pass to `model(**sample_inputs)`. These are used
                directly as the example inputs during tracing. For an autoregressive decode-step
                export, this means you need to include `past_key_values`, `cache_position`, etc.
                If you only have generation-style inputs, use [`~HfExporter.export_for_generation`]
                instead — it runs `model.generate` for you and exports each stage.
            config ([`~transformers.exporters.configs.ExportConfigMixin`]):
                Backend-specific configuration.

        Returns:
            Backend-specific export artifact.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement `export`. Pick a concrete exporter "
            "(`DynamoExporter`, `OnnxExporter`, `ExecutorchExporter`), or override `export` "
            "in your subclass with a backend-specific tracing pipeline that consumes `config` "
            "and returns the runtime artifact."
        )

    def export_for_generation(
        self,
        model: PreTrainedModel,
        sample_inputs: MutableMapping[str, torch.Tensor | Cache],
        config: ExportConfigMixin | dict[str, ExportConfigMixin],
    ) -> dict[str, object]:
        pass
