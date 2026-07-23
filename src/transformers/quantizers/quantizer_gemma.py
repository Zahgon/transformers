

from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..utils.quantization_config import GemmaQuantizationConfig


class GemmaQuantizer(HfQuantizer):

    quantization_config: "GemmaQuantizationConfig"
    requires_calibration = True

    def _process_model_before_weight_loading(self, model, **kwargs):
        from ..integrations.gemma_quant import replace_with_quant_layers

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )
        model = replace_with_quant_layers(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )

        ignored = set(getattr(model, "_keys_to_ignore_on_load_unexpected", None) or [])
        ignored.update([r".*\.k_cache_scale$", r".*\.v_cache_scale$"])
        model._keys_to_ignore_on_load_unexpected = ignored  # type: ignore[unresolved-attribute]

    @property
    def is_serializable(self):
        return True

    @property
    def is_trainable(self):
        pass

    @property
    def is_compileable(self) -> bool:
        pass
