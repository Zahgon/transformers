
from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import QuarkConfig

from ..utils import is_quark_available, logging


logger = logging.get_logger(__name__)


CHECKPOINT_KEYS = {
    "weight_scale": "weight_quantizer.scale",
    "bias_scale": "bias_quantizer.scale",
    "input_scale": "input_quantizer.scale",
    "output_scale": "output_quantizer.scale",
    "weight_zero_point": "weight_quantizer.zero_point",
    "bias_zero_point": "bias_quantizer.zero_point",
    "input_zero_point": "input_quantizer.zero_point",
    "output_zero_point": "output_quantizer.zero_point",
}


class QuarkHfQuantizer(HfQuantizer):

    requires_calibration = True  # On-the-fly quantization with quark is not supported for now.
    quantization_config: "QuarkConfig"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

        self.json_export_config = quantization_config.json_export_config

    def validate_environment(self, *args, **kwargs):
        if not is_quark_available():
            raise ImportError(
                "Loading a Quark quantized model requires the `quark` library but it was not found in the environment. Please refer to https://quark.docs.amd.com/latest/install.html."
            )

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from quark.torch.export.api import _map_to_quark

        _map_to_quark(
            model,
            self.quantization_config.quant_config,
            pack_method=self.json_export_config.pack_method,
            custom_mode=self.quantization_config.custom_mode,
        )

        return model

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        return True

    def is_serializable(self):
        return False

    @property
    def is_trainable(self):
        pass

    def get_weight_conversions(self):
        from ..core_model_loading import WeightConverter
        from ..integrations.quark import QuarkDeserialize

        converters = []
        for source_key, target_key in CHECKPOINT_KEYS.items():
            converters.append(
                WeightConverter(
                    source_patterns=[source_key],
                    target_patterns=target_key,
                    operations=[QuarkDeserialize(self)],
                )
            )
        return converters
