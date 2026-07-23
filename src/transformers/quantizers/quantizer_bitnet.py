from typing import TYPE_CHECKING

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import BitNetQuantConfig

from ..utils import is_accelerate_available, is_torch_available, logging


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class BitNetHfQuantizer(HfQuantizer):

    requires_calibration = True
    quantization_config: "BitNetQuantConfig"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Loading a BitNet quantized model requires accelerate (`pip install accelerate`)")

        if not torch.cuda.is_available():
            logger.warning_once(
                "You don't have a GPU available to load the model, the inference will be slow because of weight unpacking"
            )
            return

        device_map = kwargs.get("device_map")
        if device_map is None:
            logger.warning_once(
                "You have loaded a BitNet model on CPU and have a CUDA device available, make sure to set "
                "your model on a GPU device in order to run your model."
            )
        elif isinstance(device_map, dict):
            if len(device_map) > 1 and "cpu" in device_map.values() or "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to load a BitNet model with a device_map that contains a CPU or disk device."
                    "This is not supported. Please remove the CPU or disk device from the device_map."
                )

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        from ..integrations import replace_with_bitnet_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules
        )

        model = replace_with_bitnet_linear(
            model,
            modules_to_not_convert=self.modules_to_not_convert,
            quantization_config=self.quantization_config,
        )

    def adjust_max_memory(self, max_memory: dict[str, int | str]) -> dict[str, int | str]:
        max_memory = {key: val * 0.90 for key, val in max_memory.items()}
        return max_memory

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        pass

    @property
    def is_qat_trainable(self) -> bool:
        pass

    def get_weight_conversions(self):
        from ..core_model_loading import WeightConverter
        from ..integrations.bitnet import BitNetDeserialize

        if (
            self.quantization_config.linear_class == "autobitlinear"
            and self.quantization_config.quantization_mode == "offline"
        ):
            return [
                WeightConverter(
                    source_patterns=["weight"],
                    target_patterns=["weight"],
                    operations=[BitNetDeserialize(self)],
                )
            ]
        return []
