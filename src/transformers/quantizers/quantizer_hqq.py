
from typing import TYPE_CHECKING

from ..integrations import prepare_for_hqq_linear
from ..utils import is_hqq_available, is_torch_available, logging
from .base import HfQuantizer
from .quantizers_utils import get_module_from_name


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import HqqConfig


if is_torch_available():
    import torch

if is_hqq_available():
    from hqq.core.quantize import HQQLinear

    @property
    def weight(self):
        pass

    HQQLinear.weight = weight

logger = logging.get_logger(__name__)


class HqqHfQuantizer(HfQuantizer):

    requires_calibration = False
    quantization_config: "HqqConfig"

    def __init__(self, quantization_config, **kwargs):
        if not is_hqq_available():
            raise ImportError(
                "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`."
            )
        super().__init__(quantization_config, **kwargs)
        self.dtype = None
        self.using_multi_gpu = False
        self.hqq_keys = HQQLinear(None, None).state_dict_keys() - {"bias"}

    def validate_environment(self, *args, **kwargs):
        if self.dtype is None:
            if "dtype" in kwargs:
                self.dtype = kwargs["dtype"]
            else:
                self.dtype = torch.float32
                logger.info("Setting dtype to torch.float32 as the default value since it was not specified.")

        device_map = kwargs.get("device_map")
        if isinstance(device_map, dict):
            if "cpu" in device_map.values() or "disk" in device_map.values():
                raise ValueError(
                    "You are attempting to use an HQQ model with a device_map that contains a CPU or disk device."
                    " This is not supported. Please remove the CPU or disk device from the device_map."
                )
            else:
                self.using_multi_gpu = len(set(device_map.values())) > 1











    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        module, _ = get_module_from_name(model, param_name)
        return isinstance(module, torch.nn.Linear)















    def _patch_layer_for_multigpu(self, hqq_layer):
        pass

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        model = prepare_for_hqq_linear(model, quantization_config=self.quantization_config)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        setattr(model, "is_hqq_quantized", True)
        setattr(model, "is_hqq_serializable", self.is_serializable())
        return model

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        pass
