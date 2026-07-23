from importlib import metadata
from typing import TYPE_CHECKING

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..utils import is_gptqmodel_available, is_optimum_available, is_torch_available, logging
from ..utils.quantization_config import GPTQConfig, QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


MIN_GPTQ_VERSION = "1.4.3"
MIN_OPTIMUM_VERSION = "1.24.0"


class GptqHfQuantizer(HfQuantizer):

    requires_calibration = False
    quantization_config: "GPTQConfig"

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")
        from optimum.gptq import GPTQQuantizer

        self.optimum_quantizer = GPTQQuantizer.from_dict(self.quantization_config.to_dict_optimum())

    def validate_environment(self, *args, **kwargs):
        if not is_optimum_available():
            raise ImportError("Loading a GPTQ quantized model requires optimum (`pip install optimum`)")

        gptq_supports_cpu = is_gptqmodel_available()
        if not gptq_supports_cpu and not torch.cuda.is_available():
            raise RuntimeError("GPU is required to quantize or run quantize model.")
        elif not is_gptqmodel_available():
            raise ImportError("Loading a GPTQ quantized model requires gptqmodel (`pip install gptqmodel`) library.")
        elif is_gptqmodel_available() and (
            version.parse(metadata.version("gptqmodel")) < version.parse(MIN_GPTQ_VERSION)
            or version.parse(metadata.version("optimum")) < version.parse(MIN_OPTIMUM_VERSION)
        ):
            raise ImportError(
                f"The gptqmodel version should be >= {MIN_GPTQ_VERSION}, optimum version should >= {MIN_OPTIMUM_VERSION}"
            )

    def update_dtype(self, dtype: "torch.dtype") -> "torch.dtype":
        if dtype != torch.float16:
            logger.info("We suggest you to set `dtype=torch.float16` for better efficiency with GPTQ.")
        return dtype

    def update_device_map(self, device_map):
        if device_map is None:
            device_map = {"": torch.device("cpu")}
        return device_map

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if model.__class__.main_input_name != "input_ids":
            raise RuntimeError("We can only quantize pure text model.")

        if self.pre_quantized:
            if version.parse(metadata.version("optimum")) < version.parse(MIN_OPTIMUM_VERSION):
                model = self.optimum_quantizer.convert_model(model)
            else:
                model = self.optimum_quantizer.convert_model(model, **kwargs)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        if self.pre_quantized:
            model = self.optimum_quantizer.post_init_model(model)
        else:
            if self.quantization_config.tokenizer is None:
                self.quantization_config.tokenizer = model.name_or_path

            self.optimum_quantizer.quantize_model(model, self.quantization_config.tokenizer)
            model.config.quantization_config = GPTQConfig.from_dict(self.optimum_quantizer.to_dict())

    @property
    def is_trainable(self) -> bool:
        pass

    def is_serializable(self):
        return True
