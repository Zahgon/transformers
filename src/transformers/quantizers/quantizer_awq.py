import importlib.metadata
from typing import TYPE_CHECKING

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import AwqConfig

from ..utils import is_accelerate_available, is_gptqmodel_available, is_torch_available, logging
from ..utils.quantization_config import AwqBackend


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class AwqQuantizer(HfQuantizer):

    requires_calibration = True
    quantization_config: "AwqConfig"

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, **kwargs):
        if not is_gptqmodel_available():
            raise ImportError(
                "Loading an AWQ quantized model requires gptqmodel. Please install it with `pip install gptqmodel`"
            )

        if not is_accelerate_available():
            raise ImportError("Loading an AWQ quantized model requires accelerate (`pip install accelerate`)")

    def update_dtype(self, dtype):
        if dtype == torch.bfloat16 and (torch.cuda.is_available() or torch.xpu.is_available()):
            logger.warning(
                "`torch.bfloat16` is not supported for AWQ CUDA/XPU kernels yet. Casting to `torch.float16`."
            )
            dtype = torch.float16
        elif dtype != torch.float16 and (torch.cuda.is_available() or torch.xpu.is_available()):
            logger.warning("We suggest you to set `dtype=torch.float16` for better efficiency on CUDA/XPU with AWQ.")
        return dtype

    def _process_model_before_weight_loading(self, model: "PreTrainedModel", **kwargs):
        from ..integrations import replace_quantization_scales, replace_with_awq_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, model._keep_in_fp32_modules, add_default_skips=True
        )

        model = replace_with_awq_linear(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
            device_map=kwargs.get("device_map"),
        )

        model = replace_quantization_scales(model, model.config.model_type)

    def _process_model_after_weight_loading(self, model, **kwargs):
        from gptqmodel.utils.model import hf_gptqmodel_post_init

        hf_gptqmodel_post_init(model, use_act_order=self.quantization_config.desc_act)

    def is_serializable(self):
        if self.quantization_config.backend in [AwqBackend.EXLLAMA_V1, AwqBackend.EXLLAMA_V2]:
            logger.warning("You cannot save an AWQ model that uses Exllama backend!")
            return False

        return True

    @property
    def is_trainable(self):
        pass
