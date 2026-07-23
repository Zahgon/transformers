from importlib import metadata
from typing import TYPE_CHECKING

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel
    from ..utils.quantization_config import AqlmConfig

from ..integrations import replace_with_aqlm_linear
from ..utils import is_accelerate_available, is_aqlm_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


logger = logging.get_logger(__name__)


class AqlmHfQuantizer(HfQuantizer):

    requires_calibration = True
    quantization_config: "AqlmConfig"

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        if not is_accelerate_available():
            raise ImportError("Using `aqlm` quantization requires Accelerate: `pip install accelerate`")

        if not is_aqlm_available():
            raise ImportError("Using `aqlm` quantization requires AQLM: `pip install aqlm[gpu,cpu]`")

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        replace_with_aqlm_linear(
            model,
            modules_to_not_convert=self.quantization_config.linear_weights_not_to_quantize,
            quantization_config=self.quantization_config,
        )

    @property
    def is_trainable(self) -> bool:
        pass

    def is_serializable(self):
        return True
