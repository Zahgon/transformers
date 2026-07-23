
from ..modeling_utils import PreTrainedModel
from ..trainer_utils import unwrap_peft_model
from ..utils import is_liger_kernel_available, logging


logger = logging.get_logger(__name__)


def apply_liger_kernel(model, kernel_config):
    """
    Apply Liger Kernel optimizations to a model instance.

    Liger Kernel provides optimized Triton kernels for common transformer operations.
    This function patches the model in-place with those kernels.

    Args:
        model: The model to patch. Must be a `PreTrainedModel` or a PEFT wrapper around one.
        kernel_config: Kernel configuration.
    """
    if not is_liger_kernel_available():
        raise ImportError(
            "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
            "Please install it with `pip install liger-kernel`"
        )

    from liger_kernel.transformers import _apply_liger_kernel_to_instance

    kernel_config = kernel_config or {}
    base_model = unwrap_peft_model(model)

    if isinstance(base_model, PreTrainedModel):
        _apply_liger_kernel_to_instance(model=base_model, **kernel_config)
    else:
        logger.warning("The model is not an instance of PreTrainedModel. No liger kernels will be applied.")
