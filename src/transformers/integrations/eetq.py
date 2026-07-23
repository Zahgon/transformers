from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn


logger = logging.get_logger(__name__)


class EetqQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self, input_dict: dict[str, list[torch.Tensor]], full_layer_name: str | None = None, **kwargs
    ) -> dict[str, torch.Tensor]:
        _, value = tuple(input_dict.items())[0]
        value = value[0]

        value_device = value.device
        int8_weight = torch.t(value).contiguous().cpu()
        int8_weight, scales = eetq_kernels_hub.quant_weights(int8_weight, torch.int8, False)

        int8_weight = int8_weight.to(value_device)
        scales = scales.to(value_device)

        return {full_layer_name: int8_weight, f"{full_layer_name}_scales": scales}


class EetqLinearMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scales, bias=None):
        ctx.save_for_backward(x, weight, scales, bias)
        output = eetq_kernels_hub.w8_a16_gemm(x, weight, scales)
        output = output + bias if bias is not None else output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        pass


class EetqLinear(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.int8, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_features, out_features), dtype=dtype), requires_grad=False)
        self.weight_scales = nn.Parameter(torch.empty((out_features), dtype=torch.float16))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features), dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, input):
        output = EetqLinearMMFunction.apply(input, self.weight, self.weight_scales, self.bias)
        return output


def replace_with_eetq_linear(model, modules_to_not_convert: list[str] | None = None, pre_quantized=False):
    """
    A helper function to replace all `torch.nn.Linear` modules by `EetqLinear` modules.

    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        modules_to_not_convert (`list[`str`]`, *optional*, defaults to `None`):
            Names of the modules to not convert in `EetqLinear`. In practice we keep the `lm_head` in full precision
            for numerical stability reasons.
    """
    from .hub_kernels import get_kernel

    global eetq_kernels_hub
    eetq_kernels_hub = get_kernel("kernels-community/quantization-eetq", version=1)

    has_been_replaced = False
    module_kwargs = {} if pre_quantized else {"dtype": None}
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with torch.device("meta"):
            if isinstance(module, nn.Linear):
                new_module = EetqLinear(
                    module.in_features, module.out_features, bias=module.bias is not None, **module_kwargs
                )
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
