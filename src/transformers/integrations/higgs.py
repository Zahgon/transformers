
from math import sqrt

from ..quantizers.quantizers_utils import should_convert_module
from ..utils import is_flute_available, is_hadamard_available, is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn

if is_flute_available():
    from flute.integrations.higgs import prepare_data_transposed
    from flute.tune import TuneMetaData, qgemm_v2

if is_hadamard_available():
    from fast_hadamard_transform import hadamard_transform

logger = logging.get_logger(__name__)


def pad_to_block(tensor, dims, had_block_size, value=0):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_1024 = ((size - 1) // had_block_size + 1) * had_block_size
        delta = next_multiple_of_1024 - size
        pad_dims[-2 * dim - 1] = delta

    return nn.functional.pad(tensor, pad_dims, "constant", value)


def get_higgs_grid(p: int, n: int) -> "torch.Tensor":
    pass


def quantize_with_higgs(weight, bits: int = 4, p: int = 2, group_size: int = 256, hadamard_size: int = 1024):
    pass


class HiggsLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_bits: int,
        bias=True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        group_size: int = 256,
        hadamard_size: int = 1024,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.group_size = group_size
        self.hadamard_size = hadamard_size

        assert in_features % group_size == 0
        assert num_bits in [2, 3, 4]

        self.weight = nn.Parameter(
            torch.empty((out_features * num_bits // 16, in_features), dtype=torch.int16, device=device),
            requires_grad=False,
        )
        self.scales = nn.Parameter(
            torch.empty((out_features, in_features // group_size), dtype=dtype, device=device), requires_grad=False
        )
        self.tables = nn.Parameter(torch.empty((2**num_bits,), dtype=dtype, device=device), requires_grad=False)
        self.tables2 = nn.Parameter(
            torch.empty((2**num_bits, 2**num_bits, 2), dtype=dtype, device=device), requires_grad=False
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.workspace = None  # must be set externally to be reused among layers
        self.tune_metadata: TuneMetaData = None  # must be set externally because architecture dependent

    def forward(self, x):
        x = pad_to_block(x, [-1], self.hadamard_size)

        if self.workspace is None:
            raise Exception("Workspace must be set before calling forward")

        return qgemm_v2(
            x,
            self.weight,
            self.scales,
            self.tables,
            self.tables2.view(dtype=torch.float32),
            self.workspace,
            self.tune_metadata,
            hadamard_size=self.hadamard_size,
        )


def replace_with_higgs_linear(model, modules_to_not_convert: list[str] | None = None, quantization_config=None):
    """
    Public method that replaces the Linear layers of the given model with HIGGS quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        modules_to_not_convert (`list[str]`, *optional*, defaults to `None`):
            A list of nn.Linear weights to not convert. If a parameter path is in the list (e.g. `lm_head.weight`), the corresponding module will not be
            converted.
        quantization_config (`HiggsConfig`):
            The quantization config object that contains the quantization parameters.
    """

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        with torch.device("meta"):
            if isinstance(module, nn.Linear):
                new_module = HiggsLinear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    num_bits=quantization_config.bits,
                    hadamard_size=quantization_config.hadamard_size,
                    group_size=quantization_config.group_size,
                )
                new_module.source_cls = type(module)
                new_module.requires_grad_(False)
                model.set_submodule(module_name, new_module)
                has_been_replaced = True

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using eetq but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )
    return model


def dequantize_higgs(model, current_key_name=None):
    """
    Dequantizes the HiggsLinear layers in the given model by replacing them with standard torch.nn.Linear layers.
    Args:
        model (torch.nn.Module): The model containing HiggsLinear layers to be dequantized.
        current_key_name (list, optional): A list to keep track of the current module names during recursion. Defaults to None.
    Returns:
        torch.nn.Module: The model with HiggsLinear layers replaced by torch.nn.Linear layers.
    """

    with torch.no_grad():
        for name, module in model.named_children():
            if current_key_name is None:
                current_key_name = []
            current_key_name.append(name)

            if isinstance(module, HiggsLinear):
                in_features = module.in_features
                out_features = module.out_features

                model._modules[name] = torch.nn.Linear(
                    in_features,
                    out_features,
                    bias=module.bias is not None,
                    device=module.scales.device,
                    dtype=module.scales.dtype,
                )

                model._modules[name].weight.data = module(
                    torch.eye(in_features, device=module.scales.device, dtype=module.scales.dtype)
                ).T.contiguous()

            if len(list(module.children())) > 0:
                _ = dequantize_higgs(
                    module,
                    current_key_name=current_key_name,
                )
            current_key_name.pop(-1)
        return model
