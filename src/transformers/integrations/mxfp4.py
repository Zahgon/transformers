
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    from torch import nn

from ..core_model_loading import ConversionOps, _IdentityOp
from ..quantizers.quantizers_utils import get_module_from_name, on_device, should_convert_module


logger = logging.get_logger(__name__)

FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


class Mxfp4Quantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        missing_keys: list[str] | None = None,
        full_layer_name: str | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        _, value = tuple(input_dict.items())[0]
        value = value[0] if isinstance(value, list) else value

        module, _ = get_module_from_name(model, full_layer_name)

        with torch.device(value.device):
            if isinstance(module, Mxfp4GptOssExperts):
                triton_weight_tensor, weight_scale = quantize_to_mxfp4(value.transpose(-1, -2), triton_kernels_hub)
                PrecisionConfig, FlexCtx, InFlexData = (
                    triton_kernels_hub.matmul_ogs.PrecisionConfig,
                    triton_kernels_hub.matmul_ogs.FlexCtx,
                    triton_kernels_hub.matmul_ogs.InFlexData,
                )
                triton_weight_tensor, weight_scale = swizzle_mxfp4(
                    triton_weight_tensor, weight_scale, triton_kernels_hub
                )

                proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"

                if proj in module._parameters:
                    del module._parameters[proj]

                setattr(module, proj, triton_weight_tensor)
                setattr(
                    module,
                    f"{proj}_precision_config",
                    PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
                )

                missing_keys.discard(f"{full_layer_name}")
                module._is_hf_initialized = True

                return {}


class Mxfp4Dequantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        param_data = {}
        proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"
        if f"{proj}_blocks" in input_dict.keys():
            if isinstance(input_dict[f"{proj}_blocks"], list):
                param_data[f"{proj}_blocks"] = input_dict[f"{proj}_blocks"][0]
            else:
                param_data[f"{proj}_blocks"] = input_dict[f"{proj}_blocks"]
        if f"{proj}_scales" in input_dict.keys():
            if isinstance(input_dict[f"{proj}_scales"], list):
                param_data[f"{proj}_scales"] = input_dict[f"{proj}_scales"][0]
            else:
                param_data[f"{proj}_scales"] = input_dict[f"{proj}_scales"]

        dequantized = dequantize_convertops(param_data[f"{proj}_blocks"], param_data[f"{proj}_scales"])
        return {full_layer_name: dequantized}

    @property
    def reverse_op(self) -> "ConversionOps":
        pass


class Mxfp4Deserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        param_data = {}
        proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"

        if f"{proj}_blocks" in input_dict.keys():
            if isinstance(input_dict[f"{proj}_blocks"], list):
                param_data[f"{proj}_blocks"] = input_dict[f"{proj}_blocks"][0]
            else:
                param_data[f"{proj}_blocks"] = input_dict[f"{proj}_blocks"]
        if f"{proj}_scales" in input_dict.keys():
            if isinstance(input_dict[f"{proj}_scales"], list):
                param_data[f"{proj}_scales"] = input_dict[f"{proj}_scales"][0]
            else:
                param_data[f"{proj}_scales"] = input_dict[f"{proj}_scales"]

        module, _ = get_module_from_name(model, full_layer_name)
        swizzle_mxfp4_convertops(
            param_data[f"{proj}_blocks"],
            param_data[f"{proj}_scales"],
            module,
            proj,
            param_data[f"{proj}_blocks"].device,
            triton_kernels_hub,
        )
        missing_keys.discard(f"{full_layer_name}")
        module._is_hf_initialized = True
        return {}

    @property
    def reverse_op(self) -> ConversionOps:
        pass


class Mxfp4ReverseDeserialize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module | None = None,
        full_layer_name: str | None = None,
        missing_keys: list[str] | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        num_local_experts = getattr(model.config, "num_local_experts", 32)
        hidden_size = getattr(model.config, "hidden_size", 2880)

        proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"

        name = full_layer_name.rsplit("_", 1)[0]
        module, _ = get_module_from_name(model, full_layer_name)
        state_dict = {}
        if isinstance(module, Mxfp4GptOssExperts):
            if "bias" in full_layer_name:
                name = full_layer_name.replace("_blocks", "")
                state_dict[name] = getattr(module, proj + "_bias")
                return state_dict
            if "gate_up_proj" in full_layer_name:
                state_dict[f"{name}_blocks"] = (
                    module.gate_up_proj.storage.layout.unswizzle_data(module.gate_up_proj.storage.data)
                    .transpose(-1, -2)
                    .reshape(num_local_experts, -1, 90, 16)
                )
                state_dict[f"{name}_scales"] = (
                    module.gate_up_proj_precision_config.weight_scale.storage.layout.unswizzle_data(
                        module.gate_up_proj_precision_config.weight_scale.storage.data
                    ).transpose(-1, -2)
                )
            else:
                state_dict[f"{name}_blocks"] = (
                    module.down_proj.storage.layout.unswizzle_data(module.down_proj.storage.data)
                    .transpose(-1, -2)
                    .reshape(num_local_experts, hidden_size, 90, -1)
                )
                state_dict[f"{name}_scales"] = (
                    module.down_proj_precision_config.weight_scale.storage.layout.unswizzle_data(
                        module.down_proj_precision_config.weight_scale.storage.data
                    ).transpose(-1, -2)
                )

        return state_dict


def quantize_to_mxfp4(w, triton_kernels_hub):
    downcast_to_mxfp_torch = triton_kernels_hub.numerics_details.mxfp.downcast_to_mxfp_torch
    w, w_scale = downcast_to_mxfp_torch(w.to(torch.bfloat16), torch.uint8, axis=1)
    return w, w_scale


def swizzle_mxfp4(w, w_scale, triton_kernels_hub):
    """
    Changes the layout of the tensors depending on the hardware
    """
    FP4, convert_layout, wrap_torch_tensor = (
        triton_kernels_hub.tensor.FP4,
        triton_kernels_hub.tensor.convert_layout,
        triton_kernels_hub.tensor.wrap_torch_tensor,
    )
    layout = triton_kernels_hub.tensor_details.layout
    StridedLayout = triton_kernels_hub.tensor_details.layout.StridedLayout

    value_layout, value_layout_opts = layout.make_default_matmul_mxfp4_w_layout(mx_axis=1)
    w = convert_layout(wrap_torch_tensor(w, dtype=FP4), value_layout, **value_layout_opts)
    w_scale = convert_layout(wrap_torch_tensor(w_scale), StridedLayout)
    return w, w_scale


def _convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
) -> torch.Tensor:
    """
    Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
    pass of GPT_OSS.
    """
    import math

    blocks = blocks.to(torch.uint8)
    scales = scales.to(torch.int32) - 127  # TODO that's because 128=2**7

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape[:-1]=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        sub = out[r0:r1]

        with on_device(blk.device):
            idx_lo = (blk & 0x0F).to(torch.int)
            sub[:, 0::2] = lut[idx_lo]
            del idx_lo

            idx_hi = (blk >> 4).to(torch.int)
            sub[:, 1::2] = lut[idx_hi]
            del idx_hi

            torch.ldexp(sub, exp, out=sub)
        del blk, exp, sub

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    return out.transpose(1, 2).contiguous()


def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,  # TODO these values are not here by mistake ;)
) -> torch.Tensor:
    """
    Convert the mxfp4 weights again, dequantizing and makes them compatible with the forward
    pass of GPT_OSS.
    """
    try:
        return _convert_moe_packed_tensors(blocks, scales, dtype=dtype, rows_per_chunk=rows_per_chunk)
    except torch.OutOfMemoryError:
        blocks = blocks.to("cpu")
        scales = scales.to("cpu")
        return _convert_moe_packed_tensors(blocks, scales, dtype=dtype, rows_per_chunk=rows_per_chunk)


class Mxfp4GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        self.gate_up_proj = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, 16, dtype=torch.uint8),
            requires_grad=False,
        )

        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.float32), requires_grad=False
        )

        self.down_proj = nn.Parameter(
            torch.zeros((self.num_experts, self.hidden_size, self.intermediate_size // 32, 16), dtype=torch.uint8),
            requires_grad=False,
        )

        self.down_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, dtype=torch.float32), requires_grad=False
        )
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        self.gate_up_proj_precision_config = None
        self.down_proj_precision_config = None
        self.limit = getattr(config, "swiglu_limit", 7.0)

    def forward(self, hidden_states: torch.Tensor, routing_data, gather_idx, scatter_idx) -> torch.Tensor:
        FnSpecs, FusedActivation, matmul_ogs = (
            triton_kernels_hub.matmul_ogs.FnSpecs,
            triton_kernels_hub.matmul_ogs.FusedActivation,
            triton_kernels_hub.matmul_ogs.matmul_ogs,
        )
        swiglu_fn = triton_kernels_hub.swiglu.swiglu_fn

        with on_device(hidden_states.device):
            act = FusedActivation(FnSpecs("swiglu", swiglu_fn, ("alpha", "limit")), (self.alpha, self.limit), 2)

            intermediate_cache1 = matmul_ogs(
                hidden_states,
                self.gate_up_proj,
                self.gate_up_proj_bias.to(torch.float32),
                routing_data,
                gather_indx=gather_idx,
                precision_config=self.gate_up_proj_precision_config,
                gammas=None,
                fused_activation=act,
            )

            intermediate_cache3 = matmul_ogs(
                intermediate_cache1,
                self.down_proj,
                self.down_proj_bias.to(torch.float32),
                routing_data,
                scatter_indx=scatter_idx,
                precision_config=self.down_proj_precision_config,
                gammas=routing_data.gate_scal,
            )
        return intermediate_cache3


def routing_torch_dist(
    logits,
    n_expts_act,
):
    pass


def mlp_forward(self, hidden_states):
    pass


def dequantize(module, param_name, param_value, target_device, dq_param_name, **kwargs):
    from ..integrations.tensor_parallel import shard_and_distribute_module

    model = kwargs.get("model")
    empty_param = kwargs.get("empty_param")
    casting_dtype = kwargs.get("casting_dtype")
    to_contiguous = kwargs.get("to_contiguous")
    rank = kwargs.get("rank")
    device_mesh = kwargs.get("device_mesh")

    for proj in ["gate_up_proj", "down_proj"]:
        if proj in param_name:
            if device_mesh is not None:
                param_value = shard_and_distribute_module(
                    model,
                    param_value,
                    empty_param,
                    dq_param_name,
                    casting_dtype,
                    to_contiguous,
                    rank,
                    device_mesh,
                )
            blocks_attr = f"{proj}_blocks"
            scales_attr = f"{proj}_scales"
            setattr(module, param_name.rsplit(".", 1)[1], param_value)
            if hasattr(module, blocks_attr) and hasattr(module, scales_attr):
                dequantized = convert_moe_packed_tensors(getattr(module, blocks_attr), getattr(module, scales_attr))
                setattr(module, proj, torch.nn.Parameter(dequantized.to(target_device)))
                delattr(module, blocks_attr)
                delattr(module, scales_attr)


def dequantize_convertops(blocks, scales):
    dequantized = convert_moe_packed_tensors(blocks, scales)
    return torch.nn.Parameter(dequantized)


def load_and_swizzle_mxfp4(module, param_name, param_value, target_device, triton_kernels_hub, **kwargs):
    pass


def swizzle_mxfp4_convertops(blocks, scales, module, proj, target_device, triton_kernels_hub):
    """
    This transforms the weights obtained using `convert_gpt_oss.py` to load them into `Mxfp4GptOssExperts`.
    """
    PrecisionConfig, FlexCtx, InFlexData = (
        triton_kernels_hub.matmul_ogs.PrecisionConfig,
        triton_kernels_hub.matmul_ogs.FlexCtx,
        triton_kernels_hub.matmul_ogs.InFlexData,
    )

    local_experts = blocks.size(0)
    if (
        getattr(target_device, "type", target_device) == "cpu"
        and hasattr(torch, "accelerator")
        and torch.accelerator.current_accelerator() is not None
    ):
        target_device = torch.accelerator.current_accelerator().type

    blocks = blocks.to(target_device).contiguous()
    scales = scales.to(target_device).contiguous()

    if proj == "gate_up_proj":
        blocks = blocks.reshape(local_experts, module.intermediate_size * 2, -1)
    else:
        blocks = blocks.reshape(local_experts, -1, module.intermediate_size // 2)

    with on_device(target_device):
        triton_weight_tensor, weight_scale = swizzle_mxfp4(
            blocks.transpose(-2, -1), scales.transpose(-2, -1), triton_kernels_hub
        )
    if proj == "gate_up_proj":
        triton_weight_tensor.shape = torch.Size([local_experts, module.hidden_size, module.intermediate_size * 2])
    else:
        triton_weight_tensor.shape = torch.Size([local_experts, module.intermediate_size, module.hidden_size])

    if proj in module._parameters:
        del module._parameters[proj]
    setattr(module, proj, triton_weight_tensor)
    setattr(
        module,
        f"{proj}_precision_config",
        PrecisionConfig(weight_scale=weight_scale, flex_ctx=FlexCtx(rhs_data=InFlexData())),
    )


def replace_with_mxfp4_linear(model, quantization_config=None, modules_to_not_convert: list[str] | None = None):
    """
    Public method that replaces the expert layers of the given model with mxfp4 quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        quantization_config (`Mxfp4Config`, defaults to `None`):
            The quantization config object that contains the quantization parameters.
        modules_to_not_convert (`list`, *optional*, defaults to `None`):
            A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
    """
    if quantization_config.dequantize:
        return model

    from .hub_kernels import get_kernel

    global triton_kernels_hub
    triton_kernels_hub = get_kernel("kernels-community/gpt-oss-triton-kernels", version=1)

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        if module.__class__.__name__ == "GptOssExperts" and not quantization_config.dequantize:
            with torch.device("meta"):
                model.set_submodule(module_name, Mxfp4GptOssExperts(model.config))
                has_been_replaced = True
        if module.__class__.__name__ == "GptOssMLP" and not quantization_config.dequantize:
            from types import MethodType

            module.forward = MethodType(mlp_forward, module)

    if not has_been_replaced:
        logger.warning(
            "You are loading your model using mixed-precision FP4 quantization but no linear modules were found in your model."
            " Please double check your model architecture, or submit an issue on github if you think this is"
            " a bug."
        )

    return model
