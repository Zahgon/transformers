
import functools
import math
from collections import OrderedDict

import torch
from torch import Tensor, nn

from .integrations.hub_kernels import use_kernel_forward_from_hub
from .utils import logging
from .utils.import_utils import is_torchdynamo_compiling


logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("GeluTanh")
class GELUTanh(nn.Module):

    def __init__(self, use_gelu_tanh_python: bool = False):
        super().__init__()
        if use_gelu_tanh_python:
            self.act = self._gelu_tanh_python
        else:
            self.act = functools.partial(nn.functional.gelu, approximate="tanh")

    def _gelu_tanh_python(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


PytorchGELUTanh = GELUTanh


@use_kernel_forward_from_hub("NewGELU")
class NewGELUActivation(nn.Module):

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


@use_kernel_forward_from_hub("GeLU")
class GELUActivation(nn.Module):

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


@use_kernel_forward_from_hub("SiLU")
class SiLUActivation(nn.Module):

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.silu(input)


@use_kernel_forward_from_hub("FastGELU")
class FastGELUActivation(nn.Module):

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


@use_kernel_forward_from_hub("QuickGELU")
class QuickGELUActivation(nn.Module):

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))


class MishActivation(nn.Module):

    def __init__(self):
        super().__init__()
        self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class LinearActivation(nn.Module):

    def forward(self, input: Tensor) -> Tensor:
        return input


class LaplaceActivation(nn.Module):

    def forward(self, input, mu=0.707107, sigma=0.282095):
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(input))


class ReLUSquaredActivation(nn.Module):

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared


class SqrtSoftplusActivation(nn.Module):

    def forward(self, input):
        return nn.functional.softplus(input).sqrt()


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


class XIELUActivation(nn.Module):

    def __init__(
        self,
        alpha_p_init=0.8,
        alpha_n_init=0.8,
        beta=0.5,
        eps=-1e-6,
        dtype=torch.bfloat16,
        with_vector_loads=False,
    ):
        super().__init__()
        self.alpha_p = nn.Parameter(torch.log(torch.expm1(torch.tensor(alpha_p_init, dtype=dtype))).unsqueeze(0))
        self.alpha_n = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(alpha_n_init - beta, dtype=dtype))).unsqueeze(0)
        )
        self.register_buffer("beta", torch.tensor(beta, dtype=dtype))
        self.register_buffer("eps", torch.tensor(eps, dtype=dtype))
        self.with_vector_loads = with_vector_loads
        self._beta_scalar = float(beta)
        self._eps_scalar = float(eps)

        self._xielu_cuda_obj = None
        try:
            import xielu.ops  # noqa: F401

            self._xielu_cuda_obj = torch.classes.xielu.XIELU()
            msg = "Using experimental xIELU CUDA."
            try:
                from torch.compiler import allow_in_graph

                self._xielu_cuda_fn = allow_in_graph(self._xielu_cuda)
                msg += " Enabled torch._dynamo for xIELU CUDA."
            except Exception as err:
                msg += f" Could not enable torch._dynamo for xIELU ({err}) - this may result in slower performance."
                self._xielu_cuda_fn = self._xielu_cuda
            logger.warning_once(msg)
        except Exception as err:
            logger.warning_once(
                f"CUDA-fused xIELU not available ({err}) – falling back to a Python version.\n"
                "For CUDA xIELU (experimental), `pip install git+https://github.com/nickjbrowning/XIELU`"
            )

    def _xielu_python(self, x: Tensor) -> Tensor:
        alpha_p = nn.functional.softplus(self.alpha_p)
        alpha_n = self.beta + nn.functional.softplus(self.alpha_n)
        return torch.where(
            x > 0,
            alpha_p * x * x + self.beta * x,
            (torch.expm1(torch.min(x, self.eps)) - x) * alpha_n + self.beta * x,
        )

    def _xielu_cuda(self, x: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        if self._xielu_cuda_obj is not None and input.is_cuda:
            if not is_torchdynamo_compiling():
                return self._xielu_cuda_fn(input)
            else:
                logger.warning_once("torch._dynamo is compiling, using Python version of xIELU.")
        return self._xielu_python(input)


ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": GELUTanh,
    "gelu_python_tanh": (GELUTanh, {"use_gelu_tanh_python": True}),
    "gelu_accurate": AccurateGELUActivation,
    "hardswish": nn.Hardswish,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": SiLUActivation,
    "sqrtsoftplus": SqrtSoftplusActivation,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU,
    "xielu": XIELUActivation,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
gelu_pytorch_tanh = get_activation("gelu_pytorch_tanh")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")
