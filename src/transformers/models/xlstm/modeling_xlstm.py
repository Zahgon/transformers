
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from ... import initialization as init
from ...generation import GenerationMixin
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_utils import PreTrainedModel
from ...processing_utils import Unpack
from ...utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, is_xlstm_available
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from .configuration_xlstm import xLSTMConfig


if is_xlstm_available():
    from xlstm.xlstm_large.model import RMSNorm as xLSTMRMSNorm
    from xlstm.xlstm_large.model import mLSTMBlock, mLSTMStateType, soft_cap

    external_xlstm = True

    class xLSTMBlock(GradientCheckpointingLayer, mLSTMBlock):
        pass

else:
    from collections.abc import Callable
    from functools import partial
    from typing import Literal

    from .configuration_xlstm import round_up_to_next_multiple_of

    mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    mLSTMStateType = dict[int, mLSTMLayerStateType]

    external_xlstm = False

    def soft_cap(values: torch.Tensor, cap_value: float | torch.Tensor | None = None) -> torch.Tensor:
        """
        Soft caps a tensor to a value.

        Performs a tanh operation on the logits and scales the result to the cap value. Common technique in attention
        and output language heads to prevent large logits from dominating the softmax. See for example Gemma2:
        https://huggingface.co/papers/2408.00118

        Args:
            values: The tensor to cap.
            cap_value: The value to cap the values to. If None, no cap is applied.

        Returns:
            The capped values.
        """
        if cap_value is None:
            return values
        return cap_value * torch.tanh(values / cap_value)

    def mlstm_chunkwise_recurrent_fw_C(
        matK: torch.Tensor,
        matV: torch.Tensor,
        vecB: torch.Tensor,
        vecI: torch.Tensor,
        matC_states: torch.Tensor | None = None,
        vecN_states: torch.Tensor | None = None,
        scaMinter_states: torch.Tensor | None = None,
        matC_initial: torch.Tensor | None = None,
        vecN_initial: torch.Tensor | None = None,
        scaMinter_initial: torch.Tensor | None = None,
        qk_scale: float | None = None,
        chunk_size: int = 64,
        num_chunks: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def mlstm_chunkwise_parallel_fw_H(
        matQ: torch.Tensor,
        matK: torch.Tensor,
        matV: torch.Tensor,
        matC_states: torch.Tensor,
        vecN_states: torch.Tensor,
        scaMinter_states: torch.Tensor,
        vecI: torch.Tensor,
        vecB: torch.Tensor,
        qk_scale: float,
        chunk_size: int = 64,
        num_chunks: int = 1,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def mlstm_chunkwise_fw(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        cstate: torch.Tensor | None = None,
        nstate: torch.Tensor | None = None,
        mstate: torch.Tensor | None = None,
        qk_scale: float | None = None,
        return_last_states: bool = False,
        return_all_states: bool = False,
        chunk_size: int = 64,
        eps: float = 1e-6,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        pass

    def mlstm_chunkwise_native_autograd(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        c_initial: torch.Tensor | None = None,
        n_initial: torch.Tensor | None = None,
        m_initial: torch.Tensor | None = None,
        return_last_states: bool = False,
        eps: float = 1e-6,
        chunk_size: int = 64,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pass

    def mlstm_recurrent_step_native(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        cstate: torch.Tensor,
        nstate: torch.Tensor,
        mstate: torch.Tensor,
        eps: float = 1e-6,
        dtype_state: torch.dtype = torch.float32,
        **kwargs,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pass

    def mlstm_recurrent_sequence_native(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        igate: torch.Tensor,
        fgate: torch.Tensor,
        c_initial: torch.Tensor | None = None,
        n_initial: torch.Tensor | None = None,
        m_initial: torch.Tensor | None = None,
        return_last_states: bool = False,
        eps: float = 1e-6,
        dtype_state: torch.dtype = torch.float32,
        **kwargs,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        pass

    def wrap_chunkwise_pad_zeros(
        mlstm_chunkwise_kernel: Callable,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fgate: torch.Tensor,
        igate: torch.Tensor,
        c_initial: torch.Tensor | None = None,
        n_initial: torch.Tensor | None = None,
        m_initial: torch.Tensor | None = None,
        return_last_states: bool = False,
        eps: float = 1e-6,
        autocast_kernel_dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 64,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pass

    def wrap_chunkwise_arbitrary_sequence_length(
        mlstm_chunkwise_kernel: Callable,
        mlstm_sequence_kernel: Callable,
        mlstm_step_kernel: Callable,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        fgate: torch.Tensor,
        igate: torch.Tensor,
        c_initial: torch.Tensor | None = None,
        n_initial: torch.Tensor | None = None,
        m_initial: torch.Tensor | None = None,
        return_last_states: bool = True,
        eps: float = 1e-6,
        autocast_kernel_dtype: torch.dtype = torch.bfloat16,
        chunk_size: int = 64,
        enable_logging: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        pass

    class xLSTMBackend(nn.Module):

        config_class = xLSTMConfig

        def __init__(self, config: xLSTMConfig):
            super().__init__()
            self.config = config
            self.chunkwise_kernel_fn = mlstm_chunkwise_native_autograd
            self.sequence_kernel_fn = mlstm_recurrent_sequence_native
            self.step_kernel_fn = mlstm_recurrent_step_native

            self._inference_fn = partial(
                wrap_chunkwise_arbitrary_sequence_length,
                mlstm_chunkwise_kernel=self.chunkwise_kernel_fn,
                mlstm_sequence_kernel=partial(
                    self.sequence_kernel_fn,
                    dtype_state=getattr(torch, config.inference_state_dtype),
                ),
                mlstm_step_kernel=partial(
                    self.step_kernel_fn,
                    dtype_state=getattr(torch, config.inference_state_dtype),
                ),
                chunk_size=config.chunk_size,
                eps=config.eps,
                autocast_kernel_dtype=getattr(torch, config.autocast_kernel_dtype),
                return_last_states=True,
            )

            train_kernel_fn = partial(
                self.chunkwise_kernel_fn,
                autocast_kernel_dtype=getattr(torch, config.autocast_kernel_dtype),
                eps=config.eps,
                chunk_size=config.chunk_size,
            )
            if "with_padding" in config.mode:
                train_kernel_fn = partial(wrap_chunkwise_pad_zeros, mlstm_chunkwise_kernel=train_kernel_fn)
            self._train_fn = train_kernel_fn

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            igate: torch.Tensor,
            fgate: torch.Tensor,
            c_initial: torch.Tensor | None = None,
            n_initial: torch.Tensor | None = None,
            m_initial: torch.Tensor | None = None,
            return_last_states: bool | None = None,
            mode: Literal["train", "inference"] | None = None,
        ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            """Forward pass of the mLSTM backend.

            Depending on the configured mode, this method will call the appropriate kernel function.

            Args:
                query: The query tensor of shape (batch_size, nh, sequence_length, dhqk).
                key: The key tensor of shape (batch_size, nh, sequence_length, dhqk).
                value: The value tensor of shape (batch_size, nh, sequence_length, dhhv).
                igate: The input gate preactivation tensor of shape (batch_size, nh, sequence_length).
                fgate: The forget gate preactivation tensor of shape (batch_size, nh, sequence_length).
                c_initial: The initial cell state tensor of shape (batch_size, nh, dhqk, dhhv).
                                                    Defaults to None.
                n_initial: The initial hidden state tensor of shape (batch_size, nh, dhqk). Defaults to None.
                m_initial: The initial memory tensor of shape (batch_size, nh, 1). Defaults to None.
                return_last_states: Whether to return the last states of the sequence. Defaults to None.
                                                    If None, the value from the config is used.

            Returns:
                hidden states of shape (batch_size, nh, sequence_length, dhhv)
                hidden states and last states the last states are the cell state cstate (batch_size, nh, dhqk, dhhv),
                the normalizer state nstate (batch_size, nh, dhqk), and the max state mstate (batch_size, nh, 1)
            """
            if mode is None:
                mode = self.config.mode

            if "train" in mode:
                if return_last_states is None:
                    return_last_states = self.config.return_last_states

                if self.config.mode == "train_with_padding":
                    if return_last_states:
                        raise ValueError("return_last_states=True is not supported with train_with_padding mode.")

                return self._train_fn(
                    query=query,
                    key=key,
                    value=value,
                    igate=igate,
                    fgate=fgate,
                    c_initial=c_initial,
                    n_initial=n_initial,
                    m_initial=m_initial,
                    return_last_states=return_last_states,
                )

            elif "inference" in mode:
                return self._inference_fn(
                    query=query,
                    key=key,
                    value=value,
                    igate=igate,
                    fgate=fgate,
                    c_initial=c_initial,
                    n_initial=n_initial,
                    m_initial=m_initial,
                )
            else:
                raise ValueError(f"Unknown mode: {self.config.mode}")

        def extra_repr(self) -> str:
            pass

    class xLSTMRMSNorm(nn.Module):

        def __init__(
            self,
            num_features: int,
            eps: float = 1e-6,
            use_weight: bool = True,
            use_bias: bool = False,
            force_float32_reductions: bool = True,
        ):
            super().__init__()
            self.num_features = num_features
            self.eps = eps
            self.force_float32_reductions = force_float32_reductions

            if use_weight:
                self.weight = nn.Parameter(torch.ones(num_features))
            else:
                self.weight = None

            if use_bias:
                self.bias = nn.Parameter(torch.zeros(num_features))
            else:
                self.bias = None

        def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
            return x

        def _rms_normalize(self, x: torch.Tensor) -> torch.Tensor:
            in_dtype = x.dtype
            if self.force_float32_reductions:
                x = x.float()
            x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return x.to(in_dtype)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self._rms_normalize(x)
            x = self._apply_weight_bias(x)
            return x

    class xLSTMMultiHeadLayerNorm(nn.Module):

        def __init__(
            self,
            num_heads: int,
            head_dim: int,
            eps: float = 1e-6,
            use_weight: bool = True,
            use_bias: bool = False,
            force_float32_reductions: bool = True,
        ):
            super().__init__()
            self.num_features = num_heads * head_dim
            self.eps = eps
            self.force_float32_reductions = force_float32_reductions

            if use_weight:
                self.weight = nn.Parameter(torch.ones(self.num_features))
            else:
                self.weight = None

            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.num_features))
            else:
                self.bias = None
            self.num_heads = num_heads
            self.head_dim = head_dim

        def _apply_weight_bias(self, x: torch.Tensor) -> torch.Tensor:
            if self.weight is not None:
                x = x * self.weight
            if self.bias is not None:
                x = x + self.bias
            return x

        def _layer_normalize(self, x: torch.Tensor) -> torch.Tensor:
            in_dtype = x.dtype
            if self.force_float32_reductions:
                x = x.float()
            x_centered = x - x.mean(dim=-1, keepdim=True)
            y = x_centered * torch.rsqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
            return y.to(in_dtype)

        def forward(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:
            batch_size, sequence_length, nh, DH = x.shape
            if nh != self.num_heads:
                raise ValueError(f"Expected {self.num_heads} heads, got {nh}, input shape: {x.shape}")
            if self.head_dim != DH:
                raise ValueError(f"Expected {self.head_dim} head dimension, got {DH}, input shape: {x.shape}")

            x = self._layer_normalize(x)
            x = x.reshape(batch_size, sequence_length, -1)
            x = self._apply_weight_bias(x)
            return x

    class xLSTMFeedForward(nn.Module):
        def __init__(self, config: xLSTMConfig):
            super().__init__()
            self.config = config

            self.up_proj_dim = round_up_to_next_multiple_of(
                config.hidden_size * config.ffn_proj_factor,
                config.ffn_round_up_to_multiple_of,
            )

            if self.config.weight_mode == "single":
                self.proj_up_gate = nn.Linear(
                    in_features=config.hidden_size,
                    out_features=self.up_proj_dim,
                    bias=self.config.use_bias,
                )
                self.proj_up = nn.Linear(
                    in_features=config.hidden_size,
                    out_features=self.up_proj_dim,
                    bias=self.config.use_bias,
                )
            elif self.config.weight_mode == "fused":
                self.proj_up_gate_z = nn.Linear(
                    in_features=config.hidden_size,
                    out_features=2 * self.up_proj_dim,
                    bias=self.config.use_bias,
                )

            self.proj_down = nn.Linear(
                in_features=self.up_proj_dim,
                out_features=config.hidden_size,
                bias=self.config.use_bias,
            )

            self.act_fn = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.config.weight_mode == "single":
                x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
            elif self.config.weight_mode == "fused":
                x = self.proj_up_gate_z(x)
                gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
                x = self.act_fn(gate) * z

            y = self.proj_down(x)
            return y

    class xLSTMLayer(nn.Module):
        def __init__(self, config: xLSTMConfig):
            super().__init__()
            self.config = config

            self.v_dim = int(config.hidden_size * config.v_dim_factor)
            self.qk_dim = int(config.hidden_size * config.qk_dim_factor)

            if self.config.weight_mode == "single":
                self.q = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.qk_dim,
                    bias=self.config.use_bias,
                )
                self.k = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.qk_dim,
                    bias=self.config.use_bias,
                )
                self.v = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.v_dim,
                    bias=self.config.use_bias,
                )

                self.ogate_preact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.v_dim,
                    bias=self.config.use_bias,
                )
                self.igate_preact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.config.num_heads,
                    bias=True,
                )
                self.fgate_preact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=self.config.num_heads,
                    bias=True,
                )
            elif self.config.weight_mode == "fused":
                self.qkv_opreact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=2 * self.qk_dim + 2 * self.v_dim,
                    bias=self.config.use_bias,
                )
                self.ifgate_preact = nn.Linear(
                    in_features=self.config.hidden_size,
                    out_features=2 * self.config.num_heads,
                    bias=True,
                )

            self.ogate_act_fn = nn.Sigmoid()
            self.mlstm_backend = xLSTMBackend(config=self.config)

            self.multihead_norm = xLSTMMultiHeadLayerNorm(
                num_heads=self.config.num_heads,
                head_dim=self.v_dim // self.config.num_heads,
                eps=self.config.norm_eps,
                use_weight=True,
                use_bias=self.config.use_bias,
                force_float32_reductions=self.config.norm_reduction_force_float32,
            )
            self.out_proj = nn.Linear(
                in_features=self.v_dim,
                out_features=self.config.hidden_size,
                bias=self.config.use_bias,
            )

        def forward(
            self, x: torch.Tensor, state: mLSTMLayerStateType | None = None
        ) -> tuple[torch.Tensor, mLSTMLayerStateType | None]:
            if x.ndim != 3:
                raise ValueError(f"Input must have shape [batch_size, sequence_length, HD], got {x.shape}")
            batch_size, sequence_length, _ = x.shape
            if self.config.weight_mode == "single":
                query = self.q(x)
                key = self.k(x)
                value = self.v(x)
                o_preact = self.ogate_preact(x)
                i_preact = soft_cap(self.igate_preact(x), cap_value=self.config.gate_soft_cap)
                f_preact = soft_cap(self.fgate_preact(x), cap_value=self.config.gate_soft_cap)

            elif self.config.weight_mode == "fused":
                qkv_opreact = self.qkv_opreact(x)
                query, key, value, o_preact = torch.tensor_split(
                    qkv_opreact,
                    (
                        self.qk_dim,
                        2 * self.qk_dim,
                        2 * self.qk_dim + self.v_dim,
                    ),
                    dim=-1,
                )

                if_preact = soft_cap(self.ifgate_preact(x), cap_value=self.config.gate_soft_cap)
                i_preact, f_preact = torch.tensor_split(if_preact, (self.config.num_heads,), dim=-1)

            query = query.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
            key = key.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
            value = value.reshape(batch_size, sequence_length, self.config.num_heads, -1).transpose(1, 2)
            i_preact = i_preact.transpose(1, 2)
            f_preact = f_preact.transpose(1, 2)
            if state is None:
                c_initial, n_initial, m_initial = None, None, None
            else:
                c_initial, n_initial, m_initial = state

            h, state = self.mlstm_backend(
                query=query,
                key=key,
                value=value,
                igate=i_preact,
                fgate=f_preact,
                c_initial=c_initial,
                n_initial=n_initial,
                m_initial=m_initial,
            )
            expected_h_shape = (
                batch_size,
                self.config.num_heads,
                sequence_length,
                self.v_dim // self.config.num_heads,
            )
            if h.shape != expected_h_shape:
                raise ValueError(f"Got {h.shape}, expected {expected_h_shape}")

            h = h.transpose(1, 2)
            h_norm = self.multihead_norm(h)
            h_norm = h_norm.reshape(batch_size, sequence_length, -1)

            h_out = self.ogate_act_fn(o_preact) * h_norm

            y = self.out_proj(h_out)
            return y, state

    class xLSTMBlock(GradientCheckpointingLayer):
        def __init__(self, config: xLSTMConfig):
            super().__init__()
            self.config = config
            self.norm_mlstm = xLSTMRMSNorm(
                num_features=config.hidden_size,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
            self.mlstm_layer = xLSTMLayer(config)
            self.norm_ffn = xLSTMRMSNorm(
                num_features=config.hidden_size,
                eps=config.norm_eps,
                use_weight=True,
                use_bias=config.use_bias,
                force_float32_reductions=config.norm_reduction_force_float32,
            )
            self.ffn = xLSTMFeedForward(config)

        def forward(self, x: torch.Tensor, state: mLSTMStateType | None = None) -> tuple[torch.Tensor, mLSTMStateType]:
            x_mlstm = self.norm_mlstm(x)
            x_mlstm, state = self.mlstm_layer(x_mlstm, state)
            x = x + x_mlstm

            x_ffn = self.norm_ffn(x)
            x_ffn = self.ffn(x_ffn)
            x = x + x_ffn

            return x, state


def small_init_method(dim):
    """
    Adapted from: https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution."""
    std = (2 / (5 * dim)) ** (1 / 2)

    def init_(tensor):
        pass

    return init_


def wang_init_method(n_layers, dim):
    """
    Adapted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py
    """
    std = 2 / n_layers / dim ** (1 / 2)

    def init_(tensor):
        pass

    return init_


class xLSTMPreTrainedModel(PreTrainedModel):

    config_class = xLSTMConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["xLSTMBlock"]
    supports_gradient_checkpointing = True
    _is_stateful = True
    _can_record_outputs = {
        "hidden_states": xLSTMBlock,
    }

    def _module_name_map(self, module):
        for name, mod in self.named_modules():
            if mod is module:
                return name
        return ""

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, nn.Embedding):
            small_init_method(self.config.hidden_size)(self.embeddings.weight)
        elif isinstance(module, nn.Linear):
            if module.bias is not None:
                init.zeros_(module.bias)
            if self.config.weight_mode == "single" and "gate" in self._module_name_map(module):
                init.zeros_(module.weight)

                if "igate" in self._module_name_map(module):
                    init.copy_(module.bias, -10.0 * torch.ones_like(module.bias))
                elif "fgate" in self._module_name_map(module):
                    init.copy_(
                        module.bias,
                        torch.linspace(
                            3.0,
                            6.0,
                            module.bias.shape[-1],
                        ).to(
                            device=module.bias.device,
                            dtype=module.bias.dtype,
                        ),
                    )
            elif self.config.weight_mode == "fused" and "gate" in self._module_name_map(module):
                init.zeros_(module.weight)

                init.copy_(
                    module.bias[: self.config.num_heads],
                    module.bias[: self.config.num_heads]
                    - module.bias[: self.config.num_heads]
                    - 10.0 * torch.ones_like(module.bias),
                )
                init.copy_(
                    module.bias[: self.config.num_heads],
                    module.bias[: self.config.num_heads]
                    - module.bias[self.config.num_heads :]
                    + torch.linspace(
                        3.0,
                        6.0,
                        module.bias.shape[-1],
                    ).to(
                        device=module.bias.device,
                        dtype=module.bias.dtype,
                    ),
                )
            elif "proj_down" in self._module_name_map(module):
                wang_init_method(dim=module.weight.shape[1], n_layers=self.config.num_hidden_layers)(module.weight)
            elif "out_proj" in self._module_name_map(module):
                wang_init_method(dim=self.config.hidden_size, n_layers=self.config.num_hidden_layers)(module.weight)
            elif module.weight is not None:
                small_init_method(self.config.hidden_size)(module.weight)


class xLSTMCache:

    def __init__(
        self,
        config: xLSTMConfig,
        max_batch_size: int,
        dtype: torch.dtype = torch.bfloat16,
        device: str | None = None,
        **kwargs,
    ):
        self.seqlen_offset = torch.tensor([0], dtype=int, device=device)
        self.dtype = dtype
        self.config = config
        self.rnn_state = {
            layer: (
                torch.zeros(
                    [max_batch_size, config.num_heads, config.qk_head_dim, config.v_head_dim],
                    dtype=dtype,
                    device=device,
                ),
                torch.zeros([max_batch_size, config.num_heads, config.qk_head_dim], dtype=dtype, device=device),
                torch.zeros([max_batch_size, config.num_heads, 1], dtype=dtype, device=device),
            )
            for layer in range(config.num_hidden_layers)
        }

    def reset(self):
        self.rnn_state = {
            layer: (
                torch.zeros_like(self.rnn_state[layer][0]),
                torch.zeros_like(self.rnn_state[layer][1]),
                torch.zeros_like(self.rnn_state[layer][2]),
            )
            for layer in self.rnn_state
        }


@auto_docstring
@dataclass
class xLSTMOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor | None
    cache_params: xLSTMCache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


@auto_docstring
class xLSTMModel(xLSTMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.blocks = nn.ModuleList([xLSTMBlock(config) for _ in range(config.num_blocks)])
        self.out_norm = xLSTMRMSNorm(config.hidden_size, eps=config.norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, new_embedding):
        self.embeddings = new_embedding

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.LongTensor | None = None,
        cache_params: xLSTMCache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | xLSTMOutput:
        r"""
        cache_params (`xLSTMCache`, *optional*):
            The xLSTMCache that carries the RNN states.
        """
        output_hidden_states = kwargs.get("output_hidden_states")
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        if use_cache and cache_params is None:
            cache_params = xLSTMCache(
                self.config, inputs_embeds.size(0), device=inputs_embeds.device, dtype=inputs_embeds.dtype
            )

        hidden_states = inputs_embeds

        if (
            not self.training
            and self.config.max_inference_chunksize < hidden_states.shape[1]
            and not output_hidden_states
        ):
            offset = 0
            with torch.no_grad():
                if cache_params is None:
                    cache_params = xLSTMCache(config=self.config, max_batch_size=hidden_states.shape[0])
                final_state = torch.zeros_like(hidden_states)
                while offset < hidden_states.shape[1]:
                    hidden_states_chunk = hidden_states[
                        :, offset : min(offset + self.config.max_inference_chunksize, hidden_states.shape[1])
                    ]
                    for layer_idx, xlstm_block in enumerate(self.blocks):
                        hidden_states_chunk, rnn_state = xlstm_block(
                            hidden_states_chunk,
                            state=cache_params.rnn_state[layer_idx],
                        )
                        for state_idx in range(len(cache_params.rnn_state[layer_idx])):
                            local_rnn_state = rnn_state[state_idx]
                            cache_params.rnn_state[layer_idx][state_idx].copy_(local_rnn_state)
                        cache_params.rnn_state_initial = False
                    final_state[
                        :, offset : min(offset + self.config.max_inference_chunksize, hidden_states.shape[1])
                    ] = hidden_states_chunk
                    offset += self.config.max_inference_chunksize
                hidden_states = final_state
        else:
            for layer_idx, xlstm_block in enumerate(self.blocks):
                hidden_states, rnn_state = xlstm_block(
                    hidden_states,
                    cache_params.rnn_state[layer_idx] if cache_params is not None else None,
                )

                if cache_params:
                    for state_idx in range(len(cache_params.rnn_state[layer_idx])):
                        local_rnn_state = rnn_state[state_idx]
                        cache_params.rnn_state[layer_idx][state_idx].copy_(local_rnn_state)
                    cache_params.rnn_state_initial = False

        if use_cache:
            cache_params.seqlen_offset += inputs_embeds.shape[1]

        hidden_states = self.out_norm(hidden_states)

        return xLSTMOutput(
            last_hidden_state=hidden_states,
            cache_params=cache_params,
        )


@auto_docstring
@dataclass
class xLSTMCausalLMOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    cache_params: xLSTMCache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None


@auto_docstring
class xLSTMForCausalLM(xLSTMPreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = xLSTMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_input_embeddings(self):
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        return self.backbone.set_input_embeddings(new_embeddings)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_params: xLSTMCache | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple | xLSTMCausalLMOutput:
        r"""
        cache_params (`xLSTMCache`, *optional*):
            The xLSTMCache that carries the RNN states.
        """
        xlstm_outputs = self.backbone(
            input_ids,
            cache_params=cache_params,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = xlstm_outputs[0]

        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype)).float()

        if not self.training and self.config.max_inference_chunksize < logits.shape[1]:
            offset = 0
            with torch.no_grad():
                while offset < logits.shape[1]:
                    logits[:, offset : min(offset + self.config.max_inference_chunksize, logits.shape[1])] = soft_cap(
                        logits[:, offset : min(offset + self.config.max_inference_chunksize, logits.shape[1])],
                        self.config.output_logit_soft_cap,
                    )
                    offset += self.config.max_inference_chunksize
        else:
            logits = soft_cap(logits, self.config.output_logit_soft_cap)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return xLSTMCausalLMOutput(
            loss=loss,
            logits=logits,
            cache_params=xlstm_outputs.cache_params,
            hidden_states=xlstm_outputs.hidden_states,
        )


__all__ = [
    "xLSTMForCausalLM",
    "xLSTMModel",
    "xLSTMPreTrainedModel",
]
