
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ... import initialization as init
from ...cache_utils import Cache
from ...integrations.accelerate import force_accelerate_hooks
from ...utils import auto_docstring, logging
from ...utils.import_utils import is_mambapy_available, is_torch_greater_or_equal, is_torchdynamo_compiling, is_tracing
from ..mamba.configuration_mamba import MambaConfig
from ..mamba.modeling_mamba import (
    MambaBlock,
    MambaCausalLMOutput,
    MambaForCausalLM,
    MambaMixer,
    MambaModel,
    MambaOutput,
    MambaPreTrainedModel,
    MambaRMSNorm,
)


logger = logging.get_logger(__name__)

if is_torch_greater_or_equal("2.9.0"):
    from torch._higher_order_ops.associative_scan import associative_scan
else:
    associative_scan = None

if is_mambapy_available():
    from mambapy.pscan import pscan
else:
    pscan = None

selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, falcon_mamba_inner_fn = (
    None,
    None,
    None,
    None,
    None,
)


@auto_docstring(checkpoint="tiiuae/falcon-mamba-7b")
@strict
class FalconMambaConfig(MambaConfig):

    use_falcon_mambapy: bool = False
    use_associative_scan: bool = True
    mixer_rms_eps: float = 1e-6

    @property
    def layer_types(self):
        pass


def rms_forward(hidden_states, variance_epsilon=1e-6):
    """
    Calculates simple RMSNorm with no learnable weights. `MambaRMSNorm` will
    leverage this in order to multiply the final result with the RMSNorm weight

    Args:
        hidden_states (`torch.Tensor`):
            Hidden states to normalize
        variance_epsilon (`float`):
            The eps value to add in the square root scaling factor
    """
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


class FalconMambaMixer(MambaMixer):
    def warn_slow_implementation(self):
        is_fast_path_available = all(
            (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, falcon_mamba_inner_fn)
        )
        if not is_fast_path_available:
            if self.use_falcon_mambapy:
                if is_mambapy_available():
                    logger.warning_once(
                        "The fast path is not available because one of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                        " is None. Falling back to the mamba.py backend. The recommended way to enable the fast path is `pip install kernels`, which provides the"
                        " FalconMamba kernels (loaded on demand). Alternatively, install mamba-ssm (https://github.com/state-spaces/mamba/#installation) and"
                        " causal-conv1d (https://github.com/Dao-AILab/causal-conv1d)."
                    )
                else:
                    raise ImportError(
                        "use_mambapy is set to True but the mambapy package is not installed. To install it follow https://github.com/alxndrTL/mamba.py."
                    )
            else:
                logger.warning_once(
                    "The fast path is not available because one of `(selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)`"
                    " is None. Falling back to the sequential implementation of Mamba, as use_mambapy is set to False. The recommended way to enable the fast path is"
                    " `pip install kernels`, which provides the FalconMamba kernels (loaded on demand). Alternatively, install mamba-ssm"
                    " (https://github.com/state-spaces/mamba/#installation) and causal-conv1d (https://github.com/Dao-AILab/causal-conv1d)."
                    " For the mamba.py backend, follow https://github.com/alxndrTL/mamba.py."
                )

    def __init__(self, config: FalconMambaConfig, layer_idx: int, initialize_mixer_weights: bool = True):
        super().__init__(config, layer_idx)

        self.register_buffer("b_c_rms", torch.ones(self.ssm_state_size, requires_grad=False), persistent=False)
        self.register_buffer("dt_rms", torch.ones(self.intermediate_size, requires_grad=False), persistent=False)
        self.rms_eps = config.mixer_rms_eps

    def cuda_kernels_forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
    ):
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        if self.training and cache_params is None:  # Doesn't support outputting the states -> used for training
            contextualized_states = falcon_mamba_inner_fn(
                projected_states,
                self.conv1d.weight,
                self.conv1d.bias if self.use_conv_bias else None,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias.float() if self.use_bias else None,
                -torch.exp(self.A_log.float()),
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                b_rms_weight=self.b_c_rms,
                c_rms_weight=self.b_c_rms,
                dt_rms_weight=self.dt_rms,
                b_c_dt_rms_eps=self.rms_eps,
            )

        else:
            hidden_states, gate = projected_states.chunk(2, dim=1)

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

            is_decoding = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

            conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
            if is_decoding:
                hidden_states = causal_conv1d_update(
                    hidden_states.squeeze(-1),
                    cache_params.layers[self.layer_idx].conv_states[0],
                    conv_weights,
                    self.conv1d.bias,
                    self.activation,
                )
                hidden_states = hidden_states.unsqueeze(-1)
            else:
                if cache_params is not None:
                    conv_states = nn.functional.pad(
                        hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0)
                    )
                    cache_params.update_conv_state(conv_states, self.layer_idx)
                hidden_states = causal_conv1d_fn(
                    hidden_states, conv_weights, self.conv1d.bias, activation=self.activation
                )

            if attention_mask is not None:
                hidden_states = hidden_states * attention_mask.unsqueeze(1)

            ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
            time_step, B, C = torch.split(
                ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
            )

            B = rms_forward(B, variance_epsilon=self.rms_eps)
            C = rms_forward(C, variance_epsilon=self.rms_eps)
            time_step = rms_forward(time_step, variance_epsilon=self.rms_eps)

            if hasattr(self.config, "_is_quantized"):
                discrete_time_step = (self.dt_proj(time_step) - self.dt_proj.bias).transpose(1, 2)
            else:
                discrete_time_step = self.dt_proj.weight @ time_step.transpose(1, 2)

            A = -torch.exp(self.A_log.float())
            time_proj_bias = self.dt_proj.bias.float() if hasattr(self.dt_proj, "bias") else None
            if is_decoding:
                scan_outputs = selective_state_update(
                    cache_params.layers[self.layer_idx].recurrent_states[0],
                    hidden_states[..., 0],
                    discrete_time_step[..., 0],
                    A,
                    B[:, 0],
                    C[:, 0],
                    self.D,
                    gate[..., 0],
                    time_proj_bias,
                    dt_softplus=True,
                ).unsqueeze(-1)
            else:
                scan_outputs, ssm_state = selective_scan_fn(
                    hidden_states,
                    discrete_time_step,
                    A,
                    B.transpose(1, 2),
                    C.transpose(1, 2),
                    self.D.float(),
                    gate,
                    time_proj_bias,
                    delta_softplus=True,
                    return_last_state=True,
                )
                if ssm_state is not None and cache_params is not None:
                    cache_params.update_recurrent_state(ssm_state, self.layer_idx)

            contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def slow_forward(
        self,
        input_states,
        cache_params: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(1, 2)  # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        if cache_params is not None and cache_params.has_previous_state(self.layer_idx):
            ssm_state = cache_params.layers[self.layer_idx].recurrent_states[0].clone()
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size), device=hidden_states.device, dtype=dtype
            )

        if cache_params is not None:
            if not cache_params.has_previous_state(self.layer_idx):
                conv_state = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))

                cache_params.update_conv_state(conv_state, self.layer_idx)
                hidden_states = self.act(
                    self.conv1d(hidden_states)[..., :seq_len]
                )  # [batch, intermediate_size, seq_len]
            else:
                conv_state = cache_params.update_conv_state(hidden_states, self.layer_idx)[
                    ..., -self.conv_kernel_size :
                ]
                conv_state = conv_state.to(self.conv1d.weight.device)
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = (
                    self.act(hidden_states).to(dtype).unsqueeze(-1)
                )  # [batch, intermediate_size, 1] : decoding
        else:
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])  # [batch, intermediate_size, seq_len]

        if attention_mask is not None:
            hidden_states = hidden_states * attention_mask.unsqueeze(1)

        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        B = rms_forward(B, variance_epsilon=self.rms_eps)
        C = rms_forward(C, variance_epsilon=self.rms_eps)
        time_step = rms_forward(time_step, variance_epsilon=self.rms_eps)

        discrete_time_step = self.dt_proj(time_step)  # [batch, seq_len, intermediate_size]
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(
            1, 2
        )  # [batch, intermediate_size, seq_len]

        A = -torch.exp(self.A_log.float())  # [intermediate_size, ssm_state_size]
        discrete_A = torch.exp(
            A[None, :, None, :] * discrete_time_step[:, :, :, None]
        )  # [batch, intermediate_size, seq_len, ssm_state_size]
        discrete_B = (
            discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        )  # [batch, intermediate_size, seq_len, ssm_state_size]
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        if self.use_falcon_mambapy and self.training and cache_params is None:
            hs = pscan(
                discrete_A.transpose(1, 2), deltaB_u.transpose(1, 2)
            )  # [batch, seq_len, intermediate_size, ssm_state_size]
            scan_output = (hs @ C.unsqueeze(-1)).squeeze(3).transpose(1, 2)  # [batch, intermediate_size, seq_len]
            scan_output = scan_output + hidden_states * self.D[None, :, None]
            scan_output = scan_output * self.act(gate)
        else:
            if (
                self.use_associative_scan
                and associative_scan is not None
                and is_tracing(hidden_states)
                and cache_params is None
            ):

                def combine_fn(left, right):
                    pass

                combine_mode = "pointwise" if discrete_A.device.type in ("cuda", "xpu") else "generic"
                _, all_h = associative_scan(combine_fn, (discrete_A, deltaB_u), dim=2, combine_mode=combine_mode)
                scan_output = (
                    torch.matmul(all_h.permute(0, 2, 1, 3).to(dtype), C.unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)
                )
                ssm_state = all_h[:, :, -1, :]
            else:
                # Sequential loop for decoding or when associative_scan unavailable
                scan_outputs = []
                for i in range(seq_len):
                    ssm_state = (
                        discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
                    )  # [batch, intermediate_size, ssm_state]
                    scan_output = torch.matmul(
                        ssm_state.to(dtype), C[:, i, :].unsqueeze(-1)
                    )  # [batch, intermediate_size, 1]
                    scan_outputs.append(scan_output[:, :, 0])
                scan_output = torch.stack(scan_outputs, dim=-1)  # [batch, intermediate_size, seq_len]

            scan_output = scan_output + (hidden_states * self.D[None, :, None])
            scan_output = scan_output * self.act(gate)

            if cache_params is not None:
                cache_params.update_recurrent_state(ssm_state, self.layer_idx)

        contextualized_states = self.out_proj(scan_output.transpose(1, 2))  # [batch, seq_len, hidden_size]
        return contextualized_states

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states,
        cache_params: Cache | None = None,
        attention_mask: torch.LongTensor | None = None,
        **kwargs,
    ):
        is_fast_path_available = all(
            (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, falcon_mamba_inner_fn)
        )
        if is_fast_path_available and "cuda" in self.x_proj.weight.device.type and not is_torchdynamo_compiling():
            return self.cuda_kernels_forward(hidden_states, cache_params, attention_mask)
        return self.slow_forward(hidden_states, cache_params, attention_mask)

    @torch.no_grad()
    def init_falcon_mamba_weights(self):
        super().init_falcon_mamba_weights()
        init.ones_(self.b_c_rms)
        init.ones_(self.dt_rms)


class FalconMambaRMSNorm(MambaRMSNorm):
    def forward(self, hidden_states):
        return self.weight.to(hidden_states.device) * rms_forward(
            hidden_states, variance_epsilon=self.variance_epsilon
        )


class FalconMambaBlock(MambaBlock):
    pass


@auto_docstring
class FalconMambaPreTrainedModel(MambaPreTrainedModel):
    pass


class FalconMambaOutput(MambaOutput):
    pass


class FalconMambaCausalLMOutput(MambaCausalLMOutput):
    pass


class FalconMambaModel(MambaModel, FalconMambaPreTrainedModel):
    def __init__(self, config):
        FalconMambaPreTrainedModel.__init__(self, config)

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [FalconMambaBlock(config, layer_idx=idx) for idx in range(config.num_hidden_layers)]
        )

        self.gradient_checkpointing = False
        self.norm_f = FalconMambaRMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def load_hook(self, state_dict, prefix, *args):
        raise AttributeError("Not needed for FalconMamba")


class FalconMambaForCausalLM(MambaForCausalLM):
    pass


__all__ = [
    "FalconMambaForCausalLM",
    "FalconMambaModel",
    "FalconMambaPreTrainedModel",
    "FalconMambaConfig",
]
