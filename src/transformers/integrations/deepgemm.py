

from __future__ import annotations

import functools
import json
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..utils import logging
from ..utils.deprecation import deprecate_kwarg
from ..utils.import_utils import (
    KERNELS_MAX_VERSION,
    KERNELS_MIN_VERSION,
    is_kernels_available,
    is_torchdynamo_compiling,
    resolve_internal_import,
)
from .hub_kernels import lazy_load_kernel
from .tensor_parallel import to_local


logger = logging.get_logger(__name__)



@dataclass(frozen=True)
class DeepGEMM:

    fp8_fp4_matmul: Callable
    grouped_fp8_fp4_matmul_nt: Callable
    grouped_fp8_fp4_matmul_nn: Callable
    grouped_bf16_matmul_nt: Callable
    grouped_bf16_matmul_nn: Callable
    per_token_cast_to_fp8: Callable
    transform_sf_into_required_layout: Callable
    transform_weights_for_mega_moe: Callable
    get_symm_buffer_for_mega_moe: Callable
    fp8_fp4_mega_moe: Callable
    m_alignment: int


@functools.cache
def _get_cuda_home() -> str | None:
    """Resolve the CUDA toolkit root the way DeepGEMM's JIT does:
    ``CUDA_HOME`` → ``CUDA_PATH`` → dir of ``which nvcc`` → ``/usr/local/cuda`` (``None`` if none found).

    Mirrors DeepGEMM's own ``_find_cuda_home`` so we agree on the path it will actually use, rather than
    reusing ``torch.utils.cpp_extension.CUDA_HOME`` whose resolution inits a CUDA context (fork-unsafe).
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        return cuda_home
    nvcc = shutil.which("nvcc")
    if nvcc:
        return os.path.dirname(os.path.dirname(nvcc))
    if os.path.isdir("/usr/local/cuda"):
        return "/usr/local/cuda"
    return None


@functools.cache
def _get_nvcc_version() -> tuple[int, int] | None:
    """Version of the CUDA toolkit nvcc will use, as ``(major, minor)``, read off disk without a
    subprocess from (in order) ``{CUDA_HOME}/version.json``, ``version.txt``, or the ``CUDA_VERSION``
    define in ``include/cuda.h``. ``None`` if unreadable. This is the compiler that builds the kernels,
    unlike ``torch.version.cuda`` (torch's bundled runtime, which never drives a JIT compile).
    """
    cuda_home = _get_cuda_home()
    if cuda_home is None:
        return None

    version_json = os.path.join(cuda_home, "version.json")
    if os.path.isfile(version_json):
        try:
            with open(version_json) as f:
                components = json.load(f)
            version = components.get("cuda_nvcc", components.get("cuda", {})).get("version", "")
            major, minor = version.split(".")[:2]
            return int(major), int(minor)
        except (OSError, ValueError, AttributeError):
            pass

    version_txt = os.path.join(cuda_home, "version.txt")
    if os.path.isfile(version_txt):
        try:
            with open(version_txt) as f:
                match = re.search(r"CUDA Version (\d+)\.(\d+)", f.read())
            if match:
                return int(match.group(1)), int(match.group(2))
        except (OSError, ValueError):  # ValueError covers UnicodeDecodeError on a non-text file
            pass

    cuda_h = os.path.join(cuda_home, "include", "cuda.h")
    if os.path.isfile(cuda_h):
        try:
            with open(cuda_h) as f:
                match = re.search(r"#define CUDA_VERSION (\d+)", f.read())
            if match:
                cuda_version = int(match.group(1))
                return cuda_version // 1000, (cuda_version % 1000) // 10
        except (OSError, ValueError):  # ValueError covers UnicodeDecodeError on a non-text file
            pass

    return None


@functools.cache
def _load_deepgemm_kernel(requires_sm100: bool = False) -> DeepGEMM | str:
    """Load DeepGEMM once or returns an error message if env or any required symbol is missing. This is wrapped in a
    function that will raise an `ImportError` with the error message. The reason we raise in the wrapper rather than
    here is that @functools.cache will only cache a return value, not an exception.

    `requires_sm100` raises a Blackwell-specific error for callers (FP4 / Mega MoE) that won't work on Hopper, instead
    of the generic SM90+ message.
    """
    if not is_torchdynamo_compiling():
        if not is_kernels_available():
            return (
                "DeepGEMM kernel requires the `kernels` package. Please install a compatible version ("
                f"{KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), e.g. `pip install kernels=="
                f"{KERNELS_MIN_VERSION}`"
            )
        if not torch.cuda.is_available():
            return "DeepGEMM kernel requires CUDA, but CUDA is not available."

        major, minor = torch.cuda.get_device_capability()
        allowed = (10,) if requires_sm100 else (9, 10)
        if major not in allowed:
            arch = "Blackwell (SM100)" if requires_sm100 else "Hopper (SM90) or Blackwell (SM100)"
            return f"DeepGEMM requires {arch}; current device is SM{major}{minor}."

        min_cuda = (12, 9) if major == 10 else (12, 3)
        cuda_home = _get_cuda_home()
        if cuda_home is None:
            return (
                f"DeepGEMM's JIT needs a CUDA toolkit ≥ {min_cuda[0]}.{min_cuda[1]}, but none was found. "
                "Set `CUDA_HOME` to a CUDA toolkit."
            )

        if not os.path.isfile(os.path.join(cuda_home, "bin", "nvcc")):
            return (
                f"DeepGEMM's JIT compiles with nvcc, but none was found in `{cuda_home}/bin`. Point "
                f"`CUDA_HOME` at a full CUDA ≥ {min_cuda[0]}.{min_cuda[1]} toolkit (not a runtime-only install)."
            )

        nvcc_version = _get_nvcc_version()
        if nvcc_version is None:
            return (
                f"DeepGEMM found nvcc in `{cuda_home}/bin` but could not read its CUDA version "
                f"(no parseable `version.json`, `version.txt`, or `include/cuda.h`). Point `CUDA_HOME` at a "
                f"complete CUDA ≥ {min_cuda[0]}.{min_cuda[1]} toolkit."
            )
        if nvcc_version < min_cuda:
            return (
                f"DeepGEMM on SM{major}{minor} needs a CUDA ≥ {min_cuda[0]}.{min_cuda[1]} toolkit, but nvcc "
                f"{nvcc_version[0]}.{nvcc_version[1]} in `{cuda_home}` is too old. Point `CUDA_HOME` at a "
                f"CUDA ≥ {min_cuda[0]}.{min_cuda[1]} toolkit."
            )

    kernel = lazy_load_kernel("deep-gemm")
    if kernel is None:
        return "Failed to load `kernels-community/deep-gemm` — check that a build matches the current torch/CUDA."

    fp8_fp4_matmul = getattr(kernel, "fp8_fp4_gemm_nt", None)
    grouped_fp8_fp4_matmul_nt = getattr(kernel, "m_grouped_fp8_fp4_gemm_nt_contiguous", None)
    grouped_fp8_fp4_matmul_nn = getattr(kernel, "m_grouped_fp8_fp4_gemm_nn_contiguous", None)
    grouped_bf16_matmul_nt = getattr(kernel, "m_grouped_bf16_gemm_nt_contiguous", None)
    grouped_bf16_matmul_nn = getattr(kernel, "m_grouped_bf16_gemm_nn_contiguous", None)
    per_token_cast_to_fp8 = resolve_internal_import(kernel, chained_path="utils.per_token_cast_to_fp8")
    transform_sf_into_required_layout = getattr(kernel, "transform_sf_into_required_layout", None)
    transform_weights_for_mega_moe = getattr(kernel, "transform_weights_for_mega_moe", None)
    get_symm_buffer_for_mega_moe = getattr(kernel, "get_symm_buffer_for_mega_moe", None)
    get_mk_alignment = getattr(kernel, "get_mk_alignment_for_contiguous_layout", None)
    fp8_fp4_mega_moe = getattr(kernel, "fp8_fp4_mega_moe", None)

    missing = [
        name
        for name, attr in [
            ("fp8_fp4_gemm_nt", fp8_fp4_matmul),
            ("m_grouped_fp8_fp4_gemm_nt_contiguous", grouped_fp8_fp4_matmul_nt),
            ("m_grouped_fp8_fp4_gemm_nn_contiguous", grouped_fp8_fp4_matmul_nn),
            ("m_grouped_bf16_gemm_nt_contiguous", grouped_bf16_matmul_nt),
            ("m_grouped_bf16_gemm_nn_contiguous", grouped_bf16_matmul_nn),
            ("utils.per_token_cast_to_fp8", per_token_cast_to_fp8),
            ("transform_sf_into_required_layout", transform_sf_into_required_layout),
            ("transform_weights_for_mega_moe", transform_weights_for_mega_moe),
            ("get_symm_buffer_for_mega_moe", get_symm_buffer_for_mega_moe),
            ("get_mk_alignment_for_contiguous_layout", get_mk_alignment),
            ("fp8_fp4_mega_moe", fp8_fp4_mega_moe),
        ]
        if attr is None
    ]
    if missing:
        return (
            f"DeepGEMM kernel is missing required symbols: {', '.join(missing)}. "
            f"Please install a compatible version ({KERNELS_MIN_VERSION} <= version < {KERNELS_MAX_VERSION}), "
            f"e.g. `pip install kernels=={KERNELS_MIN_VERSION}`"
        )

    return DeepGEMM(
        fp8_fp4_matmul=fp8_fp4_matmul,
        grouped_fp8_fp4_matmul_nt=grouped_fp8_fp4_matmul_nt,
        grouped_fp8_fp4_matmul_nn=grouped_fp8_fp4_matmul_nn,
        grouped_bf16_matmul_nt=grouped_bf16_matmul_nt,
        grouped_bf16_matmul_nn=grouped_bf16_matmul_nn,
        per_token_cast_to_fp8=per_token_cast_to_fp8,
        transform_sf_into_required_layout=transform_sf_into_required_layout,
        transform_weights_for_mega_moe=transform_weights_for_mega_moe,
        get_symm_buffer_for_mega_moe=get_symm_buffer_for_mega_moe,
        fp8_fp4_mega_moe=fp8_fp4_mega_moe,
        m_alignment=get_mk_alignment(),
    )


@torch._dynamo.allow_in_graph
def _populate_deepgemm_kernel(requires_sm100: bool = False) -> None:
    """Warm the `_load_deepgemm_kernel` cache from an opaque graph node, so Dynamo never traces the loader.

    Under `torch.compile`, Dynamo ignores `@functools.cache` and traces into `_load_deepgemm_kernel`,
    whose cold path (hub download + dynamic import via `lazy_load_kernel`) is untraceable and errors under
    `fullgraph`. `@allow_in_graph` turns the call into an opaque fx node instead — but an fx node's return
    must be proxyable, and the `DeepGEMM` bundle of Python callables isn't (`Unsupported: torch.* op
    returned non-Tensor`), so we can't just decorate the real loader. Hence two loaders: this one is
    opaque, returns `None`, and only warms the cache; the real `_load_deepgemm_kernel` right after is then
    a plain cache lookup.
    """
    _load_deepgemm_kernel(requires_sm100=requires_sm100)


def load_deepgemm_kernel(requires_sm100: bool = False) -> DeepGEMM:
    _populate_deepgemm_kernel(requires_sm100=requires_sm100)
    deepgemm_or_error = _load_deepgemm_kernel(requires_sm100=requires_sm100)
    if isinstance(deepgemm_or_error, str):
        raise ImportError(deepgemm_or_error)
    return deepgemm_or_error




@functools.cache
def _is_sm100(device: torch.device) -> bool:
    """``True`` for Blackwell (SM100+). Cached: device capability is fixed for the
    process lifetime and this gets hit on every linear/expert forward.
    """
    return torch.cuda.get_device_capability(device)[0] >= 10


def _assert_sm100_scales_are_ue8m0(scale: torch.Tensor) -> None:
    pass


def _ceil_to_ue8m0(sf: torch.Tensor) -> torch.Tensor:
    """Round each fp32 SF up to the nearest power of 2 (zero mantissa).

    Mirrors `deep_gemm.utils.math.ceil_to_ue8m0`. On SM100 the kernel's
    `pack_fp32_into_ue8m0` cleanly extracts the biased exponent only when the
    mantissa is already zero — its inner shifts (`>> 15`, `>> 7`, `<< 1`)
    otherwise leak mantissa bits into adjacent UE8M0 byte slots and silently
    corrupt the SF. SM90 consumes raw fp32 SFs without going through this path.
    """
    int_view = sf.view(torch.int32)
    return (int_view + ((1 << 23) - 1)).bitwise_and_(~((1 << 23) - 1)).view(torch.float)


def _coerce_sf_for_kernel(sf: torch.Tensor, expected_mn: int | None = None) -> torch.Tensor:
    """Lay out `sf` as DeepGEMM's dispatch expects, per arch.

    On SM100 the int-SF path only *checks* the SF (`tma_stride_check`) and never
    transforms it, so we hand it a TMA-aligned MN-major layout (`stride(-2) == 1`,
    `stride(-1) == align(mn, 16/esize)`). On SM90 DeepGEMM transforms SFA itself
    (`get_mn_major_tma_aligned_tensor`) and only *checks* SFB against
    `sm90_sfb_check`, which rejects TMA padding (`stride(-1)` must equal `size(-2)`,
    not `align(mn, …)`); a padded weight SF trips `layout.hpp` whenever `mn` isn't a
    multiple of `16/esize` (e.g. N=576 → mn=5). So on SM90 we return the raw
    row-major SF and let DeepGEMM lay it out.

    Inputs come in three flavors:
      - `float8_e8m0fnu` on SM100: raw UE8M0 bytes — pack 4 K-bytes → int32
        (last dim /4) for the kernel's `(INT, 1, gran_k)` path.
      - `float8_e8m0fnu` on SM90: SM90 dispatch only accepts FP32 SFs, so cast
        UE8M0 → FP32 (exact upcast — UE8M0 is the biased-exponent half of a
        pow-of-2 FP32, so `.float()` rebuilds the original FP32 scale exactly).
      - `float32`: per-token / per-block SFs from `per_token_cast_to_fp8` or
        on-disk weights — round to UE8M0 on SM100 (see `_ceil_to_ue8m0`).
      - `int32`: already-packed UE8M0 — pass through.

    When `expected_mn` is set and the SF's M-dim is smaller (block-quantized
    UE8M0, e.g. DSv4-Flash compressor weights with `(N/128, K/128)` SFs), we
    repeat the SF on the M-axis to per-row before packing — the `(INT, 1, gran_k)`
    DeepGEMM kernel branch is the only UE8M0 path on SM100; for `gran_mn > 1`
    the kernel only handles FP32 SFs and would otherwise reject our INT SF here.
    """
    is_sm100 = _is_sm100(sf.device)
    if sf.dtype == torch.float8_e8m0fnu:
        if expected_mn is not None and sf.size(-2) < expected_mn:
            gran_mn = expected_mn // sf.size(-2)
            sf = sf.repeat_interleave(gran_mn, dim=-2)
        if is_sm100:
            sf = sf.contiguous().view(torch.int32)
        else:
            sf = sf.float()
    elif sf.dtype == torch.float32 and is_sm100:
        sf = _ceil_to_ue8m0(sf)

    if sf.dim() not in (2, 3):
        raise ValueError(f"DeepGEMM SF must be 2D or 3D, got {sf.dim()}D")

    if not is_sm100:
        return sf.contiguous()

    mn = sf.size(-2)
    kf = sf.size(-1)
    align_to = 16 // sf.element_size()  # `get_tma_aligned_size`: align(mn, 16 / element_size)
    aligned_mn = -(-mn // align_to) * align_to
    target_strides = (1, aligned_mn) if sf.dim() == 2 else (kf * aligned_mn, 1, aligned_mn)

    if tuple(sf.stride()) == target_strides:
        return sf
    out = torch.empty_strided(sf.shape, target_strides, dtype=sf.dtype, device=sf.device)
    out.copy_(sf)
    return out


def _select_fp8_cast_kwargs(
    weight: torch.Tensor, weight_scale_inv: torch.Tensor, block_size: tuple | None, is_sm100: bool
) -> dict:
    """Pick the `per_token_cast_to_fp8` kwargs from weight dtype + SF dtype + arch.

    Cases mirror the kernel's recipes:
      - FP4 weights (`int8`): gran_k=32 packed-UE8M0 SF. SM100+ only.
      - FP8 weights + UE8M0 SF on SM100: gran_k=128 packed-UE8M0 SF (DSv4).
      - FP8 weights + UE8M0 SF on SM90: gran_k=128 FP32 SF — the SM90 dispatch in
        `layout.hpp` only matches FP32 SFs, so we keep act SFs as FP32 (and float
        the weight SF in `_coerce_sf_for_kernel`; UE8M0 → FP32 is an exact upcast).
      - FP8 weights + float SF: gran_k=128 float SF (DSv3).
    """
    if weight.dtype == torch.int8:  # FP4
        return {"use_ue8m0": True, "gran_k": 32, "use_packed_ue8m0": True}
    if block_size is None:
        raise ValueError(
            "DeepGEMM requires block-wise quantized FP8 weights, but the experts have no `block_size` set."
        )
    block_size = tuple(block_size)
    if block_size not in ((128, 128), (1, 128)):
        raise ValueError(f"DeepGEMM requires `block_size` ∈ {{(128, 128), (1, 128)}}, got {block_size}.")
    if weight_scale_inv.dtype == torch.float8_e8m0fnu and is_sm100:
        return {"use_ue8m0": True, "gran_k": 128, "use_packed_ue8m0": True}
    return {"use_ue8m0": False, "gran_k": 128}




def _build_deepgemm_contiguous_layout(
    expert_ids_sorted: torch.Tensor, num_experts: int, alignment: int, use_psum_layout: bool
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the TMA-aligned grouped layout DeepGEMM expects.

    Returns `(sorted_to_padded, grouped_layout, total_padded_rows)`:
      - `grouped_layout` is per-row expert id (Hopper, with `-1` for padding /
        sentinels) or a cumsum of aligned per-expert counts (Blackwell).
      - EP sentinels (values == `num_experts`) are routed past the last expert
        block so DeepGEMM skips them.
    """
    device = expert_ids_sorted.device
    num_tokens = expert_ids_sorted.size(0)
    tokens_per_expert = torch.histc(expert_ids_sorted.int(), bins=num_experts, min=0, max=num_experts - 1).long()
    aligned_tokens_per_expert = ((tokens_per_expert + alignment - 1) // alignment) * alignment
    total_padded_rows = num_tokens + min(num_tokens, num_experts) * (alignment - 1)

    padding_per_expert = aligned_tokens_per_expert - tokens_per_expert
    cumulative_padding = torch.nn.functional.pad(padding_per_expert.cumsum(0), (1, 0))
    sorted_to_padded = torch.arange(num_tokens, device=device) + cumulative_padding[expert_ids_sorted]

    if use_psum_layout:  # SM100+: kernel reads cumsum of aligned counts as expert boundaries.
        grouped_layout = aligned_tokens_per_expert.cumsum(0).int()
    else:  # SM90: per-row expert id, -1 = skip (padding & sentinels).
        grouped_layout = torch.full((total_padded_rows,), -1, device=device, dtype=torch.int32)
        grouped_layout[sorted_to_padded] = torch.where(expert_ids_sorted < num_experts, expert_ids_sorted.int(), -1)

    return sorted_to_padded, grouped_layout, total_padded_rows


def _pad_for_deepgemm(x: torch.Tensor, sorted_to_padded: torch.Tensor, total_padded_rows: int) -> torch.Tensor:
    """Pad a sorted tensor into the TMA-aligned contiguous layout."""
    padded = torch.empty(total_padded_rows, *x.shape[1:], device=x.device, dtype=x.dtype)
    padded[sorted_to_padded] = x
    return padded


def _unpad_from_deepgemm_contiguous_layout(x_padded: torch.Tensor, sorted_to_padded: torch.Tensor) -> torch.Tensor:
    return x_padded[sorted_to_padded]




def _dispatch_routed_input(
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_experts: int,
    m_alignment: int,
    use_psum_layout: bool,
) -> tuple:
    """Sort tokens by expert id and build the M-grouped padded layout.

    Returns `(sorted_hidden_states_g, sample_weights_g, expert_ids_g,
              sentinel_mask, perm, sorted_to_padded, grouped_layout,
              total_padded_rows)`.
    """
    num_top_k = top_k_index.size(-1)
    expert_ids = top_k_index.reshape(-1)  # (S,)
    sample_weights = top_k_weights.reshape(-1)  # (S,)

    expert_ids_g, perm = torch.sort(expert_ids)
    sorted_hidden_states_g = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    sorted_to_padded, grouped_layout, total_padded_rows = _build_deepgemm_contiguous_layout(
        expert_ids_g, num_experts, m_alignment, use_psum_layout
    )

    sentinel_mask = (expert_ids_g >= num_experts).unsqueeze(-1)
    expert_ids_g.clamp_(max=num_experts - 1)
    return (
        sorted_hidden_states_g,
        sample_weights_g,
        expert_ids_g,
        sentinel_mask,
        perm,
        sorted_to_padded,
        grouped_layout,
        total_padded_rows,
    )


def _combine_routed_output(
    out_padded: torch.Tensor,
    sorted_weights: torch.Tensor,
    sentinel_mask: torch.Tensor,
    perm: torch.Tensor,
    sorted_to_padded: torch.Tensor,
    num_tokens: int,
    num_top_k: int,
    hidden_dim: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Unpad → weighted multiply → mask sentinels → restore order → top-k reduce."""
    out = _unpad_from_deepgemm_contiguous_layout(out_padded, sorted_to_padded)
    weighted = out * sorted_weights.to(out.dtype).unsqueeze(-1)
    weighted.masked_fill_(sentinel_mask, 0.0)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=out.device)
    return weighted[inv_perm].view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(out_dtype)




@deprecate_kwarg("output_dtype", version="v5.16")
def deepgemm_fp8_fp4_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    block_size: tuple[int, int] | None = None,
    output_dtype: torch.dtype | None = None,
    activation_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """End-to-end DeepGEMM linear: per-token activation quant + FP8/FP4 matmul.

    Static (per-tensor) activation quantization is rejected — DeepGEMM needs
    per-row SFs. Callers should route static activations through the Triton fallback.
    """
    if activation_scale is not None:
        raise NotImplementedError("DeepGEMM linear does not support static activation quantization.")
    if input.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"DeepGEMM linear requires FP16 or BF16 activations, got {input.dtype}")

    deepgemm = load_deepgemm_kernel(requires_sm100=weight.dtype == torch.int8)
    cast_kwargs = _select_fp8_cast_kwargs(weight, weight_scale_inv, block_size, _is_sm100(input.device))

    input_2d = input.view(-1, input.shape[-1])
    qinput_2d, scale_2d = deepgemm.per_token_cast_to_fp8(input_2d, **cast_kwargs)
    output = torch.empty(qinput_2d.shape[0], weight.shape[0], device=input.device, dtype=input.dtype)

    sf_recipe = (1, 1, cast_kwargs["gran_k"]) if cast_kwargs.get("use_packed_ue8m0") else None
    deepgemm.fp8_fp4_matmul(
        (qinput_2d, _coerce_sf_for_kernel(scale_2d, expected_mn=qinput_2d.size(0))),
        (weight, _coerce_sf_for_kernel(weight_scale_inv, expected_mn=weight.size(0))),
        output,
        recipe=sf_recipe,
    )
    output = output.view(input.shape[:-1] + (weight.shape[0],))
    if bias is not None:
        output.add_(bias)
    return output


def deepgemm_bf16_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError(f"DeepGEMM experts path requires bfloat16 hidden states, got {hidden_states.dtype}")

    deepgemm = load_deepgemm_kernel()
    grouped_bf16_matmul = deepgemm.grouped_bf16_matmul_nn if self.is_transposed else deepgemm.grouped_bf16_matmul_nt

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)

    (
        sorted_hidden,
        sorted_weights,
        expert_ids_g,
        sentinel_mask,
        perm,
        sorted_to_padded,
        grouped_layout,
        total_padded_rows,
    ) = _dispatch_routed_input(
        hidden_states, top_k_index, top_k_weights, self.num_experts, deepgemm.m_alignment, _is_sm100(device)
    )

    weight_up = to_local(self.gate_up_proj if self.has_gate else self.up_proj)
    weight_down = to_local(self.down_proj)
    up_bias = to_local(self.gate_up_proj_bias if self.has_gate else self.up_proj_bias) if self.has_bias else None
    down_bias = to_local(self.down_proj_bias) if self.has_bias else None

    up_out_dim = weight_up.shape[-1] if self.is_transposed else weight_up.shape[1]
    act = _pad_for_deepgemm(sorted_hidden, sorted_to_padded, total_padded_rows)
    proj_out = torch.empty(total_padded_rows, up_out_dim, device=device, dtype=hidden_states.dtype)
    grouped_bf16_matmul(act, weight_up, proj_out, grouped_layout, use_psum_layout=_is_sm100(device))
    if self.has_bias:
        proj_out.index_add_(0, sorted_to_padded, up_bias[expert_ids_g])

    proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)

    out = torch.empty(total_padded_rows, hidden_dim, device=device, dtype=hidden_states.dtype)
    grouped_bf16_matmul(proj_out, weight_down, out, grouped_layout, use_psum_layout=_is_sm100(device))
    if self.has_bias:
        out.index_add_(0, sorted_to_padded, down_bias[expert_ids_g])

    return _combine_routed_output(
        out,
        sorted_weights,
        sentinel_mask,
        perm,
        sorted_to_padded,
        num_tokens,
        num_top_k,
        hidden_dim,
        hidden_states.dtype,
    )


def deepgemm_fp8_fp4_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    pass


def setup_megamoe_weights(module: torch.nn.Module) -> None:
    pass


def deepgemm_fp8_fp4_megamoe_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    process_group: torch.distributed.ProcessGroup | None = None,
) -> torch.Tensor:
    pass
