
import ast
import collections
import contextlib
import copy
import doctest
import functools
import gc
import importlib
import inspect
import json
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from collections import UserDict, defaultdict
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import MISSING, fields
from functools import cache, wraps
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest import mock
from unittest.mock import patch

import httpx
from huggingface_hub import create_repo, delete_repo
from packaging import version

from transformers import logging as transformers_logging


if TYPE_CHECKING:
    from .trainer import Trainer
else:
    Trainer = Any  # type: ignore

from .integrations import (
    is_clearml_available,
    is_optuna_available,
    is_ray_available,
    is_swanlab_available,
    is_tensorboard_available,
    is_trackio_available,
    is_wandb_available,
)
from .integrations.deepspeed import is_deepspeed_available
from .utils import (
    ACCELERATE_MIN_VERSION,
    GGUF_MIN_VERSION,
    MISTRAL_COMMON_MIN_VERSION,
    SAFE_WEIGHTS_INDEX_NAME,
    TRITON_MIN_VERSION,
    WEIGHTS_INDEX_NAME,
    is_accelerate_available,
    is_apex_available,
    is_apollo_torch_available,
    is_aqlm_available,
    is_auto_round_available,
    is_av_available,
    is_bitsandbytes_available,
    is_bs4_available,
    is_causal_conv1d_available,
    is_compressed_tensors_available,
    is_cv2_available,
    is_cython_available,
    is_decord_available,
    is_detectron2_available,
    is_essentia_available,
    is_executorch_available,
    is_faiss_available,
    is_fbgemm_gpu_available,
    is_flash_attn_2_available,
    is_flash_attn_3_available,
    is_flash_attn_4_available,
    is_flash_linear_attention_available,
    is_flute_available,
    is_fouroversix_available,
    is_fp_quant_available,
    is_fsdp_available,
    is_g2p_en_available,
    is_galore_torch_available,
    is_gguf_available,
    is_gptqmodel_available,
    is_grokadamw_available,
    is_hadamard_available,
    is_hqq_available,
    is_huggingface_hub_greater_or_equal,
    is_ipython_available,
    is_jinja_available,
    is_jmespath_available,
    is_jumanpp_available,
    is_kernels_available,
    is_levenshtein_available,
    is_librosa_available,
    is_liger_kernel_available,
    is_lomo_available,
    is_mistral_common_available,
    is_multipart_available,
    is_natten_available,
    is_nltk_available,
    is_numba_available,
    is_onnx_available,
    is_onnxruntime_available,
    is_onnxscript_available,
    is_openai_available,
    is_optimum_available,
    is_optimum_quanto_available,
    is_pandas_available,
    is_peft_available,
    is_peft_greater_or_equal,
    is_phonemizer_available,
    is_pretty_midi_available,
    is_psutil_available,
    is_pyctcdecode_available,
    is_pytesseract_available,
    is_pytest_available,
    is_pytest_order_available,
    is_pytorch_quantization_available,
    is_quark_available,
    is_qutlass_available,
    is_rjieba_available,
    is_sacremoses_available,
    is_schedulefree_available,
    is_scipy_available,
    is_sentencepiece_available,
    is_seqio_available,
    is_serve_available,
    is_soundfile_available,
    is_spacy_available,
    is_speech_available,
    is_spqr_available,
    is_sudachi_available,
    is_sudachi_projection_available,
    is_tiktoken_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_bf16_available_on_device,
    is_torch_fp16_available_on_device,
    is_torch_greater_or_equal,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_neuroncore_available,
    is_torch_npu_available,
    is_torch_optimi_available,
    is_torch_tensorrt_fx_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    is_torchao_available,
    is_torchaudio_available,
    is_torchcodec_available,
    is_torchvision_available,
    is_triton_available,
    is_vision_available,
    is_vptq_available,
    strtobool,
)


if is_accelerate_available():
    from accelerate.state import AcceleratorState, PartialState
    from accelerate.utils.imports import is_fp8_available


if is_pytest_available():
    from _pytest.doctest import (
        Module,
        _get_checker,
        _get_continue_on_failure,
        _get_runner,
        _is_mocked,
        _patch_unwrap_mock_aware,
        get_optionflags,
    )
    from _pytest.outcomes import skip
    from _pytest.pathlib import import_path
    from pytest import DoctestItem
else:
    Module = object
    DoctestItem = object


SMALL_MODEL_IDENTIFIER = "julien-c/bert-xsmall-dummy"
DUMMY_UNKNOWN_IDENTIFIER = "julien-c/dummy-unknown"
DUMMY_DIFF_TOKENIZER_IDENTIFIER = "julien-c/dummy-diff-tokenizer"

USER = "__DUMMY_TRANSFORMERS_USER__"
ENDPOINT_STAGING = "https://hub-ci.huggingface.co"

TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"


_COMMON_MODEL_NAMES_MAP = {
    "config_class": "Config",
    "causal_lm_class": "ForCausalLM",
    "question_answering_class": "ForQuestionAnswering",
    "sequence_classification_class": "ForSequenceClassification",
    "token_classification_class": "ForTokenClassification",
}

_VLM_COMMON_MODEL_NAMES_MAP = {
    "config_class": "Config",
    "text_config_class": "TextConfig",
    "vision_config_class": "VisionConfig",
    "conditional_generation_class": "ForConditionalGeneration",
}

_TEXT_MODEL_TESTER_DEFAULTS = {
    "batch_size": 13,
    "seq_length": 7,
    "is_training": True,
    "use_input_mask": True,
    "use_labels": True,
    "vocab_size": 99,
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "intermediate_size": 32,
    "hidden_act": "gelu",
    "max_position_embeddings": 512,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "expert_interval": 1,
    "moe_layer_start_index": 0,
    "moe_intermediate_size": 16,
    "shared_expert_intermediate_size": 36,
    "shared_expert_gate": True,
    "moe_num_shared_experts": 2,
    "num_experts_per_tok": 2,
    "num_experts": 8,
}


if is_torch_available():
    import torch
    from safetensors.torch import load_file

    from .modeling_utils import PreTrainedModel

    IS_ROCM_SYSTEM = torch.version.hip is not None
    IS_CUDA_SYSTEM = torch.version.cuda is not None
    IS_XPU_SYSTEM = getattr(torch.version, "xpu", None) is not None
    IS_NPU_SYSTEM = getattr(torch, "npu", None) is not None
else:
    IS_ROCM_SYSTEM = False
    IS_CUDA_SYSTEM = False
    IS_XPU_SYSTEM = False
    IS_NPU_SYSTEM = False

logger = transformers_logging.get_logger(__name__)


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = strtobool(value)
        except ValueError:
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


def parse_int_from_env(key, default=None):
    pass


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
_run_flaky_tests = parse_flag_from_env("RUN_FLAKY", default=True)
_run_custom_tokenizers = parse_flag_from_env("RUN_CUSTOM_TOKENIZERS", default=False)
_run_staging = parse_flag_from_env("HUGGINGFACE_CO_STAGING", default=False)
_run_pipeline_tests = parse_flag_from_env("RUN_PIPELINE_TESTS", default=True)
_run_agent_tests = parse_flag_from_env("RUN_AGENT_TESTS", default=False)
_run_training_tests = parse_flag_from_env("RUN_TRAINING_TESTS", default=True)
_run_tensor_parallel_tests = parse_flag_from_env("RUN_TENSOR_PARALLEL_TESTS", default=True)


def is_staging_test(test_case):
    """
    Decorator marking a test as a staging test.

    Those tests will run using the staging environment of huggingface.co instead of the real model hub.
    """
    if not _run_staging:
        return unittest.skip(reason="test is staging test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_staging_test()(test_case)


def is_pipeline_test(test_case):
    """
    Decorator marking a test as a pipeline test. If RUN_PIPELINE_TESTS is set to a falsy value, those tests will be
    skipped.
    """
    if not _run_pipeline_tests:
        return unittest.skip(reason="test is pipeline test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_pipeline_test()(test_case)


def is_agent_test(test_case):
    pass


def is_training_test(test_case):
    """
    Decorator marking a test as a training test. If RUN_TRAINING_TESTS is set to a falsy value, those tests will be
    skipped.
    """
    if not _run_training_tests:
        return unittest.skip(reason="test is training test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_training_test()(test_case)


def is_tensor_parallel_test(test_case):
    """
    Decorator marking a test as a tensor parallel test. If RUN_TENSOR_PARALLEL_TESTS is set to a falsy value, those
    tests will be skipped.
    """
    if not _run_tensor_parallel_tests:
        return unittest.skip(reason="test is tensor parallel test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_tensor_parallel_test()(test_case)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


def tooslow(test_case):
    """
    Decorator marking a test as too slow.

    Slow tests are skipped while they're in the process of being fixed. No test should stay tagged as "tooslow" as
    these will not be tested by the CI.

    """
    return unittest.skip(reason="test is too slow")(test_case)


def skip_if_not_implemented(test_func):
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        try:
            return test_func(*args, **kwargs)
        except NotImplementedError as e:
            raise unittest.SkipTest(f"Test skipped due to NotImplementedError: {e}")

    return wrapper


def apply_skip_if_not_implemented(cls):
    """
    Class decorator to apply @skip_if_not_implemented to all test methods.
    """
    for attr_name in dir(cls):
        if attr_name.startswith("test_"):
            attr = getattr(cls, attr_name)
            if callable(attr):
                setattr(cls, attr_name, skip_if_not_implemented(attr))
    return cls


def custom_tokenizers(test_case):
    """
    Decorator marking a test for a custom tokenizer.

    Custom tokenizers require additional dependencies, and are skipped by default. Set the RUN_CUSTOM_TOKENIZERS
    environment variable to a truthy value to run them.
    """
    return unittest.skipUnless(_run_custom_tokenizers, "test of custom tokenizers")(test_case)


def require_bs4(test_case):
    """
    Decorator marking a test that requires BeautifulSoup4. These tests are skipped when BeautifulSoup4 isn't installed.
    """
    return unittest.skipUnless(is_bs4_available(), "test requires BeautifulSoup4")(test_case)


def require_galore_torch(test_case):
    """
    Decorator marking a test that requires GaLore. These tests are skipped when GaLore isn't installed.
    https://github.com/jiaweizzhao/GaLore
    """
    return unittest.skipUnless(is_galore_torch_available(), "test requires GaLore")(test_case)


def require_apollo_torch(test_case):
    """
    Decorator marking a test that requires GaLore. These tests are skipped when APOLLO isn't installed.
    https://github.com/zhuhanqing/APOLLO
    """
    return unittest.skipUnless(is_apollo_torch_available(), "test requires APOLLO")(test_case)


def require_torch_optimi(test_case):
    """
    Decorator marking a test that requires torch-optimi. These tests are skipped when torch-optimi isn't installed.
    https://github.com/jxnl/torch-optimi
    """
    return unittest.skipUnless(is_torch_optimi_available(), "test requires torch-optimi")(test_case)


def require_lomo(test_case):
    """
    Decorator marking a test that requires LOMO. These tests are skipped when LOMO-optim isn't installed.
    https://github.com/OpenLMLab/LOMO
    """
    return unittest.skipUnless(is_lomo_available(), "test requires LOMO")(test_case)


def require_grokadamw(test_case):
    """
    Decorator marking a test that requires GrokAdamW. These tests are skipped when GrokAdamW isn't installed.
    """
    return unittest.skipUnless(is_grokadamw_available(), "test requires GrokAdamW")(test_case)


def require_schedulefree(test_case):
    """
    Decorator marking a test that requires schedulefree. These tests are skipped when schedulefree isn't installed.
    https://github.com/facebookresearch/schedule_free
    """
    return unittest.skipUnless(is_schedulefree_available(), "test requires schedulefree")(test_case)


def require_cv2(test_case):
    """
    Decorator marking a test that requires OpenCV.

    These tests are skipped when OpenCV isn't installed.

    """
    return unittest.skipUnless(is_cv2_available(), "test requires OpenCV")(test_case)


def require_levenshtein(test_case):
    """
    Decorator marking a test that requires Levenshtein.

    These tests are skipped when Levenshtein isn't installed.

    """
    return unittest.skipUnless(is_levenshtein_available(), "test requires Levenshtein")(test_case)


def require_nltk(test_case):
    """
    Decorator marking a test that requires NLTK.

    These tests are skipped when NLTK isn't installed.

    """
    return unittest.skipUnless(is_nltk_available(), "test requires NLTK")(test_case)


def require_accelerate(test_case, min_version: str = ACCELERATE_MIN_VERSION):
    """
    Decorator marking a test that requires accelerate. These tests are skipped when accelerate isn't installed.
    """
    return unittest.skipUnless(
        is_accelerate_available(min_version), f"test requires accelerate version >= {min_version}"
    )(test_case)


def require_triton(min_version: str = TRITON_MIN_VERSION):
    """
    Decorator marking a test that requires triton. These tests are skipped when triton isn't installed.
    """

    def decorator(test_case):
        pass

    return decorator


def require_gguf(test_case, min_version: str = GGUF_MIN_VERSION):
    """
    Decorator marking a test that requires ggguf. These tests are skipped when gguf isn't installed.
    """
    return unittest.skipUnless(is_gguf_available(min_version), f"test requires gguf version >= {min_version}")(
        test_case
    )


def require_fsdp(test_case, min_version: str = "1.12.0"):
    pass


def require_g2p_en(test_case):
    """
    Decorator marking a test that requires g2p_en. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_g2p_en_available(), "test requires g2p_en")(test_case)


def require_rjieba(test_case):
    """
    Decorator marking a test that requires rjieba. These tests are skipped when rjieba isn't installed.
    """
    return unittest.skipUnless(is_rjieba_available(), "test requires rjieba")(test_case)


def require_jinja(test_case):
    """
    Decorator marking a test that requires jinja. These tests are skipped when jinja isn't installed.
    """
    return unittest.skipUnless(is_jinja_available(), "test requires jinja")(test_case)


def require_jmespath(test_case):
    """
    Decorator marking a test that requires jmespath. These tests are skipped when jmespath isn't installed.
    """
    return unittest.skipUnless(is_jmespath_available(), "test requires jmespath")(test_case)


def require_onnx(test_case):
    return unittest.skipUnless(is_onnx_available(), "test requires ONNX")(test_case)


def require_onnxscript(test_case):
    return unittest.skipUnless(is_onnxscript_available(), "test requires ONNXScript")(test_case)


def require_onnxruntime(test_case):
    return unittest.skipUnless(is_onnxruntime_available(), "test requires ONNX Runtime")(test_case)


def require_executorch(test_case):
    return unittest.skipUnless(is_executorch_available(), "test requires ExecuTorch")(test_case)


def require_timm(test_case):
    """
    Decorator marking a test that requires Timm.

    These tests are skipped when Timm isn't installed.

    """
    return unittest.skipUnless(is_timm_available(), "test requires Timm")(test_case)


def require_natten(test_case):
    """
    Decorator marking a test that requires NATTEN.

    These tests are skipped when NATTEN isn't installed.

    """
    return unittest.skipUnless(is_natten_available(), "test requires natten")(test_case)


def require_torch(test_case):
    """
    Decorator marking a test that requires PyTorch.

    These tests are skipped when PyTorch isn't installed.

    """
    return unittest.skipUnless(is_torch_available(), "test requires PyTorch")(test_case)


def require_torch_greater_or_equal(version: str):
    """
    Decorator marking a test that requires PyTorch version >= `version`.

    These tests are skipped when PyTorch version is less than `version`.
    """

    def decorator(test_case):
        pass

    return decorator


def require_huggingface_hub_greater_or_equal(version: str):
    pass


def require_flash_attn(test_case):
    """
    Decorator marking a test that requires Flash Attention.

    These tests are skipped when Flash Attention isn't installed.

    """
    flash_attn_available = is_flash_attn_2_available(kernels_fallback_ok=True)
    return unittest.skipUnless(flash_attn_available, "test requires Flash Attention")(test_case)


def require_kernels(test_case):
    """
    Decorator marking a test that requires the kernels library.

    These tests are skipped when the kernels library isn't installed.

    """
    return unittest.skipUnless(is_kernels_available(), "test requires the kernels library")(test_case)


def require_flash_attn_3(test_case):
    """
    Decorator marking a test that requires Flash Attention 3.

    These tests are skipped when Flash Attention 3 isn't installed.
    """
    return unittest.skipUnless(is_flash_attn_3_available(), "test requires Flash Attention 3")(test_case)


def require_flash_attn_4(test_case):
    """
    Decorator marking a test that requires Flash Attention 4.

    These tests are skipped when Flash Attention 4 isn't installed.
    """
    return unittest.skipUnless(is_flash_attn_4_available(), "test requires Flash Attention 4")(test_case)


def require_all_flash_attn(test_case):
    flash_attn_available = is_flash_attn_2_available(kernels_fallback_ok=True)

    return unittest.skipUnless(
        all(
            (
                flash_attn_available,
                is_flash_attn_3_available(),
                is_flash_attn_4_available(),
            )
        ),
        "test requires all mainline Flash Attention packages",
    )(test_case)


def require_flash_linear_attention(test_case):
    """
    Decorator marking a test that requires Flash Linear Attention.

    These tests are skipped when Flash Linear Attention isn't installed.
    """

    return unittest.skipUnless(
        is_flash_linear_attention_available(),
        "test requires `flash-linear-attention`",
    )(test_case)


def require_causal_conv1d(test_case):
    """
    Decorator marking a test that requires causal-conv1d.

    These tests are skipped when causal-conv1d isn't installed.
    """

    return unittest.skipUnless(
        is_causal_conv1d_available(),
        "test requires `causal-conv1d`",
    )(test_case)


def require_peft(test_case):
    """
    Decorator marking a test that requires PEFT.

    These tests are skipped when PEFT isn't installed.

    """
    return unittest.skipUnless(is_peft_available(), "test requires PEFT")(test_case)


def require_peft_greater_or_equal(version: str):
    """
    Decorator marking a test that requires PEFT version >= `version`.

    These tests are skipped when PEFT version is less than `version`.
    """

    def decorator(test_case):
        pass

    return decorator


def require_torchvision(test_case):
    """
    Decorator marking a test that requires Torchvision.

    These tests are skipped when Torchvision isn't installed.

    """
    return unittest.skipUnless(is_torchvision_available(), "test requires Torchvision")(test_case)


def require_torchcodec(test_case):
    """
    Decorator marking a test that requires Torchcodec.

    These tests are skipped when Torchcodec isn't installed.

    """
    return unittest.skipUnless(is_torchcodec_available(), "test requires Torchcodec")(test_case)


def require_torchaudio(test_case):
    """
    Decorator marking a test that requires torchaudio. These tests are skipped when torchaudio isn't installed.
    """
    return unittest.skipUnless(is_torchaudio_available(), "test requires torchaudio")(test_case)


def require_sentencepiece(test_case):
    """
    Decorator marking a test that requires SentencePiece. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_sentencepiece_available(), "test requires SentencePiece")(test_case)


def require_sacremoses(test_case):
    """
    Decorator marking a test that requires Sacremoses. These tests are skipped when Sacremoses isn't installed.
    """
    return unittest.skipUnless(is_sacremoses_available(), "test requires Sacremoses")(test_case)


def require_seqio(test_case):
    pass


def require_scipy(test_case):
    """
    Decorator marking a test that requires Scipy. These tests are skipped when SentencePiece isn't installed.
    """
    return unittest.skipUnless(is_scipy_available(), "test requires Scipy")(test_case)


def require_tokenizers(test_case):
    """
    Decorator marking a test that requires 🤗 Tokenizers. These tests are skipped when 🤗 Tokenizers isn't installed.
    """
    return unittest.skipUnless(is_tokenizers_available(), "test requires tokenizers")(test_case)


def require_pandas(test_case):
    """
    Decorator marking a test that requires pandas. These tests are skipped when pandas isn't installed.
    """
    return unittest.skipUnless(is_pandas_available(), "test requires pandas")(test_case)


def require_pytesseract(test_case):
    """
    Decorator marking a test that requires PyTesseract. These tests are skipped when PyTesseract isn't installed.
    """
    return unittest.skipUnless(is_pytesseract_available(), "test requires PyTesseract")(test_case)


def require_pytorch_quantization(test_case):
    pass


def require_vision(test_case):
    """
    Decorator marking a test that requires the vision dependencies. These tests are skipped when torchaudio isn't
    installed.
    """
    return unittest.skipUnless(is_vision_available(), "test requires vision")(test_case)


def require_spacy(test_case):
    pass


def require_torch_multi_gpu(test_case):
    """
    Decorator marking a test that requires a multi-GPU CUDA setup (in PyTorch). These tests are skipped on a machine without
    multiple CUDA GPUs.

    To run *only* the multi_gpu tests, assuming all test names contain multi_gpu: $ pytest -sv ./tests -k "multi_gpu"
    """
    if not is_torch_available():
        return unittest.skip(reason="test requires PyTorch")(test_case)

    import torch

    return unittest.skipUnless(torch.cuda.device_count() > 1, "test requires multiple CUDA GPUs")(test_case)


def require_torch_multi_accelerator(test_case):
    """
    Decorator marking a test that requires a multi-accelerator (in PyTorch). These tests are skipped on a machine
    without multiple accelerators. To run *only* the multi_accelerator tests, assuming all test names contain
    multi_accelerator: $ pytest -sv ./tests -k "multi_accelerator"
    """
    if not is_torch_available():
        return unittest.skip(reason="test requires PyTorch")(test_case)

    return unittest.skipUnless(backend_device_count(torch_device) > 1, "test requires multiple accelerators")(
        test_case
    )


def require_torch_n_accelerators(n: int):
    """Decorator marking a test that requires at least `n` accelerators (in PyTorch)."""

    def decorator(test_case):
        pass

    return decorator


def require_torch_non_multi_gpu(test_case):
    pass


def require_torch_non_multi_accelerator(test_case):
    """
    Decorator marking a test that requires 0 or 1 accelerator setup (in PyTorch).
    """
    if not is_torch_available():
        return unittest.skip(reason="test requires PyTorch")(test_case)

    return unittest.skipUnless(backend_device_count(torch_device) < 2, "test requires 0 or 1 accelerator")(test_case)


def require_torch_up_to_2_gpus(test_case):
    pass


def require_torch_up_to_2_accelerators(test_case):
    """
    Decorator marking a test that requires 0 or 1 or 2 accelerator setup (in PyTorch).
    """
    if not is_torch_available():
        return unittest.skip(reason="test requires PyTorch")(test_case)

    return unittest.skipUnless(backend_device_count(torch_device) < 3, "test requires 0 or 1 or 2 accelerators")(
        test_case
    )


def require_torch_xla(test_case):
    pass


def require_torch_neuroncore(test_case):
    pass


def require_torch_tpu(test_case):
    pass


def require_torch_npu(test_case):
    pass


def require_torch_multi_npu(test_case):
    pass


def require_non_hpu(test_case):
    """
    Decorator marking a test that should be skipped for HPU.
    """
    return unittest.skipUnless(torch_device != "hpu", "test requires a non-HPU")(test_case)


def require_torch_xpu(test_case):
    pass


def require_non_xpu(test_case):
    """
    Decorator marking a test that should be skipped for XPU.
    """
    return unittest.skipUnless(torch_device != "xpu", "test requires a non-XPU")(test_case)


def require_torch_multi_xpu(test_case):
    pass


def require_torch_multi_hpu(test_case):
    pass


if is_torch_available():
    import torch

    if "TRANSFORMERS_TEST_BACKEND" in os.environ:
        backend = os.environ["TRANSFORMERS_TEST_BACKEND"]
        try:
            _ = importlib.import_module(backend)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Failed to import `TRANSFORMERS_TEST_BACKEND` '{backend}'! This should be the name of an installed module. The original error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    if "TRANSFORMERS_TEST_DEVICE" in os.environ:
        torch_device = os.environ["TRANSFORMERS_TEST_DEVICE"]
        if torch_device == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                f"TRANSFORMERS_TEST_DEVICE={torch_device}, but CUDA is unavailable. Please double-check your testing environment."
            )
        if torch_device == "xpu" and not is_torch_xpu_available():
            raise ValueError(
                f"TRANSFORMERS_TEST_DEVICE={torch_device}, but XPU is unavailable. Please double-check your testing environment."
            )
        if torch_device == "npu" and not is_torch_npu_available():
            raise ValueError(
                f"TRANSFORMERS_TEST_DEVICE={torch_device}, but NPU is unavailable. Please double-check your testing environment."
            )
        if torch_device == "mlu" and not is_torch_mlu_available():
            raise ValueError(
                f"TRANSFORMERS_TEST_DEVICE={torch_device}, but MLU is unavailable. Please double-check your testing environment."
            )
        if torch_device == "hpu" and not is_torch_hpu_available():
            raise ValueError(
                f"TRANSFORMERS_TEST_DEVICE={torch_device}, but HPU is unavailable. Please double-check your testing environment."
            )

        try:
            _ = torch.device(torch_device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Unknown testing device specified by environment variable `TRANSFORMERS_TEST_DEVICE`: {torch_device}"
            ) from e
    elif torch.cuda.is_available():
        torch_device = "cuda"
    elif is_torch_npu_available():
        torch_device = "npu"
    elif is_torch_mlu_available():
        torch_device = "mlu"
    elif is_torch_hpu_available():
        torch_device = "hpu"
    elif is_torch_xpu_available():
        torch_device = "xpu"
    else:
        torch_device = "cpu"
else:
    torch_device = None


def require_torchao(test_case):
    """Decorator marking a test that requires torchao"""
    return unittest.skipUnless(is_torchao_available(), "test requires torchao")(test_case)


def require_torchao_version_greater_or_equal(torchao_version):
    pass


def require_torch_tensorrt_fx(test_case):
    pass


def require_torch_gpu(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(torch_device == "cuda", "test requires CUDA")(test_case)


def require_torch_mps(test_case):
    """Decorator marking a test that requires CUDA and PyTorch."""
    return unittest.skipUnless(torch_device == "mps", "test requires MPS")(test_case)


def require_rocm(test_case):
    """Decorator marking a test that requires a ROCm (AMD) GPU and PyTorch."""
    return unittest.skipUnless(torch_device == "cuda" and IS_ROCM_SYSTEM, "test requires a ROCm (AMD) GPU")(test_case)


def require_large_cpu_ram(test_case, memory: float = 80):
    """Decorator marking a test that requires a CPU RAM with more than `memory` GiB of memory."""
    if not is_psutil_available():
        return test_case

    import psutil

    return unittest.skipUnless(
        psutil.virtual_memory().total / 1024**3 > memory,
        f"test requires a machine with more than {memory} GiB of CPU RAM memory",
    )(test_case)


def require_torch_large_gpu(test_case, memory: float = 20):
    """Decorator marking a test that requires a CUDA GPU with more than `memory` GiB of memory."""
    if torch_device != "cuda":
        return unittest.skip(reason=f"test requires a CUDA GPU with more than {memory} GiB of memory")(test_case)

    return unittest.skipUnless(
        torch.cuda.get_device_properties(0).total_memory / 1024**3 > memory,
        f"test requires a GPU with more than {memory} GiB of memory",
    )(test_case)


def require_torch_large_accelerator(test_case=None, *, memory: float = 20):
    """Decorator marking a test that requires an accelerator with more than `memory` GiB of memory."""

    def memory_decorator(tc):
        if torch_device not in ("cuda", "xpu"):
            return unittest.skip(f"test requires a GPU or XPU with more than {memory} GiB of memory")(tc)

        torch_accel = getattr(torch, torch_device)
        return unittest.skipUnless(
            torch_accel.get_device_properties(0).total_memory / 1024**3 > memory,
            f"test requires a GPU or XPU with more than {memory} GiB of memory",
        )(tc)

    return memory_decorator if test_case is None else memory_decorator(test_case)


def require_torch_accelerator(test_case):
    """Decorator marking a test that requires an accessible accelerator and PyTorch."""
    return unittest.skipUnless(torch_device is not None and torch_device != "cpu", "test requires accelerator")(
        test_case
    )


def require_torch_fp16(test_case):
    """Decorator marking a test that requires a device that supports fp16"""
    return unittest.skipUnless(
        is_torch_fp16_available_on_device(torch_device), "test requires device with fp16 support"
    )(test_case)


def require_fp8(test_case):
    pass


def require_cuda_capability_at_least(major, minor):
    """Decorator: when running on CUDA, skip if device capability is below the given
    threshold. On non-CUDA backends this is a no-op — pair with :func:`require_torch_gpu`
    if the test is CUDA-only, or leave alone if it should also run on other backends
    (which won't be capability-gated)."""
    import torch

    if not torch.cuda.is_available():
        return lambda test_case: test_case
    capability = torch.cuda.get_device_capability()
    return unittest.skipIf(capability < (major, minor), f"Requires CUDA capability >= {major}.{minor}")


def require_torch_bf16(test_case):
    """Decorator marking a test that requires a device that supports bf16"""
    return unittest.skipUnless(
        is_torch_bf16_available_on_device(torch_device), "test requires device with bf16 support"
    )(test_case)


def require_deterministic_for_xpu(test_case):
    @wraps(test_case)
    def wrapper(*args, **kwargs):
        if is_torch_xpu_available():
            original_state = torch.are_deterministic_algorithms_enabled()
            try:
                torch.use_deterministic_algorithms(True)
                return test_case(*args, **kwargs)
            finally:
                torch.use_deterministic_algorithms(original_state)
        else:
            return test_case(*args, **kwargs)

    return wrapper


def require_torch_tf32(test_case):
    """Decorator marking a test that requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7."""
    return unittest.skipUnless(
        is_torch_tf32_available(), "test requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7"
    )(test_case)


def require_detectron2(test_case):
    """Decorator marking a test that requires detectron2."""
    return unittest.skipUnless(is_detectron2_available(), "test requires `detectron2`")(test_case)


def require_faiss(test_case):
    """Decorator marking a test that requires faiss."""
    return unittest.skipUnless(is_faiss_available(), "test requires `faiss`")(test_case)


def require_ipython(test_case):
    """Decorator marking a test that requires IPython. These tests are skipped when IPython isn't installed."""
    return unittest.skipUnless(is_ipython_available(), "test requires `IPython`")(test_case)


def require_optuna(test_case):
    """
    Decorator marking a test that requires optuna.

    These tests are skipped when optuna isn't installed.

    """
    return unittest.skipUnless(is_optuna_available(), "test requires optuna")(test_case)


def require_ray(test_case):
    """
    Decorator marking a test that requires Ray/tune.

    These tests are skipped when Ray/tune isn't installed.

    """
    return unittest.skipUnless(is_ray_available(), "test requires Ray/tune")(test_case)


def require_swanlab(test_case):
    pass


def require_trackio(test_case):
    pass


def require_wandb(test_case):
    """
    Decorator marking a test that requires wandb.

    These tests are skipped when wandb isn't installed.

    """
    return unittest.skipUnless(is_wandb_available(), "test requires wandb")(test_case)


def require_clearml(test_case):
    pass


def require_deepspeed(test_case):
    """
    Decorator marking a test that requires deepspeed
    """
    return unittest.skipUnless(is_deepspeed_available(), "test requires deepspeed")(test_case)


def require_apex(test_case):
    pass


def require_aqlm(test_case):
    """
    Decorator marking a test that requires aqlm
    """
    return unittest.skipUnless(is_aqlm_available(), "test requires aqlm")(test_case)


def require_vptq(test_case):
    """
    Decorator marking a test that requires vptq
    """
    return unittest.skipUnless(is_vptq_available(), "test requires vptq")(test_case)


def require_spqr(test_case):
    """
    Decorator marking a test that requires spqr
    """
    return unittest.skipUnless(is_spqr_available(), "test requires spqr")(test_case)


def require_av(test_case):
    """
    Decorator marking a test that requires av
    """
    return unittest.skipUnless(is_av_available(), "test requires av")(test_case)


def require_decord(test_case):
    """
    Decorator marking a test that requires decord
    """
    return unittest.skipUnless(is_decord_available(), "test requires decord")(test_case)


def require_bitsandbytes(test_case):
    """
    Decorator marking a test that requires the bitsandbytes library. Will be skipped when the library or its hard dependency torch is not installed.
    """
    return unittest.skipUnless(is_bitsandbytes_available(), "test requires bitsandbytes")(test_case)


def require_optimum(test_case):
    """
    Decorator for optimum dependency
    """
    return unittest.skipUnless(is_optimum_available(), "test requires optimum")(test_case)


def require_tensorboard(test_case):
    """
    Decorator for `tensorboard` dependency
    """
    return unittest.skipUnless(is_tensorboard_available(), "test requires tensorboard")


def require_gptqmodel(test_case):
    """
    Decorator for gptqmodel dependency
    """
    return unittest.skipUnless(is_gptqmodel_available(), "test requires gptqmodel")(test_case)


def require_hqq(test_case):
    """
    Decorator for hqq dependency
    """
    return unittest.skipUnless(is_hqq_available(), "test requires hqq")(test_case)


def require_auto_round(test_case):
    """
    Decorator for auto_round dependency
    """
    return unittest.skipUnless(is_auto_round_available(), "test requires autoround")(test_case)


def require_optimum_quanto(test_case):
    """
    Decorator for quanto dependency
    """
    return unittest.skipUnless(is_optimum_quanto_available(), "test requires optimum-quanto")(test_case)


def require_compressed_tensors(test_case):
    """
    Decorator for compressed_tensors dependency
    """
    return unittest.skipUnless(is_compressed_tensors_available(), "test requires compressed_tensors")(test_case)


def require_fbgemm_gpu(test_case):
    pass


def require_quark(test_case):
    """
    Decorator for quark dependency
    """
    return unittest.skipUnless(is_quark_available(), "test requires quark")(test_case)


def require_flute_hadamard(test_case):
    """
    Decorator marking a test that requires higgs and hadamard
    """
    return unittest.skipUnless(
        is_flute_available() and is_hadamard_available(), "test requires flute and fast_hadamard_transform"
    )(test_case)


def require_fouroversix(test_case):
    """
    Decorator marking a test that requires fouroversix
    """
    return unittest.skipUnless(is_fouroversix_available(), "test requires fouroversix")(test_case)


def require_fp_quant(test_case):
    """
    Decorator marking a test that requires fp_quant and qutlass
    """
    return unittest.skipUnless(is_fp_quant_available(), "test requires fp_quant")(test_case)


def require_qutlass(test_case):
    """
    Decorator marking a test that requires qutlass
    """
    return unittest.skipUnless(is_qutlass_available(), "test requires qutlass")(test_case)


def require_phonemizer(test_case):
    """
    Decorator marking a test that requires phonemizer
    """
    return unittest.skipUnless(is_phonemizer_available(), "test requires phonemizer")(test_case)


def require_pyctcdecode(test_case):
    """
    Decorator marking a test that requires pyctcdecode
    """
    return unittest.skipUnless(is_pyctcdecode_available(), "test requires pyctcdecode")(test_case)


def require_numba(test_case):
    """
    Decorator marking a test that requires numba
    """
    return unittest.skipUnless(is_numba_available(), "test requires numba")(test_case)


def require_librosa(test_case):
    """
    Decorator marking a test that requires librosa
    """
    return unittest.skipUnless(is_librosa_available(), "test requires librosa")(test_case)


def require_soundfile(test_case):
    """
    Decorator marking a test that requires soundfile
    """
    return unittest.skipUnless(is_soundfile_available(), "test requires soundfile")(test_case)


def require_multipart(test_case):
    """
    Decorator marking a test that requires python-multipart
    """
    return unittest.skipUnless(is_multipart_available(), "test requires python-multipart")(test_case)


def require_liger_kernel(test_case):
    """
    Decorator marking a test that requires liger_kernel
    """
    return unittest.skipUnless(is_liger_kernel_available(), "test requires liger_kernel")(test_case)


def require_essentia(test_case):
    """
    Decorator marking a test that requires essentia
    """
    return unittest.skipUnless(is_essentia_available(), "test requires essentia")(test_case)


def require_pretty_midi(test_case):
    """
    Decorator marking a test that requires pretty_midi
    """
    return unittest.skipUnless(is_pretty_midi_available(), "test requires pretty_midi")(test_case)


def cmd_exists(cmd):
    pass


def require_usr_bin_time(test_case):
    pass


def require_sudachi(test_case):
    pass


def require_sudachi_projection(test_case):
    """
    Decorator marking a test that requires sudachi_projection
    """
    return unittest.skipUnless(is_sudachi_projection_available(), "test requires sudachi which supports projection")(
        test_case
    )


def require_jumanpp(test_case):
    """
    Decorator marking a test that requires jumanpp
    """
    return unittest.skipUnless(is_jumanpp_available(), "test requires jumanpp")(test_case)


def require_cython(test_case):
    pass


def require_tiktoken(test_case):
    """
    Decorator marking a test that requires TikToken. These tests are skipped when TikToken isn't installed.
    """
    return unittest.skipUnless(is_tiktoken_available(), "test requires TikToken")(test_case)


def require_speech(test_case):
    """
    Decorator marking a test that requires speech. These tests are skipped when speech isn't available.
    """
    return unittest.skipUnless(is_speech_available(), "test requires torchaudio")(test_case)


def require_openai(test_case):
    pass


def require_serve(test_case):
    """
    Decorator marking a test that requires the serving dependencies (fastapi, uvicorn, pydantic, openai).
    """
    return unittest.skipUnless(is_serve_available(), "test requires serving dependencies")(test_case)


def require_mistral_common(test_case, min_version: str = MISTRAL_COMMON_MIN_VERSION):
    """
    Decorator marking a test that requires mistral-common. These tests are skipped when mistral-common isn't available.
    """
    return unittest.skipUnless(
        is_mistral_common_available(min_version), f"test requires mistral-common version >= {min_version}"
    )(test_case)


def get_gpu_count():
    """
    Return the number of available gpus
    """
    if is_torch_available():
        import torch

        return torch.cuda.device_count()
    else:
        return 0


def get_tests_dir(append_path=None):
    """
    Args:
        append_path: optional path to append to the tests dir path

    Return:
        The full path to the `tests` dir, so that the tests can be invoked from anywhere. Optionally `append_path` is
        joined after the `tests` dir the former is provided.

    """
    caller__file__ = inspect.stack()[1][1]
    tests_dir = os.path.abspath(os.path.dirname(caller__file__))

    while not tests_dir.endswith("tests"):
        tests_dir = os.path.dirname(tests_dir)

    if append_path:
        return os.path.join(tests_dir, append_path)
    else:
        return tests_dir


def get_steps_per_epoch(trainer: Trainer) -> int:
    training_args = trainer.args
    train_dataloader = trainer.get_train_dataloader()

    initial_training_values = trainer.set_initial_training_values(args=training_args, dataloader=train_dataloader)
    steps_per_epoch = initial_training_values[5]

    return steps_per_epoch


def evaluate_side_effect_factory(
    side_effect_values: list[dict[str, float]],
) -> Generator[dict[str, float], None, None]:
    """
    Function that returns side effects for the _evaluate method.
    Used when we're unsure of exactly how many times _evaluate will be called.
    """
    yield from side_effect_values

    while True:
        yield side_effect_values[-1]




def apply_print_resets(buf):
    return re.sub(r"^.*\r", "", buf, 0, re.MULTILINE)


def assert_screenout(out, what):
    pass


def set_config_for_less_flaky_test(config):
    target_attrs = [
        "rms_norm_eps",
        "layer_norm_eps",
        "norm_eps",
        "norm_epsilon",
        "layer_norm_epsilon",
        "batch_norm_eps",
    ]
    for target_attr in target_attrs:
        setattr(config, target_attr, 1.0)

    attrs = ["text_config", "vision_config", "audio_config", "text_encoder", "audio_encoder", "decoder"]
    for attr in attrs:
        if hasattr(config, attr) and getattr(config, attr) is not None:
            for target_attr in target_attrs:
                setattr(getattr(config, attr), target_attr, 1.0)


def set_model_for_less_flaky_test(model):
    target_names = (
        "LayerNorm",
        "GroupNorm",
        "BatchNorm",
        "RMSNorm",
        "BatchNorm2d",
        "BatchNorm1d",
        "BitGroupNormActivation",
        "WeightStandardizedConv2d",
    )
    target_attrs = ["eps", "epsilon", "variance_epsilon"]
    if is_torch_available() and isinstance(model, torch.nn.Module):
        for module in model.modules():
            if type(module).__name__.endswith(target_names):
                for attr in target_attrs:
                    if hasattr(module, attr):
                        setattr(module, attr, 1.0)


class CaptureStd:

    def __init__(self, out=True, err=True, replay=True):
        self.replay = replay

        if out:
            self.out_buf = StringIO()
            self.out = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.out_buf = None
            self.out = "not capturing stdout"

        if err:
            self.err_buf = StringIO()
            self.err = "error: CaptureStd context is unfinished yet, called too early"
        else:
            self.err_buf = None
            self.err = "not capturing stderr"

    def __enter__(self):
        if self.out_buf:
            self.out_old = sys.stdout
            sys.stdout = self.out_buf

        if self.err_buf:
            self.err_old = sys.stderr
            sys.stderr = self.err_buf

        return self

    def __exit__(self, *exc):
        if self.out_buf:
            sys.stdout = self.out_old
            captured = self.out_buf.getvalue()
            if self.replay:
                sys.stdout.write(captured)
            self.out = apply_print_resets(captured)

        if self.err_buf:
            sys.stderr = self.err_old
            captured = self.err_buf.getvalue()
            if self.replay:
                sys.stderr.write(captured)
            self.err = captured

    def __repr__(self):
        msg = ""
        if self.out_buf:
            msg += f"stdout: {self.out}\n"
        if self.err_buf:
            msg += f"stderr: {self.err}\n"
        return msg




class CaptureStdout(CaptureStd):

    def __init__(self, replay=True):
        super().__init__(err=False, replay=replay)


class CaptureStderr(CaptureStd):

    def __init__(self, replay=True):
        super().__init__(out=False, replay=replay)


class CaptureLogger:

    def __init__(self, logger):
        self.logger = logger
        self.io = StringIO()
        self.sh = logging.StreamHandler(self.io)
        self.out = ""

    def __enter__(self):
        self.logger.addHandler(self.sh)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.sh)
        self.out = self.io.getvalue()

    def __repr__(self):
        return f"captured: {self.out}\n"


@contextlib.contextmanager
def LoggingLevel(level):
    """
    This is a context manager to temporarily change transformers modules logging level to the desired value and have it
    restored to the original setting at the end of the scope.

    Example:

    ```python
    with LoggingLevel(logging.INFO):
        AutoModel.from_pretrained("openai-community/gpt2")  # calls logger.info() several times
    ```
    """
    orig_level = transformers_logging.get_verbosity()
    try:
        transformers_logging.set_verbosity(level)
        yield
    finally:
        transformers_logging.set_verbosity(orig_level)


class TemporaryHubRepo:

    def __init__(self, namespace: str | None = None, token: str | None = None) -> None:
        self.token = token
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_id = Path(tmp_dir).name
            if namespace is not None:
                repo_id = f"{namespace}/{repo_id}"
            self.repo_url = create_repo(repo_id, token=self.token)

    def __enter__(self):
        return self.repo_url

    def __exit__(self, exc, value, tb):
        delete_repo(repo_id=self.repo_url.repo_id, token=self.token, missing_ok=True)


@contextlib.contextmanager
def ExtendSysPath(path: str | os.PathLike) -> Iterator[None]:
    """
    Temporary add given path to `sys.path`.

    Usage :

    ```python
    with ExtendSysPath("/path/to/dir"):
        mymodule = importlib.import_module("mymodule")
    ```
    """

    path = os.fspath(path)
    try:
        sys.path.insert(0, path)
        yield
    finally:
        sys.path.remove(path)


class TestCasePlus(unittest.TestCase):

    def setUp(self):
        pass

    @property
    def test_file_path(self):
        pass

    @property
    def test_file_path_str(self):
        pass

    @property
    def test_file_dir(self):
        pass

    @property
    def test_file_dir_str(self):
        pass

    @property
    def tests_dir(self):
        pass

    @property
    def tests_dir_str(self):
        pass

    @property
    def examples_dir(self):
        pass

    @property
    def examples_dir_str(self):
        pass

    @property
    def repo_root_dir(self):
        pass

    @property
    def repo_root_dir_str(self):
        pass

    @property
    def src_dir(self):
        pass

    @property
    def src_dir_str(self):
        pass

    def get_env(self):
        pass

    def get_auto_remove_tmp_dir(self, tmp_dir=None, before=None, after=None, return_pathlib_obj=False):
        pass

    def python_one_liner_max_rss(self, one_liner_str):
        pass

    def tearDown(self):
        pass


def mockenv(**kwargs):
    """
    this is a convenience wrapper, that allows this ::

    @mockenv(RUN_SLOW=True, USE_TF=False) def test_something():
        run_slow = os.getenv("RUN_SLOW", False) use_tf = os.getenv("USE_TF", False)

    """
    return mock.patch.dict(os.environ, kwargs)


@contextlib.contextmanager
def mockenv_context(*remove, **update):
    """
    Temporarily updates the `os.environ` dictionary in-place. Similar to mockenv

    The `os.environ` dictionary is updated in-place so that the modification is sure to work in all situations.

    Args:
      remove: Environment variables to remove.
      update: Dictionary of environment variables and values to add/update.
    """
    env = os.environ
    update = update or {}
    remove = remove or []

    stomped = (set(update.keys()) | set(remove)) & set(env.keys())
    update_after = {k: env[k] for k in stomped}
    remove_after = frozenset(k for k in update if k not in env)

    try:
        env.update(update)
        [env.pop(k, None) for k in remove]
        yield
    finally:
        env.update(update_after)
        [env.pop(k) for k in remove_after]



pytest_opt_registered = {}


def pytest_addoption_shared(parser):
    pass


def pytest_terminal_summary_main(tr, id):
    pass



import asyncio  # noqa


class _RunOutput:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


async def _read_stream(stream, callback):
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def _stream_subprocess(cmd, env=None, stdin=None, timeout=None, quiet=False, echo=False) -> _RunOutput:
    if echo:
        print("\nRunning: ", " ".join(cmd))

    p = await asyncio.create_subprocess_exec(
        cmd[0],
        *cmd[1:],
        stdin=stdin,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )


    out = []
    err = []

    def tee(line, sink, pipe, label=""):
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        if not quiet:
            print(label, line, file=pipe)

    await asyncio.wait(
        [
            asyncio.create_task(_read_stream(p.stdout, lambda l: tee(l, out, sys.stdout, label="stdout:"))),
            asyncio.create_task(_read_stream(p.stderr, lambda l: tee(l, err, sys.stderr, label="stderr:"))),
        ],
        timeout=timeout,
    )
    return _RunOutput(await p.wait(), out, err)


def execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True) -> _RunOutput:
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(
        _stream_subprocess(cmd, env=env, stdin=stdin, timeout=timeout, quiet=quiet, echo=echo)
    )

    cmd_str = " ".join(cmd)
    if result.returncode > 0:
        stderr = "\n".join(result.stderr)
        raise RuntimeError(
            f"'{cmd_str}' failed with returncode {result.returncode}\n\n"
            f"The combined stderr from workers follows:\n{stderr}"
        )

    if not result.stdout and not result.stderr:
        raise RuntimeError(f"'{cmd_str}' produced no output.")

    return result


def pytest_xdist_worker_id():
    pass


def get_torch_dist_unique_port():
    """
    Returns a free port number that can be fed to `torch.distributed.launch`'s `--master_port` argument.

    Binds to port 0 to let the OS assign an available port, avoiding collisions from hardcoded ports
    and TCP TIME_WAIT issues between sequential subprocess launches.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def nested_simplify(obj, decimals=3):
    """
    Simplifies an object by rounding float numbers, and downcasting tensors/numpy arrays to get simple equality test
    within tests.
    """
    import numpy as np

    if isinstance(obj, list):
        return [nested_simplify(item, decimals) for item in obj]
    if isinstance(obj, tuple):
        return tuple(nested_simplify(item, decimals) for item in obj)
    elif isinstance(obj, np.ndarray):
        return nested_simplify(obj.tolist())
    elif isinstance(obj, Mapping):
        return {nested_simplify(k, decimals): nested_simplify(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, (str, int, np.int64)) or obj is None:
        return obj
    elif is_torch_available() and isinstance(obj, torch.Tensor):
        return nested_simplify(obj.tolist(), decimals)
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, (np.int32, np.float32, np.float16)):
        return nested_simplify(obj.item(), decimals)
    else:
        raise Exception(f"Not supported: {type(obj)}")


def check_json_file_has_correct_format(file_path):
    with open(file_path) as f:
        lines = f.readlines()
        if len(lines) == 1:
            assert lines[0] == "{}"
        else:
            assert len(lines) >= 3
            assert lines[0].strip() == "{"
            for line in lines[1:-1]:
                left_indent = len(lines[1]) - len(lines[1].lstrip())
                assert left_indent == 2
            assert lines[-1].strip() == "}"


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


class SubprocessCallException(Exception):
    pass


def run_command(command: list[str], return_stdout=False):
    pass


class RequestCounter:

    def __enter__(self):
        self._counter = defaultdict(int)
        self._thread_id = threading.get_ident()
        self._extra_info = []

        def patched_with_thread_info(func):
            pass

        import urllib3

        self.patcher = patch.object(
            urllib3.connectionpool.log, "debug", side_effect=patched_with_thread_info(urllib3.connectionpool.log.debug)
        )
        self.mock = self.patcher.start()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        assert len(self.mock.call_args_list) == len(self._extra_info)
        for thread_id, call in zip(self._extra_info, self.mock.call_args_list):
            if thread_id != self._thread_id:
                continue
            if call.args[-2] == 307:
                continue
            log = call.args[0] % call.args[1:]
            for method in ("HEAD", "GET", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"):
                if method in log:
                    self._counter[method] += 1
                    break
        self.patcher.stop()

    def __getitem__(self, key: str) -> int:
        return self._counter[key]

    @property
    def total_calls(self) -> int:
        pass


def is_flaky(max_attempts: int = 5, wait_before_retry: float | None = None, description: str | None = None):
    """
    To decorate flaky tests. They will be retried on failures.

    Please note that our push tests use `pytest-rerunfailures`, which prompts the CI to rerun certain types of
    failed tests. More specifically, if the test exception contains any substring in `FLAKY_TEST_FAILURE_PATTERNS`
    (in `.circleci/create_circleci_config.py`), it will be rerun. If you find a recurrent pattern of failures,
    expand `FLAKY_TEST_FAILURE_PATTERNS` in our CI configuration instead of using `is_flaky`.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """

    def decorator(test_func_ref):
        pass

    return decorator


def hub_retry(max_attempts: int = 5, wait_before_retry: float | None = 2):
    """
    To decorate tests that download from the Hub. They can fail due to a
    variety of network issues such as timeouts, connection resets, etc.

    Uses exponential backoff starting from `wait_before_retry`.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*, defaults to 2):
            If provided, the initial delay in seconds before the first retry.
            Subsequent retries use exponential backoff with jitter.
    """
    from .utils.generic import retry

    return retry(
        max_retries=max_attempts,
        initial_delay=wait_before_retry or 0,
        jitter=wait_before_retry is not None,
        exceptions=(httpx.HTTPError,),
    )


def run_first(test_case):
    """
    Decorator marking a test with order(1). When pytest-order plugin is installed, tests marked with this decorator
    are guaranteed to run first.

    This is especially useful in some test settings like on a Gaudi instance where a Gaudi device can only be used by a
    single process at a time. So we make sure all tests that run in a subprocess are launched first, to avoid device
    allocation conflicts.
    """
    if is_pytest_order_available():
        import pytest

        return pytest.mark.order(1)(test_case)
    else:
        return test_case


def run_test_in_subprocess(test_case, target_func, inputs=None, timeout=None):
    """
    To run a test in a subprocess. In particular, this can avoid (GPU) memory issue.

    Args:
        test_case (`unittest.TestCase`):
            The test that will run `target_func`.
        target_func (`Callable`):
            The function implementing the actual testing logic.
        inputs (`dict`, *optional*, defaults to `None`):
            The inputs that will be passed to `target_func` through an (input) queue.
        timeout (`int`, *optional*, defaults to `None`):
            The timeout (in seconds) that will be passed to the input and output queues. If not specified, the env.
            variable `PYTEST_TIMEOUT` will be checked. If still `None`, its value will be set to `600`.
    """
    if timeout is None:
        timeout = int(os.environ.get("PYTEST_TIMEOUT", "600"))

    start_methohd = "spawn"
    ctx = multiprocessing.get_context(start_methohd)

    input_queue = ctx.Queue(1)
    output_queue = ctx.JoinableQueue(1)

    input_queue.put(inputs, timeout=timeout)

    process = ctx.Process(target=target_func, args=(input_queue, output_queue, timeout))
    process.start()
    try:
        results = output_queue.get(timeout=timeout)
        output_queue.task_done()
    except Exception as e:
        process.terminate()
        test_case.fail(e)
    process.join(timeout=timeout)

    if results["error"] is not None:
        test_case.fail(f"{results['error']}")


def run_test_using_subprocess(func):
    """
    To decorate a test to run in a subprocess using the `subprocess` module. This could avoid potential GPU memory
    issues (GPU OOM or a test that causes many subsequential failing with `CUDA error: device-side assert triggered`).
    """
    import pytest

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.getenv("_INSIDE_SUB_PROCESS", None) == "1":
            func(*args, **kwargs)
        else:
            test = " ".join(os.environ.get("PYTEST_CURRENT_TEST").split(" ")[:-1])
            try:
                env = copy.deepcopy(os.environ)
                env["_INSIDE_SUB_PROCESS"] = "1"
                env["CI"] = "true"

                if "pytestconfig" in kwargs:
                    command = list(kwargs["pytestconfig"].invocation_params.args)
                    for idx, x in enumerate(command):
                        if x in kwargs["pytestconfig"].args:
                            test = test.split("::")[1:]
                            command[idx] = "::".join([f"{func.__globals__['__file__']}"] + test)
                    command = [f"{sys.executable}", "-m", "pytest"] + command
                    command = [x for x in command if x != "--no-summary"]
                else:
                    command = [f"{sys.executable}", "-m", "pytest", f"{test}"]

                subprocess.run(command, env=env, check=True, capture_output=True)
            except subprocess.CalledProcessError as e:
                exception_message = e.stdout.decode()
                lines = exception_message.split("\n")
                if "= test session starts =" in lines[0]:
                    text = ""
                    for line in lines[1:]:
                        if line.startswith("FAILED "):
                            text = line[len("FAILED ") :]
                            text = "".join(text.split(" - ")[1:])
                        elif line.startswith("=") and line.endswith("=") and " failed in " in line:
                            break
                        elif len(text) > 0:
                            text += f"\n{line}"
                    text = "(subprocess) " + text
                    lines = [text] + lines
                exception_message = "\n".join(lines)
                raise pytest.fail(exception_message, pytrace=False)

    return wrapper


"""
The following contains utils to run the documentation tests without having to overwrite any files.

The `preprocess_string` function adds `# doctest: +IGNORE_RESULT` markers on the fly anywhere a `load_dataset` call is
made as a print would otherwise fail the corresponding line.

To skip cuda tests, make sure to call `SKIP_CUDA_DOCTEST=1 pytest --doctest-modules <path_to_files_to_test>
"""


def preprocess_string(string, skip_cuda_tests):
    """Prepare a docstring or a `.md` file to be run by doctest.

    The argument `string` would be the whole file content if it is a `.md` file. For a python file, it would be one of
    its docstring. In each case, it may contain multiple python code examples. If `skip_cuda_tests` is `True` and a
    cuda stuff is detective (with a heuristic), this method will return an empty string so no doctest will be run for
    `string`.
    """
    codeblock_pattern = r"(```(?:python|py)[^\S\n]*\n\s*>>> )(.*?```)"
    codeblocks = re.split(codeblock_pattern, string, flags=re.DOTALL)
    is_cuda_found = False
    for i, codeblock in enumerate(codeblocks):
        if "load_dataset(" in codeblock and "# doctest: +IGNORE_RESULT" not in codeblock:
            codeblocks[i] = re.sub(r"(>>> .*load_dataset\(.*)", r"\1 # doctest: +IGNORE_RESULT", codeblock)
        if (
            (">>>" in codeblock or "..." in codeblock)
            and re.search(r"cuda|to\(0\)|device=0", codeblock)
            and skip_cuda_tests
        ):
            is_cuda_found = True
            break

    modified_string = ""
    if not is_cuda_found:
        modified_string = "".join(codeblocks)

    return modified_string


class HfDocTestParser(doctest.DocTestParser):

    _EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?![ ]*$)    # Not a blank line
             (?![ ]*>>>)          # Not a line starting with PS1
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:(?!```).)*        # Match any character except '`' until a '```' is found (this is specific to HF because black removes the last line)
             # !!!!!!!!!!! HF Specific !!!!!!!!!!!
             (?:\n|$)  # Match a new line or end of string
          )*)
        ''', re.MULTILINE | re.VERBOSE
    )

    skip_cuda_tests: bool = os.environ.get("SKIP_CUDA_DOCTEST", "0") == "1"

    def parse(self, string, name="<string>"):
        """
        Overwrites the `parse` method to incorporate a skip for CUDA tests, and remove logs and dataset prints before
        calling `super().parse`
        """
        string = preprocess_string(string, self.skip_cuda_tests)
        return super().parse(string, name)


class HfDoctestModule(Module):

    def collect(self) -> Iterable[DoctestItem]:
        class MockAwareDocTestFinder(doctest.DocTestFinder):

            def _find_lineno(self, obj, source_lines):
                """Doctest code does not take into account `@property`, this
                is a hackish way to fix it. https://bugs.python.org/issue17446

                Wrapped Doctests will need to be unwrapped so the correct line number is returned. This will be
                reported upstream. #8796
                """
                if isinstance(obj, property):
                    obj = getattr(obj, "fget", obj)

                if hasattr(obj, "__wrapped__"):
                    obj = inspect.unwrap(obj)

                return super()._find_lineno(  # type:ignore[misc]
                    obj,
                    source_lines,
                )

            def _find(self, tests, obj, name, module, source_lines, globs, seen) -> None:
                if _is_mocked(obj):
                    return
                with _patch_unwrap_mock_aware():
                    super()._find(  # type:ignore[misc]
                        tests, obj, name, module, source_lines, globs, seen
                    )

        if self.path.name == "conftest.py":
            module = self.config.pluginmanager._importconftest(
                self.path,
                self.config.getoption("importmode"),
                rootpath=self.config.rootpath,
            )
        else:
            try:
                module = import_path(
                    self.path,
                    root=self.config.rootpath,
                    mode=self.config.getoption("importmode"),
                )
            except ImportError:
                if self.config.getvalue("doctest_ignore_import_errors"):
                    skip("unable to import module %r" % self.path)
                else:
                    raise

        finder = MockAwareDocTestFinder(parser=HfDocTestParser())
        optionflags = get_optionflags(self)
        runner = _get_runner(
            verbose=False,
            optionflags=optionflags,
            checker=_get_checker(),
            continue_on_failure=_get_continue_on_failure(self.config),
        )
        for test in finder.find(module, module.__name__):
            if test.examples:  # skip empty doctests and cuda
                yield DoctestItem.from_parent(self, name=test.name, runner=runner, dtest=test)


def _device_agnostic_dispatch(device: str, dispatch_table: dict[str, Callable], *args, **kwargs):
    if device not in dispatch_table:
        if not callable(dispatch_table["default"]):
            return dispatch_table["default"]

        return dispatch_table["default"](*args, **kwargs)

    fn = dispatch_table[device]

    if not callable(fn):
        return fn

    return fn(*args, **kwargs)


if is_torch_available():
    BACKEND_MANUAL_SEED = {
        "cuda": torch.cuda.manual_seed,
        "cpu": torch.manual_seed,
        "default": torch.manual_seed,
    }
    BACKEND_EMPTY_CACHE = {
        "cuda": torch.cuda.empty_cache,
        "cpu": None,
        "default": None,
    }
    BACKEND_DEVICE_COUNT = {
        "cuda": torch.cuda.device_count,
        "cpu": lambda: 0,
        "default": lambda: 1,
    }
    BACKEND_RESET_MAX_MEMORY_ALLOCATED = {
        "cuda": torch.cuda.reset_max_memory_allocated,
        "cpu": None,
        "default": None,
    }
    BACKEND_MAX_MEMORY_ALLOCATED = {
        "cuda": torch.cuda.max_memory_allocated,
        "cpu": 0,
        "default": 0,
    }
    BACKEND_RESET_PEAK_MEMORY_STATS = {
        "cuda": torch.cuda.reset_peak_memory_stats,
        "cpu": None,
        "default": None,
    }
    BACKEND_MEMORY_ALLOCATED = {
        "cuda": torch.cuda.memory_allocated,
        "cpu": 0,
        "default": 0,
    }
    BACKEND_SYNCHRONIZE = {
        "cuda": torch.cuda.synchronize,
        "cpu": None,
        "default": None,
    }
    BACKEND_TORCH_ACCELERATOR_MODULE = {
        "cuda": torch.cuda,
        "cpu": None,
        "default": None,
    }
else:
    BACKEND_MANUAL_SEED = {"default": None}
    BACKEND_EMPTY_CACHE = {"default": None}
    BACKEND_DEVICE_COUNT = {"default": lambda: 0}
    BACKEND_RESET_MAX_MEMORY_ALLOCATED = {"default": None}
    BACKEND_RESET_PEAK_MEMORY_STATS = {"default": None}
    BACKEND_MAX_MEMORY_ALLOCATED = {"default": 0}
    BACKEND_MEMORY_ALLOCATED = {"default": 0}
    BACKEND_SYNCHRONIZE = {"default": None}
    BACKEND_TORCH_ACCELERATOR_MODULE = {"default": None}


if is_torch_hpu_available():
    BACKEND_MANUAL_SEED["hpu"] = torch.hpu.manual_seed
    BACKEND_DEVICE_COUNT["hpu"] = torch.hpu.device_count
    BACKEND_TORCH_ACCELERATOR_MODULE["hpu"] = torch.hpu

if is_torch_mlu_available():
    BACKEND_EMPTY_CACHE["mlu"] = torch.mlu.empty_cache
    BACKEND_MANUAL_SEED["mlu"] = torch.mlu.manual_seed
    BACKEND_DEVICE_COUNT["mlu"] = torch.mlu.device_count
    BACKEND_TORCH_ACCELERATOR_MODULE["mlu"] = torch.mlu

if is_torch_npu_available():
    BACKEND_EMPTY_CACHE["npu"] = torch.npu.empty_cache
    BACKEND_MANUAL_SEED["npu"] = torch.npu.manual_seed
    BACKEND_DEVICE_COUNT["npu"] = torch.npu.device_count
    BACKEND_TORCH_ACCELERATOR_MODULE["npu"] = torch.npu

if is_torch_xpu_available():
    BACKEND_EMPTY_CACHE["xpu"] = torch.xpu.empty_cache
    BACKEND_MANUAL_SEED["xpu"] = torch.xpu.manual_seed
    BACKEND_DEVICE_COUNT["xpu"] = torch.xpu.device_count
    BACKEND_RESET_MAX_MEMORY_ALLOCATED["xpu"] = torch.xpu.reset_peak_memory_stats
    BACKEND_RESET_PEAK_MEMORY_STATS["xpu"] = torch.xpu.reset_peak_memory_stats
    BACKEND_MAX_MEMORY_ALLOCATED["xpu"] = torch.xpu.max_memory_allocated
    BACKEND_MEMORY_ALLOCATED["xpu"] = torch.xpu.memory_allocated
    BACKEND_SYNCHRONIZE["xpu"] = torch.xpu.synchronize
    BACKEND_TORCH_ACCELERATOR_MODULE["xpu"] = torch.xpu


if is_torch_xla_available():
    BACKEND_EMPTY_CACHE["xla"] = torch.cuda.empty_cache
    BACKEND_MANUAL_SEED["xla"] = torch.cuda.manual_seed
    BACKEND_DEVICE_COUNT["xla"] = torch.cuda.device_count


def backend_manual_seed(device: str, seed: int):
    return _device_agnostic_dispatch(device, BACKEND_MANUAL_SEED, seed)


def backend_empty_cache(device: str):
    return _device_agnostic_dispatch(device, BACKEND_EMPTY_CACHE)


def backend_device_count(device: str):
    return _device_agnostic_dispatch(device, BACKEND_DEVICE_COUNT)


def backend_reset_max_memory_allocated(device: str):
    return _device_agnostic_dispatch(device, BACKEND_RESET_MAX_MEMORY_ALLOCATED)


def backend_reset_peak_memory_stats(device: str):
    pass


def backend_max_memory_allocated(device: str):
    return _device_agnostic_dispatch(device, BACKEND_MAX_MEMORY_ALLOCATED)


def backend_memory_allocated(device: str):
    return _device_agnostic_dispatch(device, BACKEND_MEMORY_ALLOCATED)


def backend_synchronize(device: str):
    return _device_agnostic_dispatch(device, BACKEND_SYNCHRONIZE)


def backend_torch_accelerator_module(device: str):
    return _device_agnostic_dispatch(device, BACKEND_TORCH_ACCELERATOR_MODULE)


if is_torch_available():
    if "TRANSFORMERS_TEST_DEVICE_SPEC" in os.environ:
        device_spec_path = os.environ["TRANSFORMERS_TEST_DEVICE_SPEC"]
        if not Path(device_spec_path).is_file():
            raise ValueError(
                f"Specified path to device spec file is not a file or not found. Received '{device_spec_path}"
            )

        device_spec_dir, _ = os.path.split(os.path.realpath(device_spec_path))
        sys.path.append(device_spec_dir)
        try:
            import_name = device_spec_path[: device_spec_path.index(".py")]
        except ValueError as e:
            raise ValueError(f"Provided device spec file was not a Python file! Received '{device_spec_path}") from e

        device_spec_module = importlib.import_module(import_name)

        try:
            device_name = device_spec_module.DEVICE_NAME
        except AttributeError as e:
            raise AttributeError("Device spec file did not contain `DEVICE_NAME`") from e

        if "TRANSFORMERS_TEST_DEVICE" in os.environ and torch_device != device_name:
            msg = f"Mismatch between environment variable `TRANSFORMERS_TEST_DEVICE` '{torch_device}' and device found in spec '{device_name}'\n"
            msg += "Either unset `TRANSFORMERS_TEST_DEVICE` or ensure it matches device spec name."
            raise ValueError(msg)

        torch_device = device_name

        def update_mapping_from_spec(device_fn_dict: dict[str, Callable], attribute_name: str):
            pass

        update_mapping_from_spec(BACKEND_MANUAL_SEED, "MANUAL_SEED_FN")
        update_mapping_from_spec(BACKEND_EMPTY_CACHE, "EMPTY_CACHE_FN")
        update_mapping_from_spec(BACKEND_DEVICE_COUNT, "DEVICE_COUNT_FN")


def compare_pipeline_output_to_hub_spec(output, hub_spec):
    missing_keys = []
    unexpected_keys = []
    all_field_names = {field.name for field in fields(hub_spec)}
    matching_keys = sorted([key for key in output if key in all_field_names])

    for field in fields(hub_spec):
        if field.default is MISSING and field.name not in output:
            missing_keys.append(field.name)

    for output_key in output:
        if output_key not in all_field_names:
            unexpected_keys.append(output_key)

    if missing_keys or unexpected_keys:
        error = ["Pipeline output does not match Hub spec!"]
        if matching_keys:
            error.append(f"Matching keys: {matching_keys}")
        if missing_keys:
            error.append(f"Missing required keys in pipeline output: {missing_keys}")
        if unexpected_keys:
            error.append(f"Keys in pipeline output that are not in Hub spec: {unexpected_keys}")
        raise KeyError("\n".join(error))


@require_torch
def cleanup(device: str, gc_collect=False):
    if gc_collect:
        gc.collect()
    backend_empty_cache(device)
    torch.compiler.reset()


DeviceProperties = tuple[str | None, int | None, int | None]
PackedDeviceProperties = tuple[str | None, None | int | tuple[int, int]]


@cache
def get_device_properties() -> DeviceProperties:
    """
    Get environment device properties.
    """
    if IS_CUDA_SYSTEM or IS_ROCM_SYSTEM:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if IS_ROCM_SYSTEM:
                return ("rocm", major, minor)
            else:
                return ("cuda", major, minor)
    if IS_XPU_SYSTEM:
        import torch

        if torch.xpu.is_available():
            arch = torch.xpu.get_device_capability()["architecture"]
            gen_mask = 0x000000FF00000000
            gen = (arch & gen_mask) >> 32
            return ("xpu", gen, None)
    if IS_NPU_SYSTEM:
        return ("npu", None, None)
    return (torch_device, None, None)


def unpack_device_properties(
    properties: PackedDeviceProperties | None = None,
) -> DeviceProperties:
    """
    Unpack a `PackedDeviceProperties` tuple into consistently formatted `DeviceProperties` tuple. If properties is None, it is fetched.
    """
    if properties is None:
        return get_device_properties()
    device_type, major_minor = properties
    if major_minor is None:
        major, minor = None, None
    elif isinstance(major_minor, int):
        major, minor = major_minor, None
    else:
        major, minor = major_minor
    return device_type, major, minor


class Expectations(UserDict[PackedDeviceProperties, Any]):
    def get_expectation(self) -> Any:
        pass

    def unpacked(self) -> list[tuple[DeviceProperties, Any]]:
        return [(unpack_device_properties(k), v) for k, v in self.data.items()]

    @staticmethod
    def is_default(expectation_key: PackedDeviceProperties) -> bool:
        """
        This function returns True if the expectation_key is the Default expectation (None, None).
        When an Expectation dict contains a Default value, it is generally because the test existed before Expectations.
        When we modify a test to use Expectations for a specific hardware, we don't want to affect the tests on other
        hardwares. Thus we set the previous value as the Default expectation with key (None, None) and add a value for
        the specific hardware with key (hardware_type, (major, minor)).
        """
        return all(p is None for p in expectation_key)

    @staticmethod
    def score(properties: DeviceProperties, other: DeviceProperties) -> float:
        """
        Returns score indicating how similar two instances of the `Properties` tuple are.
        Rules are as follows:
            * Matching `type` adds one point, semi-matching `type` adds 0.1 point (e.g. cuda and rocm).
            * If types match, matching `major` adds another point, and then matching `minor` adds another.
            * The Default expectation (None, None) is worth 0.5 point, which is better than semi-matching. More on this
            in the `is_default` function.
        """
        device_type, major, minor = properties
        other_device_type, other_major, other_minor = other

        score = 0
        if device_type is not None and device_type == other_device_type:
            score += 1
            if major is not None and major == other_major:
                score += 1
                if minor is not None and minor == other_minor:
                    score += 1
        elif device_type in ["cuda", "rocm"] and other_device_type in ["cuda", "rocm"]:
            score = 0.1

        if Expectations.is_default(other):
            score = 0.5

        return score

    def find_expectation(self, properties: DeviceProperties = (None, None, None)) -> Any:
        """
        Find best matching expectation based on provided device properties. We score each expectation, and to
        distinguish between expectations with the same score, we use the major and minor version numbers, prioritizing
        most recent versions.
        """
        (result_key, result) = max(
            self.unpacked(),
            key=lambda x: (
                Expectations.score(properties, x[0]),  # x[0] is a device properties tuple (device_type, major, minor)
                x[0][1] if x[0][1] is not None else -1,  # This key is the major version, -1 if major is None
                x[0][2] if x[0][2] is not None else -1,  # This key is the minor version, -1 if minor is None
            ),
        )

        if Expectations.score(properties, result_key) == 0:
            raise ValueError(f"No matching expectation found for {properties}")

        return result

    def __repr__(self):
        return f"{self.data}"


def patch_torch_compile_force_graph():
    pass


def _get_test_info():
    pass


def _get_call_arguments(code_context):
    pass


def _prepare_debugging_info(test_info, info):
    pass


def _patched_tearDown(self, *args, **kwargs):
    pass


def _patch_with_call_info(module_or_class, attr_name, _parse_call_info_func, target_args):
    pass


def _parse_call_info(func, args, kwargs, call_argument_expressions, target_args):
    pass


def patch_testing_methods_to_collect_info():
    pass


def torchrun(script: str, nproc_per_node: int, is_torchrun: bool = True, env: dict | None = None):
    pass


def _format_tensor(t, indent_level=0, sci_mode=None):
    pass


def _quote_string(s):
    pass


def _format_py_obj(obj, indent=0, mode="", cache=None, prefix=""):
    pass


def write_file(file, content):
    pass


def read_json_file(file):
    with open(file, "r") as fh:
        return json.load(fh)




class Colors:

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_CYAN = "\033[96m"


class ColoredFormatter(logging.Formatter):

    LEVEL_COLORS = {
        logging.DEBUG: Colors.DIM + Colors.CYAN,
        logging.INFO: Colors.WHITE,
        logging.WARNING: Colors.BRIGHT_YELLOW,
        logging.ERROR: Colors.BRIGHT_RED,
        logging.CRITICAL: Colors.BOLD + Colors.BRIGHT_RED,
    }

    DIMMED_LOGGERS = {"httpx", "httpcore", "urllib3", "requests"}

    def __init__(self, fmt: str | None = None, datefmt: str | None = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        pass


_warn_once_logged: set[str] = set()


def init_test_logger() -> logging.Logger:
    """Initialize a test-specific logger with colored stderr handler and INFO level for tests.

    Uses a named logger instead of root logger to avoid conflicts with pytest-xdist parallel execution.
    Uses stderr instead of stdout to avoid deadlocks with pytest-xdist output capture.
    """
    logger = logging.getLogger("transformers.training_test")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.INFO)

        if sys.stderr.isatty():
            formatter = ColoredFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )

        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.propagate = False  # Don't propagate to root logger to avoid duplicate output
    return logger


def warn_once(logger_instance: logging.Logger, msg: str) -> None:
    pass


MemoryStats = collections.namedtuple(
    "MemoryStats",
    [
        "rss_gib",  # Resident Set Size in GiB
        "rss_pct",  # RSS as percentage of total memory
        "vms_gib",  # Virtual Memory Size in GiB
        "peak_rss_gib",  # Peak RSS in GiB
        "peak_rss_pct",  # Peak RSS as percentage of total memory
        "available_gib",  # Available system memory in GiB
        "total_gib",  # Total system memory in GiB
    ],
)


class CPUMemoryMonitor:

    def __init__(self):
        self.device_name = "CPU"
        self._peak_rss = 0
        self._process = None
        self.total_memory = 0
        self.total_memory_gib = 0

        if is_psutil_available():
            import psutil

            self._process = psutil.Process(os.getpid())
            mem_info = psutil.virtual_memory()
            self.total_memory = mem_info.total
            self.total_memory_gib = self._to_gib(self.total_memory)

    def _to_gib(self, memory_in_bytes: int) -> float:
        """Convert bytes to GiB."""
        return memory_in_bytes / (1024 * 1024 * 1024)

    def _to_pct(self, memory_in_bytes: int) -> float:
        pass

    def _update_peak(self) -> None:
        pass

    def get_stats(self) -> MemoryStats:
        pass

    def reset_peak_stats(self) -> None:
        pass


def build_cpu_memory_monitor(logger_instance: logging.Logger | None = None) -> CPUMemoryMonitor:
    """Build and initialize a CPU memory monitor.

    Args:
        logger_instance: Optional logger to log initialization info. If None, no logging is done.

    Returns:
        CPUMemoryMonitor instance.
    """
    monitor = CPUMemoryMonitor()
    if logger_instance is not None:
        if is_psutil_available():
            logger_instance.info(f"CPU memory monitor initialized: {monitor.total_memory_gib:.2f} GiB total")
        else:
            logger_instance.warning("psutil not available, memory monitoring disabled")
    return monitor


def convert_all_safetensors_to_bins(folder: str):
    """Convert all safetensors files into torch bin files, to mimic saving with torch (since we still support loading
    bin files, but not saving them anymore)"""
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.endswith(".safetensors"):
            new_path = path.replace(".safetensors", ".bin").replace("model", "pytorch_model")
            state_dict = load_file(path)
            os.remove(path)
            torch.save(state_dict, new_path)
        elif file == SAFE_WEIGHTS_INDEX_NAME:
            new_path = os.path.join(folder, WEIGHTS_INDEX_NAME)
            with open(path) as f:
                index = json.loads(f.read())
            os.remove(path)
            if "weight_map" in index.keys():
                weight_map = index["weight_map"]
                new_weight_map = {}
                for k, v in weight_map.items():
                    new_weight_map[k] = v.replace(".safetensors", ".bin").replace("model", "pytorch_model")
            index["weight_map"] = new_weight_map
            with open(new_path, "w") as f:
                f.write(json.dumps(index, indent=4))


@contextmanager
def force_serialization_as_bin_files():
    """Since we don't support saving with torch `.bin` files anymore, but still support loading them, we use this context
    to easily create the bin files and try to load them back"""
    try:
        original_save = PreTrainedModel.save_pretrained

        def new_save(self, save_directory, *args, **kwargs):
            pass

        PreTrainedModel.save_pretrained = new_save

        yield
    finally:
        PreTrainedModel.save_pretrained = original_save
