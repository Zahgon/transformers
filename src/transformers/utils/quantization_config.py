#!/usr/bin/env python

import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from packaging import version

from ..utils import (
    is_compressed_tensors_available,
    is_hqq_available,
    is_quark_available,
    is_torch_available,
    is_torchao_available,
    logging,
)


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuantizationMethod(str, Enum):
    BITS_AND_BYTES = "bitsandbytes"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    VPTQ = "vptq"
    QUANTO = "quanto"
    EETQ = "eetq"
    HIGGS = "higgs"
    HQQ = "hqq"
    COMPRESSED_TENSORS = "compressed-tensors"
    FBGEMM_FP8 = "fbgemm_fp8"
    TORCHAO = "torchao"
    BITNET = "bitnet"
    SPQR = "spqr"
    FP8 = "fp8"
    QUARK = "quark"
    FPQUANT = "fp_quant"
    AUTOROUND = "auto-round"
    MXFP4 = "mxfp4"
    MXFP8 = "mxfp8"
    METAL = "metal"
    FOUR_OVER_SIX = "fouroversix"
    SINQ = "sinq"
    GEMMA = "gemma"


class AwqFormat(str, Enum):
    GEMM = "gemm"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    LLM_AWQ = "llm-awq"


class AwqBackend(str, Enum):
    LEGACY_AWQ = "autoawq"
    AUTO = "auto"
    AUTO_TRAINABLE = "auto_trainable"
    MACHETE = "machete"
    MARLIN = "marlin"
    EXLLAMA_V2 = "exllama_v2"
    EXLLAMA_V1 = "exllama_v1"
    GEMM = "gemm"
    GEMM_TRITON = "gemm_triton"
    GEMV = "gemv"
    GEMV_FAST = "gemv_fast"
    TORCH_AWQ = "torch_awq"
    TORCH_FUSED_AWQ = "torch_fused_awq"


@dataclass
class QuantizationConfigMixin:

    quant_method: QuantizationMethod

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`QuantizationConfigMixin`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.
        """
        config = cls(**config_dict)

        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    def to_json_file(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default
                `QuantizationConfig()` is serialized to JSON file.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return copy.deepcopy(self.__dict__)

    def __iter__(self):
        """allows `dict(obj)` for situations where obj may be a dict or QuantizationConfigMixin"""
        yield from copy.deepcopy(self.__dict__).items()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Default behavior: no diffing implemented for this config.
        """
        return self.to_dict()

    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PreTrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_diff_dict() if use_diff else self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def update(self, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            kwargs (`dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)

        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class AutoRoundConfig(QuantizationConfigMixin):

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        sym: bool = True,
        backend: str = "auto",
        **kwargs,
    ):
        self.bits = bits
        self.group_size = group_size
        self.sym = sym
        self.backend = backend
        self.packing_format = "auto_round:gptq"
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        self.quant_method = QuantizationMethod.AUTOROUND
        self.post_init()

    def post_init(self):
        r"""Safety checker that arguments are correct."""
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")

    def get_loading_attributes(self):
        loading_attributes_dict = {"backend": self.backend}
        return loading_attributes_dict

    def to_dict(self):
        config_dict = super().to_dict()
        return config_dict

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        quant_method = config_dict["quant_method"]
        if "auto-round" not in quant_method and "gptq" not in quant_method and "awq" not in quant_method:
            raise NotImplementedError(
                "Failed to convert to auto_round format. Only `gptqv1`, `awq`, and `auto-round` formats are supported."
            )

        if "gptq" in quant_method and "meta" in config_dict:
            raise NotImplementedError("Failed to convert gptq format to auto_round format. Only supports `gptqv1`")

        if "awq" in quant_method and config_dict.get("version", "gemm") != "gemm":
            raise NotImplementedError(
                "Failed to convert awq format to auto_round format. Only supports awq format with gemm version"
            )

        if "auto-round" not in quant_method:
            config_dict["packing_format"] = f"auto_round:{quant_method}"

        return super().from_dict(config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs)


@dataclass
class HqqConfig(QuantizationConfigMixin):

    def __init__(
        self,
        nbits: int = 4,
        group_size: int = 64,
        view_as_float: bool = False,
        axis: int | None = None,
        dynamic_config: dict | None = None,
        skip_modules: list[str] = ["lm_head"],
        **kwargs,
    ):
        if is_hqq_available():
            from hqq.core.quantize import BaseQuantizeConfig as HQQBaseQuantizeConfig
        else:
            raise ImportError(
                "A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`."
            )

        if axis is None:
            axis = 1
            logger.info("Setting axis=1 as faster backends such as TorchAO or BitBlas are only compatible with it.")

        if axis not in [0, 1]:
            raise ValueError("Invalid axis value. Only 0 and 1 are allowed.")

        if dynamic_config is not None:
            self.quant_config = {}
            for key in dynamic_config:
                self.quant_config[key] = HQQBaseQuantizeConfig(**dynamic_config[key])
        else:
            self.quant_config = HQQBaseQuantizeConfig(
                nbits=nbits, group_size=group_size, view_as_float=view_as_float, axis=axis
            )

        self.quant_method = QuantizationMethod.HQQ
        self.skip_modules = skip_modules

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

    @classmethod
    def from_dict(cls, config: dict[str, Any]):
        """
        Override from_dict, used in AutoQuantizationConfig.from_dict in quantizers/auto.py
        """
        instance = cls()
        instance.quant_config = config["quant_config"]
        instance.skip_modules = config["skip_modules"]
        return instance

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return {
            "quant_config": self.quant_config,
            "quant_method": self.quant_method,
            "skip_modules": self.skip_modules,
        }

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.
        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        default_config_dict = HqqConfig().to_dict()

        serializable_config_dict = {}

        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


@dataclass
class BitsAndBytesConfig(QuantizationConfigMixin):

    def __init__(
        self,
        load_in_8bit=False,
        load_in_4bit=False,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=None,
        llm_int8_enable_fp32_cpu_offload=False,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=None,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_storage=None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.BITS_AND_BYTES

        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are both True, but only one can be used at the same time")

        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self.llm_int8_threshold = llm_int8_threshold
        self.llm_int8_skip_modules = llm_int8_skip_modules
        self.llm_int8_enable_fp32_cpu_offload = llm_int8_enable_fp32_cpu_offload
        self.llm_int8_has_fp16_weight = llm_int8_has_fp16_weight
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant

        if bnb_4bit_compute_dtype is None:
            self.bnb_4bit_compute_dtype = torch.float32
        elif isinstance(bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        elif isinstance(bnb_4bit_compute_dtype, torch.dtype):
            self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        else:
            raise ValueError("bnb_4bit_compute_dtype must be a string or a torch.dtype")

        if bnb_4bit_quant_storage is None:
            self.bnb_4bit_quant_storage = torch.uint8
        elif isinstance(bnb_4bit_quant_storage, str):
            if bnb_4bit_quant_storage not in ["float16", "float32", "int8", "uint8", "float64", "bfloat16"]:
                raise ValueError(
                    "`bnb_4bit_quant_storage` must be a valid string (one of 'float16', 'float32', 'int8', 'uint8', 'float64', 'bfloat16') "
                )
            self.bnb_4bit_quant_storage = getattr(torch, bnb_4bit_quant_storage)
        elif isinstance(bnb_4bit_quant_storage, torch.dtype):
            self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        else:
            raise ValueError("bnb_4bit_quant_storage must be a string or a torch.dtype")

        if kwargs:
            logger.info(f"Unused kwargs: {list(kwargs.keys())}. These kwargs are not used in {self.__class__}.")

        self.post_init()

    @property
    def load_in_4bit(self):
        pass

    @load_in_4bit.setter
    def load_in_4bit(self, value: bool):
        pass

    @property
    def load_in_8bit(self):
        pass

    @load_in_8bit.setter
    def load_in_8bit(self, value: bool):
        pass

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.load_in_4bit, bool):
            raise TypeError("load_in_4bit must be a boolean")

        if not isinstance(self.load_in_8bit, bool):
            raise TypeError("load_in_8bit must be a boolean")

        if not isinstance(self.llm_int8_threshold, float):
            raise TypeError("llm_int8_threshold must be a float")

        if self.llm_int8_skip_modules is not None and not isinstance(self.llm_int8_skip_modules, list):
            raise TypeError("llm_int8_skip_modules must be a list of strings")
        if not isinstance(self.llm_int8_enable_fp32_cpu_offload, bool):
            raise TypeError("llm_int8_enable_fp32_cpu_offload must be a boolean")

        if not isinstance(self.llm_int8_has_fp16_weight, bool):
            raise TypeError("llm_int8_has_fp16_weight must be a boolean")

        if self.bnb_4bit_compute_dtype is not None and not isinstance(self.bnb_4bit_compute_dtype, torch.dtype):
            raise TypeError("bnb_4bit_compute_dtype must be torch.dtype")

        if not isinstance(self.bnb_4bit_quant_type, str):
            raise TypeError("bnb_4bit_quant_type must be a string")

        if not isinstance(self.bnb_4bit_use_double_quant, bool):
            raise TypeError("bnb_4bit_use_double_quant must be a boolean")

    def is_quantizable(self):
        pass

    def quantization_method(self):
        r"""
        This method returns the quantization method used for the model. If the model is not quantizable, it returns
        `None`.
        """
        if self.load_in_8bit:
            return "llm_int8"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "fp4":
            return "fp4"
        elif self.load_in_4bit and self.bnb_4bit_quant_type == "nf4":
            return "nf4"
        else:
            return None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["bnb_4bit_compute_dtype"] = str(output["bnb_4bit_compute_dtype"]).split(".")[1]
        output["bnb_4bit_quant_storage"] = str(output["bnb_4bit_quant_storage"]).split(".")[1]
        output["load_in_4bit"] = self.load_in_4bit
        output["load_in_8bit"] = self.load_in_8bit

        return output

    def __repr__(self):
        config_dict = self.to_dict()
        return f"{self.__class__.__name__} {json.dumps(config_dict, indent=2, sort_keys=True)}\n"

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        default_config_dict = BitsAndBytesConfig().to_dict()

        serializable_config_dict = {}

        for key, value in config_dict.items():
            if value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict


class ExllamaVersion(int, Enum):
    ONE = 1
    TWO = 2


@dataclass
class GPTQConfig(QuantizationConfigMixin):

    def __init__(
        self,
        bits: int,
        tokenizer: Any = None,
        dataset: list[str] | str | None = None,
        group_size: int = 128,
        damp_percent: float = 0.1,
        desc_act: bool = False,
        act_group_aware: bool = True,
        sym: bool = True,
        true_sequential: bool = True,
        format: str = "gptq",
        meta: dict[str, Any] | None = None,
        backend: str | None = None,
        model_seqlen: int | None = None,
        block_name_to_quantize: str | None = None,
        module_name_preceding_first_block: list[str] | None = None,
        batch_size: int = 1,
        pad_token_id: int | None = None,
        max_input_length: int | None = None,
        cache_block_outputs: bool = True,
        modules_in_block_to_quantize: list[list[str]] | None = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.GPTQ
        self.bits = bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.group_size = group_size
        self.damp_percent = damp_percent
        self.desc_act = desc_act
        self.act_group_aware = act_group_aware
        self.sym = sym
        self.true_sequential = true_sequential
        self.format = format.lower()
        if kwargs.get("checkpoint_format") is not None:
            self.format = kwargs.pop("checkpoint_format").lower()
        self.meta = meta
        self.backend = backend.lower() if isinstance(backend, str) else backend
        self.model_seqlen = model_seqlen
        self.block_name_to_quantize = block_name_to_quantize
        self.module_name_preceding_first_block = module_name_preceding_first_block
        self.batch_size = batch_size
        self.pad_token_id = pad_token_id
        self.max_input_length = max_input_length
        self.cache_block_outputs = cache_block_outputs
        self.modules_in_block_to_quantize = modules_in_block_to_quantize
        self.post_init()

    def get_loading_attributes(self):
        attributes_dict = copy.deepcopy(self.__dict__)
        loading_attributes = ["max_input_length", "backend"]
        loading_attributes_dict = {i: j for i, j in attributes_dict.items() if i in loading_attributes}
        return loading_attributes_dict

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.bits not in [2, 3, 4, 8]:
            raise ValueError(f"Only support quantization to [2,3,4,8] bits but found {self.bits}")
        if self.group_size != -1 and self.group_size <= 0:
            raise ValueError("group_size must be greater than 0 or equal to -1")
        if not (0 < self.damp_percent < 1):
            raise ValueError("damp_percent must between 0 and 1.")
        if self.dataset is not None:
            if isinstance(self.dataset, str):
                if self.dataset not in ["wikitext2", "c4", "c4-new"]:
                    raise ValueError(
                        f"""You have entered a string value for dataset. You can only choose between
                        ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
                    )
            elif not isinstance(self.dataset, list):
                raise ValueError(
                    f"""dataset needs to be either a list of string or a value in
                    ['wikitext2','c4','c4-new'], but we found {self.dataset}"""
                )

        if self.desc_act and self.act_group_aware:
            self.act_group_aware = False
            logger.warning("`act_group_aware` has been auto-disabled as it is not compatible with `desc_act = True`.")

        if self.backend is None:
            self.backend = "auto"
        if self.modules_in_block_to_quantize is not None:
            optimum_version = version.parse(importlib.metadata.version("optimum"))
            if optimum_version < version.parse("1.15.0"):
                raise ValueError(
                    "You current version of `optimum` does not support `modules_in_block_to_quantize` quantization argument, please upgrade `optimum` package to a version superior than 1.15.0 ."
                )

    def to_dict(self) -> dict[str, Any]:
        config_dict = super().to_dict()
        config_dict["checkpoint_format"] = self.format
        return config_dict

    def to_dict_optimum(self):
        """
        Get compatible dict for optimum gptq config
        """
        return self.to_dict()

    @classmethod
    def from_dict_optimum(cls, config_dict):
        pass


@dataclass
class AwqConfig(GPTQConfig):

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        zero_point: bool = True,
        backend: AwqBackend = AwqBackend.AUTO,
        modules_to_not_convert: list | None = None,
        **kwargs,
    ):
        format = kwargs.pop("format", AwqFormat.GEMM)
        if kwargs.get("version") is not None:
            format = kwargs.pop("version").lower()
        if backend == AwqBackend.LEGACY_AWQ:
            backend = AwqBackend.AUTO
        self.zero_point = zero_point
        self.modules_to_not_convert = modules_to_not_convert

        super().__init__(bits=bits, group_size=group_size, backend=backend, format=format, **kwargs)
        self.quant_method = QuantizationMethod.AWQ

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """

        if self.backend == "llm-awq":
            self.format = AwqFormat.LLM_AWQ
            self.backend = AwqBackend.AUTO

        if self.format not in AwqFormat.__members__.values():
            raise ValueError(f"Invalid format '{self.format}'. Must be one of: {[b.value for b in AwqFormat]}")

        if self.backend not in AwqBackend.__members__.values():
            raise ValueError(f"Invalid backend '{self.backend}'. Must be one of: {[b.value for b in AwqBackend]}")

    def to_dict(self) -> dict[str, Any]:
        config_dict = super().to_dict()
        config_dict.pop("checkpoint_format")
        config_dict["version"] = self.format
        return config_dict


@dataclass
class AqlmConfig(QuantizationConfigMixin):

    def __init__(
        self,
        in_group_size: int = 8,
        out_group_size: int = 1,
        num_codebooks: int = 1,
        nbits_per_codebook: int = 16,
        linear_weights_not_to_quantize: list[str] | None = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.AQLM
        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.num_codebooks = num_codebooks
        self.nbits_per_codebook = nbits_per_codebook
        self.linear_weights_not_to_quantize = linear_weights_not_to_quantize

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.in_group_size, int):
            raise TypeError("in_group_size must be an int")
        if not isinstance(self.out_group_size, int):
            raise TypeError("out_group_size must be an int")
        if not isinstance(self.num_codebooks, int):
            raise TypeError("num_codebooks must be an int")
        if not isinstance(self.nbits_per_codebook, int):
            raise TypeError("nbits_per_codebook must be an int")

        if self.linear_weights_not_to_quantize is not None and not isinstance(
            self.linear_weights_not_to_quantize, list
        ):
            raise ValueError("linear_weights_not_to_quantize must be a list of strings")

        if self.linear_weights_not_to_quantize is None:
            self.linear_weights_not_to_quantize = []


@dataclass
class VptqLayerConfig(QuantizationConfigMixin):

    def __init__(
        self,
        enable_norm: bool = True,
        enable_perm: bool = True,
        group_num: int = 1,
        group_size: int = -1,
        in_features: int = -1,
        indices_as_float: bool = False,
        is_indice_packed: bool = True,
        num_centroids: list = [-1, -1],
        num_res_centroids: list = [-1, -1],
        out_features: int = -1,
        outlier_size: int = 0,
        vector_lens: list = [-1, -1],
        **kwargs,
    ):
        self.enable_norm = enable_norm
        self.enable_perm = enable_perm
        self.group_num = group_num
        self.group_size = group_size
        self.in_features = in_features
        self.indices_as_float = indices_as_float
        self.is_indice_packed = is_indice_packed
        self.num_centroids = num_centroids
        self.num_res_centroids = num_res_centroids
        self.out_features = out_features
        self.outlier_size = outlier_size
        self.vector_lens = vector_lens
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        if self.is_indice_packed is False:
            raise ValueError("is_indice_packed should always be True")


@dataclass
class VptqConfig(QuantizationConfigMixin):

    def __init__(
        self,
        enable_proxy_error: bool = False,
        config_for_layers: dict[str, Any] = {},
        shared_layer_config: dict[str, Any] = {},
        modules_to_not_convert: list | None = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.VPTQ
        self.enable_proxy_error = enable_proxy_error
        self.config_for_layers: dict[str, Any] = config_for_layers
        self.shared_layer_config: dict[str, Any] = shared_layer_config
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        for layer_param in self.config_for_layers.values():
            VptqLayerConfig(**layer_param)
        if self.enable_proxy_error is True:
            raise ValueError("enable_proxy_error should always be False until we support training")


@dataclass
class QuantoConfig(QuantizationConfigMixin):

    def __init__(
        self,
        weights="int8",
        activations=None,
        modules_to_not_convert: list | None = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.QUANTO
        self.weights = weights
        self.activations = activations
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["float8", "int8", "int4", "int2"]
        accepted_activations = [None, "int8", "float8"]
        if self.weights not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights}")
        if self.activations not in accepted_activations:
            raise ValueError(f"Only support weights in {accepted_activations} but found {self.activations}")


@dataclass
class EetqConfig(QuantizationConfigMixin):

    def __init__(
        self,
        weights: str = "int8",
        modules_to_not_convert: list | None = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.EETQ
        self.weights = weights
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        accepted_weights = ["int8"]
        if self.weights not in accepted_weights:
            raise ValueError(f"Only support weights in {accepted_weights} but found {self.weights}")


class CompressedTensorsConfig(QuantizationConfigMixin):

    def __init__(
        self,
        config_groups: dict[str, Union["QuantizationScheme", list[str]]] | None = None,  # noqa: F821
        format: str = "dense",
        quantization_status: "QuantizationStatus" = "initialized",  # noqa: F821
        kv_cache_scheme: Optional["QuantizationArgs"] = None,  # noqa: F821
        global_compression_ratio: float | None = None,
        ignore: list[str] | None = None,
        quant_method: str = "compressed-tensors",
        run_compressed: bool = True,
        **kwargs,
    ):
        if is_compressed_tensors_available():
            from compressed_tensors.quantization import QuantizationConfig
        else:
            raise ImportError(
                "compressed-tensors>=0.15.0 is required for compressed-tensors quantization. Please install it with `pip install compressed-tensors>=0.15.0`."
            )
        self.quantization_config = None

        self.run_compressed = run_compressed

        if config_groups or kv_cache_scheme:
            self.quantization_config = QuantizationConfig.model_validate(
                {
                    "config_groups": config_groups,
                    "quant_method": quant_method,
                    "format": format,
                    "quantization_status": quantization_status,
                    "kv_cache_scheme": kv_cache_scheme,
                    "global_compression_ratio": global_compression_ratio,
                    "ignore": ignore,
                    **kwargs,
                }
            )

        self.quant_method = QuantizationMethod.COMPRESSED_TENSORS

    def post_init(self):
        if self.run_compressed and not self.is_quantization_compressed:
            logger.warning("`run_compressed` is only supported for compressed models. Setting `run_compressed=False`")
            self.run_compressed = False

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """
        Instantiates a [`CompressedTensorsConfig`] from a Python dictionary of parameters.
        Optionally unwraps any args from the nested quantization_config

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            return_unused_kwargs (`bool`,*optional*, defaults to `False`):
                Whether or not to return a list of unused keyword arguments. Used for `from_pretrained` method in
                `PreTrainedModel`.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`QuantizationConfigMixin`]: The configuration object instantiated from those parameters.

        """

        if "quantization_config" in config_dict:
            config_dict = config_dict["quantization_config"]

        return super().from_dict(config_dict, return_unused_kwargs=return_unused_kwargs, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """
        Quantization config to be added to config.json

        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        quantization_config = {}
        if self.quantization_config is not None:
            quantization_config = self.quantization_config.model_dump()
        else:
            quantization_config["quant_method"] = QuantizationMethod.COMPRESSED_TENSORS

        return quantization_config

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.
        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        default_config_dict = CompressedTensorsConfig().to_dict()

        serializable_config_dict = {}

        for key, value in config_dict.items():
            if key not in default_config_dict or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        return serializable_config_dict

    def get_loading_attributes(self):
        return {"run_compressed": self.run_compressed}

    @property
    def is_quantized(self):
        pass

    @property
    def is_quantization_compressed(self):
        pass


@dataclass
class FbgemmFp8Config(QuantizationConfigMixin):

    def __init__(
        self,
        activation_scale_ub: float = 1200.0,
        modules_to_not_convert: list | None = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.FBGEMM_FP8
        self.activation_scale_ub = activation_scale_ub
        self.modules_to_not_convert = modules_to_not_convert

    def get_loading_attributes(self):
        attributes_dict = copy.deepcopy(self.__dict__)
        loading_attributes = ["activation_scale_ub"]
        loading_attributes_dict = {i: j for i, j in attributes_dict.items() if i in loading_attributes}
        return loading_attributes_dict


@dataclass
class HiggsConfig(QuantizationConfigMixin):

    def __init__(
        self,
        bits: int = 4,
        p: int = 2,
        modules_to_not_convert: list[str] | None = None,
        hadamard_size: int = 512,
        group_size: int = 256,
        tune_metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        if tune_metadata is None:
            tune_metadata = {}
        self.quant_method = QuantizationMethod.HIGGS
        self.bits = bits
        self.p = p
        self.modules_to_not_convert = modules_to_not_convert
        self.hadamard_size = hadamard_size
        self.group_size = group_size
        self.tune_metadata = tune_metadata

        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if self.bits not in [2, 3, 4]:
            raise ValueError("bits must be 2, 3, or 4")
        if self.p not in [1, 2]:
            raise ValueError("p must be 1 or 2. 2 is always better in practice")
        if self.group_size not in [64, 128, 256]:
            raise ValueError("group_size must be 64, 128, or 256")
        if self.hadamard_size % self.group_size != 0:
            raise ValueError("hadamard_size must be divisible by group_size")


@dataclass
class FPQuantConfig(QuantizationConfigMixin):

    def __init__(
        self,
        forward_dtype: str = "nvfp4",
        forward_method: str = "abs_max",
        backward_dtype: str = "bf16",
        store_master_weights: bool = False,
        hadamard_group_size: int | None = None,
        pseudoquantization: bool = False,
        transform_init: str = "hadamard",
        modules_to_not_convert: list[str] | None = None,
        **kwargs,
    ):
        self.forward_dtype = forward_dtype
        self.forward_method = forward_method
        self.backward_dtype = backward_dtype
        self.store_master_weights = store_master_weights
        self.hadamard_group_size = hadamard_group_size
        self.pseudoquantization = pseudoquantization
        self.transform_init = transform_init
        self.modules_to_not_convert = modules_to_not_convert

        self.quant_method = QuantizationMethod.FPQUANT
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """

        if self.hadamard_group_size is None:
            if self.forward_dtype == "nvfp4":
                self.hadamard_group_size = 16
            else:
                self.hadamard_group_size = 32

        if self.forward_dtype == "mxfp4":
            if self.forward_method not in ["abs_max", "quest"]:
                raise ValueError("Only 'abs_max' and 'quest' are supported for forward_method for 'mxfp4'.")
            if self.hadamard_group_size is None:
                self.hadamard_group_size = 32
            if self.hadamard_group_size not in [32, 64, 128]:
                raise ValueError("Only a `hadamard_group_size` of [32, 64, 128] is supported for 'mxfp4'.")
        elif self.forward_dtype == "nvfp4":
            if self.forward_method != "abs_max":
                raise ValueError("Only 'abs_max' is supported for forward_method for 'nvfp4'.")
            if self.hadamard_group_size is None:
                self.hadamard_group_size = 16
            if self.hadamard_group_size not in [16, 32, 64, 128]:
                raise ValueError("Only a `hadamard_group_size` of [16, 32, 64, 128] is supported for 'nvfp4'.")
        else:
            raise ValueError("Only 'mxfp4' and 'nvfp4' are supported for forward_dtype for now.")

        if self.backward_dtype not in ["bf16", "mxfp8", "mxfp4"]:
            raise ValueError("Only 'bf16', 'mxfp8' and 'mxfp4' are supported for backward_dtype for now.")

        if self.backward_dtype != "bf16" and self.forward_dtype != "mxfp4":
            raise ValueError("Only 'mxfp4' forward is compatible with non-bf16 backwards for now.")

        if self.transform_init not in ["hadamard", "identity", "gsr"]:
            raise ValueError("Only 'hadamard', 'identity' and 'gsr' are supported for transform_init.")

        if self.modules_to_not_convert is None:
            self.modules_to_not_convert = ["lm_head"]


@dataclass
class TorchAoConfig(QuantizationConfigMixin):

    quant_method: QuantizationMethod
    quant_type: "AOBaseConfig"  # noqa: F821
    modules_to_not_convert: list | None
    include_input_output_embeddings: bool
    untie_embedding_weights: bool

    def __init__(
        self,
        quant_type: "AOBaseConfig",  # noqa: F821
        modules_to_not_convert: list | None = None,
        include_input_output_embeddings: bool = False,
        untie_embedding_weights: bool = False,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.TORCHAO
        self.quant_type = quant_type
        self.modules_to_not_convert = modules_to_not_convert
        self.include_input_output_embeddings = include_input_output_embeddings
        self.untie_embedding_weights = untie_embedding_weights
        self.post_init()

    def post_init(self):
        """Validate configuration and set defaults."""
        if not is_torchao_available():
            raise ValueError("TorchAoConfig requires torchao to be installed. Install with `pip install torchao`")

        if isinstance(self.quant_type, str):
            raise ValueError(
                f"String-based quantization type '{self.quant_type}' is no longer supported. "
                f"Please use the corresponding Config object directly, e.g. "
                f"TorchAoConfig(Int4WeightOnlyConfig(group_size=32)) instead of "
                f"TorchAoConfig('int4_weight_only', group_size=32)."
            )

        from torchao.quantization.quant_api import AOBaseConfig

        if not isinstance(self.quant_type, AOBaseConfig):
            raise TypeError(f"quant_type must be an AOBaseConfig instance, got {type(self.quant_type)}")

    def get_apply_tensor_subclass(self):
        """Return the quantization config to apply."""
        return self.quant_type

    def to_dict(self):
        """Convert configuration to a dictionary."""
        d = super().to_dict()

        from torchao.core.config import config_to_dict

        d["quant_type"] = {"default": config_to_dict(self.quant_type)}

        return d

    @classmethod
    def from_dict(cls, config_dict, return_unused_kwargs=False, **kwargs):
        """Create configuration from a dictionary."""
        from torchao.core.config import config_from_dict

        config_dict = config_dict.copy()
        quant_type = config_dict.pop("quant_type")

        assert len(quant_type) == 1 and "default" in quant_type, (
            "Expected only one key 'default' in quant_type dictionary"
        )
        quant_type = quant_type["default"]
        quant_type = config_from_dict(quant_type)

        return cls(quant_type=quant_type, **config_dict)


@dataclass
class BitNetQuantConfig(QuantizationConfigMixin):

    def __init__(
        self,
        modules_to_not_convert: list | None = None,
        linear_class: str = "bitlinear",
        quantization_mode: str = "offline",
        use_rms_norm: bool = False,
        rms_norm_eps: float | None = 1e-6,
        **kwargs,
    ):
        if linear_class not in ["bitlinear", "autobitlinear"]:
            raise ValueError(f"linear_class must be either 'bitlinear' or 'autobitlinear', but got {linear_class}")
        if quantization_mode not in ["online", "offline"]:
            raise ValueError(f"quantization_mode must be either 'online' or 'offline', but got {quantization_mode}")
        self.quant_method = QuantizationMethod.BITNET
        self.modules_to_not_convert = modules_to_not_convert
        self.linear_class = linear_class
        self.quantization_mode = quantization_mode
        self.use_rms_norm = use_rms_norm
        self.rms_norm_eps = rms_norm_eps
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """


@dataclass
class SpQRConfig(QuantizationConfigMixin):

    def __init__(
        self,
        bits: int = 3,
        beta1: int = 16,
        beta2: int = 16,
        shapes: dict[str, int] | None = None,
        modules_to_not_convert: list[str] | None = None,
        **kwargs,
    ):
        if shapes is None:
            shapes = {}
        self.shapes = shapes
        self.quant_method = QuantizationMethod.SPQR
        self.bits = bits
        self.beta1 = beta1
        self.beta2 = beta2
        self.modules_to_not_convert = modules_to_not_convert
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct - also replaces some NoneType arguments with their default values.
        """
        if not isinstance(self.bits, int):
            raise TypeError("bits must be an int")
        if not isinstance(self.beta1, int):
            raise TypeError("beta1 must be an int")
        if not isinstance(self.beta2, int):
            raise TypeError("beta2 must be an int")

        if self.bits != 3:
            raise ValueError("SpQR currently only supports bits = 3")
        if self.beta1 != 16:
            raise ValueError("SpQR currently only supports beta1 = 16")
        if self.beta2 != 16:
            raise ValueError("SpQR currently only supports beta2 = 16")
        if not isinstance(self.shapes, dict):
            raise TypeError("shapes must be a dict")


@dataclass
class FineGrainedFP8Config(QuantizationConfigMixin):

    def __init__(
        self,
        activation_scheme: str = "dynamic",
        weight_block_size: tuple[int, int] = (128, 128),
        dequantize: bool = False,
        modules_to_not_convert: list | None = None,
        scale_fmt: str = "float",
        **kwargs,
    ):
        self.quant_method = kwargs.pop("quant_method", QuantizationMethod.FP8)
        if modules_to_not_convert is None and "ignored_layers" in kwargs:
            modules_to_not_convert = kwargs.pop("ignored_layers")
        self.modules_to_not_convert = modules_to_not_convert
        self.activation_scheme = activation_scheme
        self.weight_block_size = weight_block_size
        self.dequantize = dequantize
        self.scale_fmt = scale_fmt
        self.post_init()

    def post_init(self):
        r"""
        Safety checker that arguments are correct
        """
        self.activation_scheme = self.activation_scheme.lower()
        if self.activation_scheme not in ["dynamic", "static"]:
            raise ValueError(f"Activation scheme {self.activation_scheme} not supported")
        if self.weight_block_size is not None and len(self.weight_block_size) != 2:
            raise ValueError("weight_block_size must be a tuple of two integers")
        if self.weight_block_size is not None and (self.weight_block_size[0] <= 0 or self.weight_block_size[1] <= 0):
            raise ValueError("weight_block_size must be a tuple of two positive integers")
        if self.scale_fmt not in ("float", "ue8m0"):
            raise ValueError(f"scale_fmt must be 'float' or 'ue8m0'; got {self.scale_fmt!r}")

    def get_loading_attributes(self):
        return {"dequantize": self.dequantize, "modules_to_not_convert": self.modules_to_not_convert}


class QuarkConfig(QuantizationConfigMixin):
    def __init__(
        self,
        **kwargs,
    ):
        if is_torch_available() and is_quark_available():
            from quark import __version__ as quark_version
            from quark.torch.export.config.config import JsonExporterConfig
            from quark.torch.export.main_export.quant_config_parser import QuantConfigParser
            from quark.torch.quantization.config.config import Config
        else:
            raise ImportError(
                "Quark is not installed. Please refer to https://quark.docs.amd.com/latest/install.html."
            )
        self.custom_mode = kwargs["quant_method"]
        self.legacy = "export" not in kwargs

        if self.custom_mode in ["awq", "fp8"]:
            self.quant_config = QuantConfigParser.from_custom_config(kwargs, is_bias_quantized=False)
            self.json_export_config = JsonExporterConfig()
        else:
            self.quant_config = Config.from_dict(kwargs)

            if "export" in kwargs:
                if "min_kv_scale" in kwargs["export"] and version.parse(quark_version) < version.parse("0.8"):
                    min_kv_scale = kwargs["export"].pop("min_kv_scale")
                    logger.warning(
                        f"The parameter `min_kv_scale={min_kv_scale}` was found in the model config.json's `quantization_config.export` configuration, but this parameter is supported only for quark>=0.8. Ignoring this configuration parameter. Please update the `amd-quark` package."
                    )

                self.json_export_config = JsonExporterConfig(**kwargs["export"])
            else:
                self.json_export_config = JsonExporterConfig()

        self.quant_method = QuantizationMethod.QUARK


@dataclass
class Mxfp4Config(QuantizationConfigMixin):

    def __init__(
        self,
        modules_to_not_convert: list | None = None,
        dequantize: bool = False,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.MXFP4
        self.modules_to_not_convert = modules_to_not_convert
        self.dequantize = dequantize

    def get_loading_attributes(self):
        return {"dequantize": self.dequantize}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        return {"quant_method": self.quant_method, "modules_to_not_convert": self.modules_to_not_convert}


class MetalConfig(QuantizationConfigMixin):

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 64,
        modules_to_not_convert: list | None = None,
        dequantize: bool = False,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.METAL
        self.bits = bits
        self.group_size = group_size
        self.modules_to_not_convert = modules_to_not_convert
        self.dequantize = dequantize
        self.post_init()

    def post_init(self):
        if self.bits not in (2, 4, 8):
            raise ValueError(f"Metal quantization only supports bits in {{2, 4, 8}}, got {self.bits}")
        if self.group_size <= 0:
            raise ValueError(f"group_size must be positive, got {self.group_size}")

    def get_loading_attributes(self):
        return {"dequantize": self.dequantize}

    def to_dict(self) -> dict[str, Any]:
        return {
            "quant_method": self.quant_method,
            "bits": self.bits,
            "group_size": self.group_size,
            "modules_to_not_convert": self.modules_to_not_convert,
        }


@dataclass
class FourOverSixConfig(QuantizationConfigMixin):

    def __init__(
        self,
        activation_dtype: str | None = None,
        activation_scale_rule: str | None = None,
        dtype: str = "nvfp4",
        gradient_dtype: str | None = None,
        gradient_scale_rule: str | None = None,
        keep_master_weights: bool = False,
        matmul_backend: str | None = None,
        output_dtype: str | None = "bfloat16",
        quantize_backend: str | None = None,
        scale_rule: str = "mse",
        weight_dtype: str | None = None,
        weight_scale_2d: bool = False,
        weight_scale_rule: str | None = None,
        module_config_overrides: dict[str, dict[str, Any]] | None = None,
        modules_to_not_convert: list[str] | None = ["lm_head"],
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.FOUR_OVER_SIX

        self.activation_dtype = activation_dtype
        self.activation_scale_rule = activation_scale_rule
        self.dtype = dtype
        self.gradient_dtype = gradient_dtype
        self.gradient_scale_rule = gradient_scale_rule
        self.keep_master_weights = keep_master_weights
        self.matmul_backend = matmul_backend
        self.quantize_backend = quantize_backend
        self.output_dtype = output_dtype
        self.scale_rule = scale_rule
        self.weight_dtype = weight_dtype
        self.weight_scale_2d = weight_scale_2d
        self.weight_scale_rule = weight_scale_rule
        self.module_config_overrides = module_config_overrides
        self.modules_to_not_convert = modules_to_not_convert


class SinqConfig(QuantizationConfigMixin):

    def __init__(
        self,
        nbits: int = 4,
        group_size: int = 64,
        tiling_mode: str = "1D",
        method: str = "sinq",  # "sinq" | "asinq"
        modules_to_not_convert: list[str] | None = None,
        **kwargs: Any,
    ):
        self.quant_method = QuantizationMethod.SINQ

        self.nbits = nbits
        self.group_size = group_size
        self.tiling_mode = tiling_mode
        self.method = method

        self.modules_to_not_convert = modules_to_not_convert

        self._extra_kwargs: dict[str, Any] = dict(kwargs)

        self.post_init()

    def post_init(self):
        self.nbits = int(self.nbits)
        self.group_size = int(self.group_size)
        self.tiling_mode = str(self.tiling_mode)
        self.method = str(self.method).lower()

        if not isinstance(self.nbits, int):
            raise TypeError("`nbits` must be convertible to an int")
        if not isinstance(self.group_size, int):
            raise TypeError("`group_size` must be convertible to an int")
        if not isinstance(self.tiling_mode, str):
            raise TypeError("`tiling_mode` must be convertible to a string")
        if self.method not in {"sinq", "asinq"}:
            raise ValueError(f"`method` must be either 'sinq' or 'asinq', got {self.method}")
        if self.group_size is not None and self.group_size % 8 != 0:
            logger.warning(
                f"SINQ: group_size={self.group_size} is not a multiple of 8; this may be rejected by the backend."
            )


@dataclass
class GemmaQuantizationConfig(QuantizationConfigMixin):

    def __init__(
        self,
        num_bits: int = 4,
        quantize_embeddings: bool = False,
        module_quant_configs: dict[str, dict] | None = None,
        modules_to_not_convert: list[str] | None = None,
        **kwargs,
    ):
        self.quant_method = QuantizationMethod.GEMMA
        self.num_bits = num_bits
        self.quantize_embeddings = quantize_embeddings
        self.module_quant_configs = module_quant_configs
        self.modules_to_not_convert = modules_to_not_convert
