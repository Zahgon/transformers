import inspect
import json
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Optional

from safetensors import safe_open

from .._typing import PeftConfigLike
from ..conversion_mapping import get_model_conversion_mapping
from ..utils import (
    CONFIG_NAME,
    cached_file,
    check_peft_version,
    extract_commit_hash,
    find_adapter_config_file,
    is_accelerate_available,
    is_peft_available,
    is_torch_available,
    logging,
)
from ..utils.hub import DownloadKwargs
from ..utils.loading_report import log_state_dict_report


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.utils import get_balanced_memory, infer_auto_device_map

MIN_PEFT_VERSION = "0.19.1"


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from ..modeling_utils import LoadStateDictConfig, LoadStateDictInfo


class PeftAdapterMixin:

    _hf_peft_config_loaded = False
    _prepare_peft_hotswap_kwargs: dict | None = None
    peft_config: dict[str, PeftConfigLike]

    def load_adapter(
        self,
        peft_model_id: str | None = None,
        adapter_name: str | None = None,
        peft_config: dict[str, Any] | None = None,
        adapter_state_dict: dict[str, "torch.Tensor"] | None = None,
        low_cpu_mem_usage: bool = False,
        is_trainable: bool = False,
        hotswap: bool | Literal["auto"] = "auto",
        local_files_only: bool = False,
        adapter_kwargs: dict[str, Any] | None = None,
        load_config: Optional["LoadStateDictConfig"] = None,
        **kwargs,
    ) -> "LoadStateDictInfo":
        """
        Load adapter weights from file or remote Hub folder. If you are not familiar with adapters and PEFT methods, we
        invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

        Requires PEFT to be installed as a backend to load the adapter weights.

        Args:
            peft_model_id (`str`, *optional*):
                The identifier of the model to look for on the Hub, or a local path to the saved adapter config file
                and adapter weights.
            adapter_name (`str`, *optional*):
                The adapter name to use. If not set, will use the name "default".
            load_config (`LoadStateDictConfig`, *optional*):
                A load configuration to reuse when pulling adapter weights, typically from `from_pretrained`.
            kwargs (`dict[str, Any]`, *optional*):
                Additional `LoadStateDictConfig` fields passed as keyword arguments.
            peft_config (`dict[str, Any]`, *optional*):
                The configuration of the adapter to add, supported adapters are all non-prompt learning configs (LoRA,
                IA³, etc). This argument is used in case users directly pass PEFT state dicts.
            adapter_state_dict (`dict[str, torch.Tensor]`, *optional*):
                The state dict of the adapter to load. This argument is used in case users directly pass PEFT state
                dicts.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `False`):
                Reduce memory usage while loading the PEFT adapter. This should also speed up the loading process.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            hotswap : (`"auto"` or `bool`, *optional*, defaults to `"auto"`)
                Whether to substitute an existing (LoRA) adapter with the newly loaded adapter in-place. This means
                that, instead of loading an additional adapter, this will take the existing adapter weights and replace
                them with the weights of the new adapter. This can be faster and more memory efficient. However, the
                main advantage of hotswapping is that when the model is compiled with torch.compile, loading the new
                adapter does not require recompilation of the model. When using hotswapping, the passed `adapter_name`
                should be the name of an already loaded adapter.

                If the new adapter and the old adapter have different ranks and/or LoRA alphas (i.e. scaling), you need
                to call an additional method before loading the adapter:

                ```py
                model = AutoModel.from_pretrained(...)
                max_rank = ...  # the highest rank among all LoRAs that you want to load
                # call *before* compiling and loading the LoRA adapter
                model.enable_peft_hotswap(target_rank=max_rank)
                model.load_adapter(file_name_1, adapter_name="default")
                # optionally compile the model now
                model = torch.compile(model, ...)
                output_1 = model(...)
                # now you can hotswap the 2nd adapter, use the same name as for the 1st
                # hotswap is activated by default since enable_peft_hotswap was called
                model.load_adapter(file_name_2, adapter_name="default")
                output_2 = model(...)
                ```

                By default, hotswap is disabled and requires passing `hotswap=True`. If you called
                `enable_peft_hotswap` first, it is enabled. You can still manually disable it in that case by passing
                `hotswap=False`.

                Note that hotswapping comes with a couple of limitations documented here:
                https://huggingface.co/docs/peft/main/en/package_reference/hotswap
            adapter_kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the `from_pretrained` method of the adapter config and
                `find_adapter_config_file` method.
        """
        from peft import PeftType
        from peft.utils.save_and_load import _maybe_shard_state_dict_for_tp

        from ..modeling_utils import LoadStateDictConfig, _get_resolved_checkpoint_files, load_state_dict

        if local_files_only:
            kwargs["local_files_only"] = True
        base_load_config = load_config.__dict__ if load_config is not None else {}
        base_load_config.update(kwargs)
        base_load_config.setdefault("pretrained_model_name_or_path", None)
        load_config = LoadStateDictConfig(**base_load_config)
        peft_model_id = peft_model_id or load_config.pretrained_model_name_or_path

        if hotswap == "auto":
            hotswap_enabled = getattr(self, "_hotswap_enabled", False)
            not_first_adapter = bool(self._hf_peft_config_loaded and (adapter_name in self.peft_config))
            hotswap = hotswap_enabled and not_first_adapter

        if hotswap:
            if (not self._hf_peft_config_loaded) or (adapter_name not in self.peft_config):
                raise ValueError(
                    "To hotswap an adapter, there must already be an existing adapter with the same adapter name."
                )
            if any(conf.peft_type != PeftType.LORA for conf in self.peft_config.values()):
                raise ValueError("Hotswapping is currently only supported for LoRA, please set `hotswap=False`.")

        adapter_name = adapter_name if adapter_name is not None else "default"
        adapter_kwargs = adapter_kwargs or {}

        from peft import PeftConfig, inject_adapter_in_model

        if self._hf_peft_config_loaded and (not hotswap) and (adapter_name in self.peft_config):
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")
        elif hotswap and ((not self._hf_peft_config_loaded) or (adapter_name not in self.peft_config)):
            raise ValueError(
                "To hotswap an adapter, there must already be an existing adapter with the same adapter name."
            )

        if peft_model_id is None and (adapter_state_dict is None and peft_config is None):
            raise ValueError(
                "You should either pass a `peft_model_id` or a `peft_config` and `adapter_state_dict` to load an adapter."
            )

        if peft_config is None:
            load_config.download_kwargs.update(**adapter_kwargs)
            adapter_config_file = find_adapter_config_file(
                peft_model_id,
                **load_config.download_kwargs,
            )

            if adapter_config_file is None:
                raise ValueError(
                    f"adapter model file not found in {peft_model_id}. Make sure you are passing the correct path to the "
                    "adapter model."
                )

            peft_config = PeftConfig.from_pretrained(
                peft_model_id,
                **load_config.download_kwargs,
            )

        from peft.utils.transformers_weight_conversion import build_peft_weight_mapping

        weight_conversions = load_config.weight_mapping or get_model_conversion_mapping(self)

        if hasattr(peft_config, "inference_mode"):
            peft_config.inference_mode = not is_trainable

        peft_weight_conversions = build_peft_weight_mapping(weight_conversions, adapter_name, peft_config=peft_config)

        if not hotswap:
            inject_adapter_in_model(peft_config, self, adapter_name)

        adapter_key_markers = {adapter_name}
        if peft_config is not None and getattr(peft_config, "peft_type", None) is not None:
            adapter_key_markers.add(peft_config.peft_type.value.lower())

        def is_adapter_key(key: str) -> bool:
            return any(marker in key for marker in adapter_key_markers)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True

        if adapter_state_dict is None:
            adapter_filenames = ["adapter_model.safetensors", "adapter_model.bin"]
            if load_config.use_safetensors is False:
                adapter_filenames.reverse()

            checkpoint_files = sharded_metadata = None
            last_error = None
            for adapter_filename in adapter_filenames:
                try:
                    checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
                        pretrained_model_name_or_path=peft_model_id,
                        variant=None,
                        gguf_file=None,
                        use_safetensors=(
                            load_config.use_safetensors if adapter_filename.endswith(".safetensors") else False
                        ),
                        user_agent=None,
                        is_remote_code=False,
                        transformers_explicit_filename=adapter_filename,
                        download_kwargs=load_config.download_kwargs,
                    )
                    break
                except OSError as error:
                    last_error = error

            if checkpoint_files is None:
                raise last_error or OSError("Could not download either a .bin or a .safetensors adapter file.")
        else:
            checkpoint_files, sharded_metadata = [], {}

        device_map = getattr(self, "hf_device_map", {"": self.device})

        has_tp_adapters = False
        for module in self.modules():
            tp_info = getattr(module, "_tp_info", None)
            if tp_info is not None:
                has_tp_adapters = True
                break

        if has_tp_adapters:
            all_pointer = set()
            if adapter_state_dict is not None:
                merged_state_dict = adapter_state_dict
            elif (
                checkpoint_files is not None
                and checkpoint_files[0].endswith(".safetensors")
                and adapter_state_dict is None
            ):
                merged_state_dict = {}
                for file in checkpoint_files:
                    file_pointer = safe_open(file, framework="pt", device="cpu")
                    all_pointer.add(file_pointer)
                    for k in file_pointer.keys():
                        merged_state_dict[k] = file_pointer.get_tensor(k)
            elif checkpoint_files is not None:
                merged_state_dict = {}
                for ckpt_file in checkpoint_files:
                    merged_state_dict.update(load_state_dict(ckpt_file))
            else:
                raise ValueError("Neither a state dict nor checkpoint files were found.")

            adapter_state_dict = merged_state_dict

            if any(not isinstance(v, torch.Tensor) for v in adapter_state_dict.values()):
                raise ValueError("Expected all values in the adapter state dict to be tensors.")

            _maybe_shard_state_dict_for_tp(self, adapter_state_dict, adapter_name)

        load_config = replace(
            load_config,
            pretrained_model_name_or_path=peft_model_id,
            sharded_metadata=sharded_metadata,
            weight_mapping=peft_weight_conversions,
            device_map=device_map,
        )

        loading_info, _ = self._load_pretrained_model(
            model=self,
            state_dict=adapter_state_dict,
            checkpoint_files=checkpoint_files,
            load_config=load_config,
            expected_keys=[n for n, _ in self.named_parameters() if is_adapter_key(n)],
        )

        if peft_config.inference_mode:
            from peft.tuners.tuners_utils import BaseTunerLayer

            self.eval()
            for module in self.modules():
                if isinstance(module, BaseTunerLayer):
                    module.requires_grad_(False)

        loading_info.missing_keys = {k for k in loading_info.missing_keys if is_adapter_key(k)}

        log_state_dict_report(
            model=self,
            pretrained_model_name_or_path=load_config.pretrained_model_name_or_path,
            ignore_mismatched_sizes=load_config.ignore_mismatched_sizes,
            loading_info=loading_info,
            logger=logger,
        )
        return loading_info

    def enable_peft_hotswap(
        self, target_rank: int = 128, check_compiled: Literal["error", "warn", "ignore"] = "error"
    ) -> None:
        pass

    def add_adapter(self, adapter_config, adapter_name: str | None = None) -> None:
        pass

    def set_adapter(self, adapter_name: list[str] | str) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

        Args:
            adapter_name (`Union[list[str], str]`):
                The name of the adapter to set. Can be also a list of strings to set multiple adapters.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.set_adapter(adapter_name)
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def disable_adapters(self) -> None:
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Disable all adapters that are attached to the model. This leads to inferring with the base model only.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled=False)

    def enable_adapters(self) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Enable adapters that are attached to the model.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled=True)

    def active_adapters(self) -> list[str]:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters
        for inference) returns the list of all active adapters so that users can deal with them accordingly.

        For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return
        a single string.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                active_adapters = module.active_adapter
                break

        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        return active_adapters

    def get_adapter_state_dict(self, adapter_name: str | None = None, state_dict: dict | None = None) -> dict:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter.
        If no adapter_name is passed, the active adapter is used.

        Args:
            adapter_name (`str`, *optional*):
                The name of the adapter to get the state dict from. If no name is passed, the active adapter is used.
            state_dict (nested dictionary of `torch.Tensor`, *optional*)
                The state dictionary of the model. Will default to `self.state_dict()`, but can be used if special
                precautions need to be taken when recovering the state dictionary of a model (like when using model
                parallelism).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft import get_peft_model_state_dict

        if adapter_name is None:
            adapter_name = self.active_adapters()[0]

        adapter_state_dict = get_peft_model_state_dict(self, state_dict=state_dict, adapter_name=adapter_name)
        return adapter_state_dict

    def _dispatch_accelerate_model(
        self,
        device_map: str,
        max_memory: int | None = None,
        offload_folder: str | None = None,
        offload_index: int | None = None,
    ) -> None:
        pass

    def delete_adapter(self, adapter_names: list[str] | str) -> None:
        pass


def maybe_load_adapters(
    pretrained_model_name_or_path,
    download_kwargs: DownloadKwargs,
    **adapter_kwargs,
):
    if pretrained_model_name_or_path is None or not is_peft_available():
        return None, pretrained_model_name_or_path, adapter_kwargs

    token = download_kwargs.get("token")

    if download_kwargs.get("commit_hash") is None:
        resolved_config_file = cached_file(
            pretrained_model_name_or_path,
            CONFIG_NAME,
            cache_dir=download_kwargs.get("cache_dir"),
            force_download=bool(download_kwargs.get("force_download", False)),
            proxies=download_kwargs.get("proxies"),
            local_files_only=bool(download_kwargs.get("local_files_only", False)),
            token=token,
            revision=download_kwargs.get("revision"),
            subfolder=download_kwargs.get("subfolder"),
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        download_kwargs["commit_hash"] = extract_commit_hash(resolved_config_file, None)

    _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

    token_from_adapter_kwargs = adapter_kwargs.pop("token", None)

    if _adapter_model_path is None:
        peft_kwargs = adapter_kwargs.copy()
        for arg_name in ("cache_dir", "proxies", "subfolder"):  # don't override revision
            if (arg_name not in peft_kwargs) and (arg_name in download_kwargs):
                peft_kwargs[arg_name] = download_kwargs[arg_name]
        if "commit_hash" in download_kwargs:
            peft_kwargs["_commit_hash"] = download_kwargs["commit_hash"]
        peft_kwargs["force_download"] = bool(download_kwargs.get("force_download", False))
        peft_kwargs["local_files_only"] = bool(download_kwargs.get("local_files_only", False))
        peft_kwargs["token"] = token or token_from_adapter_kwargs
        _adapter_model_path = find_adapter_config_file(
            pretrained_model_name_or_path,
            **peft_kwargs,
        )

    if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
        with open(_adapter_model_path, "r", encoding="utf-8") as f:
            _adapter_model_path = pretrained_model_name_or_path
            if not os.path.exists(pretrained_model_name_or_path) or not os.path.exists(
                os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
            ):
                pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]

    return _adapter_model_path, pretrained_model_name_or_path, adapter_kwargs
