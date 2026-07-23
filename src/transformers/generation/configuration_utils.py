
import copy
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, is_dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from .. import __version__
from ..utils import (
    GENERATION_CONFIG_NAME,
    ExplicitEnum,
    PushToHubMixin,
    cached_file,
    extract_commit_hash,
    hf_api,
    is_torch_available,
    logging,
)


if TYPE_CHECKING:
    import torch

    from ..configuration_utils import PreTrainedConfig
    from ..modeling_utils import PreTrainedModel


logger = logging.get_logger(__name__)
METADATA_FIELDS = ("_from_model_config", "_commit_hash", "_original_object_hash", "transformers_version")
STATIC_CACHE_IMPLEMENTATIONS = ("static", "offloaded_static")
DYNAMIC_CACHE_IMPLEMENTATIONS = ("dynamic", "offloaded", "quantized")
DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS = (
    "sliding_window",
    "hybrid",
    "hybrid_chunked",
    "offloaded_hybrid",
    "offloaded_hybrid_chunked",
)
ALL_STATIC_CACHE_IMPLEMENTATIONS = STATIC_CACHE_IMPLEMENTATIONS + DEPRECATED_STATIC_CACHE_IMPLEMENTATIONS
ALL_CACHE_IMPLEMENTATIONS = ALL_STATIC_CACHE_IMPLEMENTATIONS + DYNAMIC_CACHE_IMPLEMENTATIONS


if is_torch_available():
    from .logits_process import SynthIDTextWatermarkLogitsProcessor, WatermarkLogitsProcessor


def _should_warn(outer_attr: str, inner_attr: str, user_set_attributes: set | None) -> bool:
    """Determine if we should raise a warning for the combination `outer_attr` and `inner_attr`, based on whether
    they were provided explicitly, i.e. if they were in `user_set_attributes`.
    For example, if `outer_attr="do_sample"`, the warnings should be suppressed for `inner_attr` flags (e.g. "top_p") that weren't
    explicitly set by the caller. When `do_sample=False` is explicitly required by the user, values such as `top_p` inherited
    from a model's `generation_config.json` are harmless when the user opts for greedy decoding.
    """
    outer_sample_set = user_set_attributes is not None and outer_attr in user_set_attributes
    inner_attr_set = user_set_attributes is not None and inner_attr in user_set_attributes
    return (
        (outer_sample_set and inner_attr_set)
        or (not outer_sample_set and not inner_attr_set)
        or (inner_attr_set and not outer_sample_set)
    )


class GenerationMode(ExplicitEnum):

    CONTRASTIVE_SEARCH = "contrastive_search"
    GREEDY_SEARCH = "greedy_search"
    SAMPLE = "sample"
    ASSISTED_GENERATION = "assisted_generation"
    DOLA_GENERATION = "dola_generation"
    BEAM_SEARCH = "beam_search"
    BEAM_SAMPLE = "beam_sample"
    CONSTRAINED_BEAM_SEARCH = "constrained_beam_search"
    GROUP_BEAM_SEARCH = "group_beam_search"


class GenerationConfig(PushToHubMixin):

    extra_output_flags = ("output_attentions", "output_hidden_states", "output_scores", "output_logits")

    _bos_token_tensor: "torch.Tensor | None"
    _eos_token_tensor: "torch.Tensor | None"
    _pad_token_tensor: "torch.Tensor | None"
    _decoder_start_token_tensor: "torch.Tensor | None"

    _original_object_hash: int | None

    def __init__(self, **kwargs):
        user_set_attributes = set(kwargs.keys())

        self.max_length = kwargs.pop("max_length", None)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        self.min_length = kwargs.pop("min_length", None)
        self.min_new_tokens = kwargs.pop("min_new_tokens", None)
        self.early_stopping = kwargs.pop("early_stopping", None)
        self.max_time = kwargs.pop("max_time", None)
        self.stop_strings = kwargs.pop("stop_strings", None)

        self.do_sample = kwargs.pop("do_sample", None)
        self.num_beams = kwargs.pop("num_beams", None)
        self.use_mtp = kwargs.pop("use_mtp", None)

        self.use_cache = kwargs.pop("use_cache", None)
        self.cache_implementation = kwargs.pop("cache_implementation", None)
        self.cache_config = kwargs.pop("cache_config", None)
        self.max_cache_len = kwargs.pop("max_cache_len", None)

        self.temperature = kwargs.pop("temperature", None)
        self.top_k = kwargs.pop("top_k", None)
        self.top_p = kwargs.pop("top_p", None)
        self.min_p = kwargs.pop("min_p", None)
        self.top_h = kwargs.pop("top_h", None)
        self.typical_p = kwargs.pop("typical_p", None)
        self.epsilon_cutoff = kwargs.pop("epsilon_cutoff", None)
        self.eta_cutoff = kwargs.pop("eta_cutoff", None)
        self.repetition_penalty = kwargs.pop("repetition_penalty", None)
        self.encoder_repetition_penalty = kwargs.pop("encoder_repetition_penalty", None)
        self.length_penalty = kwargs.pop("length_penalty", None)
        self.no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", None)
        self.bad_words_ids = kwargs.pop("bad_words_ids", None)
        self.renormalize_logits = kwargs.pop("renormalize_logits", None)
        self.forced_bos_token_id = kwargs.pop("forced_bos_token_id", None)
        self.forced_eos_token_id = kwargs.pop("forced_eos_token_id", None)
        self.remove_invalid_values = kwargs.pop("remove_invalid_values", None)
        self.exponential_decay_length_penalty = kwargs.pop("exponential_decay_length_penalty", None)
        self.suppress_tokens = kwargs.pop("suppress_tokens", None)
        self.begin_suppress_tokens = kwargs.pop("begin_suppress_tokens", None)
        self.sequence_bias = kwargs.pop("sequence_bias", None)
        self.token_healing = kwargs.pop("token_healing", None)
        self.guidance_scale = kwargs.pop("guidance_scale", None)

        self.watermarking_config = kwargs.pop("watermarking_config", None)
        if isinstance(self.watermarking_config, dict):
            self.watermarking_config = WatermarkingConfig.from_dict(self.watermarking_config)

        self.num_return_sequences = kwargs.pop("num_return_sequences", None)
        self.output_attentions = kwargs.pop("output_attentions", None)
        self.output_hidden_states = kwargs.pop("output_hidden_states", None)
        self.output_scores = kwargs.pop("output_scores", None)
        self.output_logits = kwargs.pop("output_logits", None)
        self.return_dict_in_generate = kwargs.pop("return_dict_in_generate", None)

        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        self.encoder_no_repeat_ngram_size = kwargs.pop("encoder_no_repeat_ngram_size", None)
        self.decoder_start_token_id = kwargs.pop("decoder_start_token_id", None)

        self.is_assistant = kwargs.pop("is_assistant", None)
        self.num_assistant_tokens = kwargs.pop("num_assistant_tokens", None)
        self.num_assistant_tokens_schedule = kwargs.pop("num_assistant_tokens_schedule", None)
        self.assistant_confidence_threshold = kwargs.pop("assistant_confidence_threshold", None)
        self.prompt_lookup_num_tokens = kwargs.pop("prompt_lookup_num_tokens", None)
        self.max_matching_ngram_size = kwargs.pop("max_matching_ngram_size", None)
        self.assistant_early_exit = kwargs.pop("assistant_early_exit", None)
        self.assistant_lookbehind = kwargs.pop("assistant_lookbehind", None)
        self.target_lookbehind = kwargs.pop("target_lookbehind", None)
        self.assistant_ensemble_weight = kwargs.pop("assistant_ensemble_weight", None)

        self.compile_config = kwargs.pop("compile_config", None)
        self.disable_compile = kwargs.pop("disable_compile", None)

        self.continuous_batching_config = kwargs.pop("continuous_batching_config", None)

        self.low_memory = kwargs.pop("low_memory", None)
        self.penalty_alpha = kwargs.pop("penalty_alpha", None)
        self.dola_layers = kwargs.pop("dola_layers", None)
        self.diversity_penalty = kwargs.pop("diversity_penalty", None)
        self.num_beam_groups = kwargs.pop("num_beam_groups", None)
        self.constraints = kwargs.pop("constraints", None)
        self.force_words_ids = kwargs.pop("force_words_ids", None)

        self.prefill_chunk_size = kwargs.pop("prefill_chunk_size", None)

        self._commit_hash = kwargs.pop("_commit_hash", None)
        self._from_model_config = kwargs.pop("_from_model_config", None)
        self.transformers_version = kwargs.pop("transformers_version", None)

        if not self._from_model_config:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err
        else:
            if kwargs.get("force_bos_token_to_be_generated", False):
                self.forced_bos_token_id = self.bos_token_id
                logger.warning_once(
                    f"Please make sure the generation config includes `forced_bos_token_id={self.bos_token_id}`. "
                )

        self.validate(user_set_attributes=user_set_attributes)

    def __hash__(self):
        return hash(self.to_json_string(ignore_metadata=True))

    def __eq__(self, other):
        if not isinstance(other, GenerationConfig):
            return False

        self_without_metadata = self.to_json_string(use_diff=False, ignore_metadata=True)
        other_without_metadata = other.to_json_string(use_diff=False, ignore_metadata=True)
        return self_without_metadata == other_without_metadata

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string(ignore_metadata=True)}"

    def get_generation_mode(self, assistant_model: Optional["PreTrainedModel"] = None) -> GenerationMode:
        """
        Returns the generation mode triggered by the [`GenerationConfig`] instance.

        Arg:
            assistant_model (`PreTrainedModel`, *optional*):
                The assistant model to be used for assisted generation. If set, the generation mode will be
                assisted generation.

        Returns:
            `GenerationMode`: The generation mode triggered by the instance.
        """
        if self.constraints is not None or self.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif self.num_beams is None or self.num_beams == 1:
            if self.do_sample is not True:
                if (
                    self.top_k is not None
                    and self.top_k > 1
                    and self.penalty_alpha is not None
                    and self.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if self.num_beam_groups is not None and self.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif self.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        if (
            assistant_model is not None
            or self.use_mtp
            or self.prompt_lookup_num_tokens is not None
            or self.assistant_early_exit is not None
        ):
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                logger.warning(
                    "You've set `assistant_model`or `use_mtp`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample. However, the base decoding mode (based on "
                    f"current flags) is {generation_mode} -- some of the set flags will be ignored."
                )

        if self.dola_layers is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.DOLA_GENERATION
            else:
                logger.warning(
                    "You've set `dola_layers`, which triggers DoLa generate. Currently, DoLa generate "
                    "is only supported with Greedy Search and Sample.  However, the base decoding mode (based on "
                    f"current flags) is {generation_mode} -- some of the set flags will be ignored."
                )
        return generation_mode

    @staticmethod
    def _get_default_generation_params() -> dict[str, Any]:
        """
        Defaults to be applied when unset by the model OR by the user, such that `model.generate()` works with minimal
        parameterization.

        Pretrained checkpoints should set these as appropriate in their `generation_config.json`, to establish
        a better default baseline. Be mindful that tests will often use these values.
        """
        return {
            "max_length": 20,
            "min_length": 0,
            "do_sample": False,
            "use_cache": True,
            "early_stopping": False,
            "num_beams": 1,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "typical_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "encoder_no_repeat_ngram_size": 0,
            "bad_words_ids": None,
            "num_return_sequences": 1,
            "output_scores": False,
            "return_dict_in_generate": False,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "remove_invalid_values": False,
            "exponential_decay_length_penalty": None,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "epsilon_cutoff": 0.0,
            "eta_cutoff": 0.0,
            "encoder_repetition_penalty": 1.0,
            "num_assistant_tokens": 20,
            "num_assistant_tokens_schedule": "constant",
            "assistant_confidence_threshold": 0.4,
            "assistant_lookbehind": 10,
            "target_lookbehind": 10,
            "num_beam_groups": 1,
            "diversity_penalty": 0.0,
        }

    def validate(self, strict=False, user_set_attributes: set[str] | None = None):
        """
        Validates the values of the attributes of the [`GenerationConfig`] instance. Raises exceptions in the presence
        of parameterization that can be detected as incorrect from the configuration instance alone.

        Note that some parameters not validated here are best validated at generate runtime, as they may depend on
        other inputs and/or the model, such as parameters related to the generation length.

        Args:
            strict (bool): If True, raise an exception for any issues found. If False, only log issues.
            user_set_attributes (set[str], *optional*): Names of attributes the caller explicitly provided. When
                supplied, "minor issue" warnings about conflicting flag combinations (e.g. sampling-only flags set
                while `do_sample=False`) only fire if the conflicting flag is in this set -- avoiding noisy warnings
                when the value was inherited from a model's default `generation_config.json`. When `None`, all set
                attributes are considered user-set (backward-compatible behavior for direct `validate()` calls).
        """
        minor_issues = {}  # format: {attribute_name: issue_description}

        # 1.1. Decoding attributes
        if self.early_stopping not in {None, True, False, "never"}:
            raise ValueError(f"`early_stopping` must be a boolean or 'never', but is {self.early_stopping}.")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError(f"`max_new_tokens` must be greater than 0, but is {self.max_new_tokens}.")
        if self.assistant_ensemble_weight is not None and not (0.0 < self.assistant_ensemble_weight < 1.0):
            raise ValueError(
                f"`assistant_ensemble_weight` must be in the open interval `(0.0, 1.0)`, "
                f"but is {self.assistant_ensemble_weight}. Use `None` for standard (lossless) speculative decoding."
            )
        if self.pad_token_id is not None and self.pad_token_id < 0:
            minor_issues["pad_token_id"] = (
                f"`pad_token_id` should be positive but got {self.pad_token_id}. This will cause errors when batch "
                "generating, if there is padding. Please set `pad_token_id` explicitly as "
                "`model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation"
            )
        valid_cache_implementations = ALL_CACHE_IMPLEMENTATIONS + ("paged",)
        if self.cache_implementation is not None and self.cache_implementation not in valid_cache_implementations:
            raise ValueError(
                f"Invalid `cache_implementation` ({self.cache_implementation}). Choose one of: "
                f"{valid_cache_implementations}"
            )
        if self.max_cache_len is not None and self.cache_implementation not in ALL_STATIC_CACHE_IMPLEMENTATIONS:
            logger.warning_once(
                f"`max_cache_len` is only used with static caches ({STATIC_CACHE_IMPLEMENTATIONS}); it will be "
                f"ignored with `cache_implementation={self.cache_implementation!r}`."
            )
        if self.compile_config is not None and not isinstance(self.compile_config, CompileConfig):
            raise ValueError(
                f"You provided `compile_config` as an instance of {type(self.compile_config)}, but it must be an "
                "instance of `CompileConfig`."
            )
        if self.watermarking_config is not None:
            self.watermarking_config.validate()



        if self.do_sample is not True:
            greedy_wrong_parameter_msg = (
                "`do_sample` is not set to `True`. However, `{flag_name}` is set to `{flag_value}` -- this flag is "
                "only used in sample-based generation modes. You should set `do_sample=True` or unset `{flag_name}`."
            )

            if (
                self.temperature is not None
                and self.temperature != 1.0
                and _should_warn("do_sample", "temperature", user_set_attributes)
            ):
                minor_issues["temperature"] = greedy_wrong_parameter_msg.format(
                    flag_name="temperature", flag_value=self.temperature
                )
            if (
                self.top_p is not None
                and self.top_p != 1.0
                and _should_warn("do_sample", "top_p", user_set_attributes)
            ):
                minor_issues["top_p"] = greedy_wrong_parameter_msg.format(flag_name="top_p", flag_value=self.top_p)
            if self.min_p is not None and _should_warn("do_sample", "min_p", user_set_attributes):
                minor_issues["min_p"] = greedy_wrong_parameter_msg.format(flag_name="min_p", flag_value=self.min_p)
            if self.top_h is not None and _should_warn("do_sample", "top_h", user_set_attributes):
                minor_issues["top_h"] = greedy_wrong_parameter_msg.format(flag_name="top_h", flag_value=self.top_h)
            if (
                self.typical_p is not None
                and self.typical_p != 1.0
                and _should_warn("do_sample", "typical_p", user_set_attributes)
            ):
                minor_issues["typical_p"] = greedy_wrong_parameter_msg.format(
                    flag_name="typical_p", flag_value=self.typical_p
                )
            if self.top_k is not None and self.top_k != 50 and _should_warn("do_sample", "top_k", user_set_attributes):
                minor_issues["top_k"] = greedy_wrong_parameter_msg.format(flag_name="top_k", flag_value=self.top_k)
            if (
                self.epsilon_cutoff is not None
                and self.epsilon_cutoff != 0.0
                and _should_warn("do_sample", "epsilon_cutoff", user_set_attributes)
            ):
                minor_issues["epsilon_cutoff"] = greedy_wrong_parameter_msg.format(
                    flag_name="epsilon_cutoff", flag_value=self.epsilon_cutoff
                )
            if (
                self.eta_cutoff is not None
                and self.eta_cutoff != 0.0
                and _should_warn("do_sample", "eta_cutoff", user_set_attributes)
            ):
                minor_issues["eta_cutoff"] = greedy_wrong_parameter_msg.format(
                    flag_name="eta_cutoff", flag_value=self.eta_cutoff
                )

        if self.num_beams is None or self.num_beams == 1:
            single_beam_wrong_parameter_msg = (
                "`num_beams` is set to {num_beams}. However, `{flag_name}` is set to `{flag_value}` -- this flag is "
                "only used in beam-based generation modes. You should set `num_beams>1` or unset `{flag_name}`."
            )

            if (
                self.early_stopping is not None
                and self.early_stopping is not False
                and _should_warn("num_beams", "early_stopping", user_set_attributes)
            ):
                minor_issues["early_stopping"] = single_beam_wrong_parameter_msg.format(
                    num_beams=self.num_beams, flag_name="early_stopping", flag_value=self.early_stopping
                )
            if (
                self.length_penalty is not None
                and self.length_penalty != 1.0
                and _should_warn("num_beams", "length_penalty", user_set_attributes)
            ):
                minor_issues["length_penalty"] = single_beam_wrong_parameter_msg.format(
                    num_beams=self.num_beams, flag_name="length_penalty", flag_value=self.length_penalty
                )

        if self.num_return_sequences is not None and self.num_return_sequences > 1:
            if self.num_beams is None or self.num_beams == 1:
                if not self.do_sample:
                    raise ValueError(
                        "Greedy methods (do_sample != True) without beam search do not support "
                        f"`num_return_sequences` different than 1 (got {self.num_return_sequences})."
                    )
            elif (
                self.num_beams is not None
                and self.num_return_sequences is not None
                and self.num_return_sequences > self.num_beams
            ):
                raise ValueError(
                    f"`num_return_sequences` ({self.num_return_sequences}) has to be smaller or equal to `num_beams` "
                    f"({self.num_beams})."
                )

        if self.use_cache is False:
            no_cache_warning = (
                "You have not set `use_cache` to `True`, but {cache_arg} is set to {cache_arg_value}."
                "{cache_arg} will have no effect."
            )
            for arg_name in ("cache_implementation", "cache_config"):
                if getattr(self, arg_name) is not None:
                    minor_issues[arg_name] = no_cache_warning.format(
                        cache_arg=arg_name, cache_arg_value=getattr(self, arg_name)
                    )

        if self.return_dict_in_generate is not True:
            for extra_output_flag in self.extra_output_flags:
                if getattr(self, extra_output_flag) is True:
                    minor_issues[extra_output_flag] = (
                        f"`return_dict_in_generate` is NOT set to `True`, but `{extra_output_flag}` is. When "
                        f"`return_dict_in_generate` is not `True`, `{extra_output_flag}` is ignored."
                    )

        generate_arguments = (
            "logits_processor",
            "stopping_criteria",
            "prefix_allowed_tokens_fn",
            "synced_gpus",
            "assistant_model",
            "streamer",
            "negative_prompt_ids",
            "negative_prompt_attention_mask",
        )
        for arg in generate_arguments:
            if hasattr(self, arg):
                raise ValueError(
                    f"Argument `{arg}` is not a valid argument of `GenerationConfig`. It should be passed to "
                    "`generate()` (or a pipeline) directly."
                )

        if len(minor_issues) > 0:
            info_message = []
            for attribute_name, issue_description in minor_issues.items():
                info_message.append(f"- `{attribute_name}`: {issue_description}")
            info_message = "\n".join(info_message)
            info_message += (
                "\nIf you're using a pretrained model, note that some of these attributes may be set through the "
                "model's `generation_config.json` file."
            )

            if strict:
                raise ValueError("GenerationConfig is invalid: \n" + info_message)
            else:
                attributes_with_issues = list(minor_issues.keys())
                warning_message = (
                    f"The following generation flags are not valid and may be ignored: {attributes_with_issues}."
                )
                if logging.get_verbosity() >= logging.WARNING:
                    warning_message += " Set `TRANSFORMERS_VERBOSITY=info` for more details."
                logger.warning_once(warning_message)
                logger.info_once(info_message)

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        config_file_name: str | os.PathLike | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        r"""
        Save a generation configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~GenerationConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`):
                Name of the generation configuration JSON file to be saved in `save_directory`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        try:
            self.validate(strict=True)
        except ValueError as exc:
            raise ValueError(str(exc) + "\n\nFix these issues to save the configuration.")

        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", str(save_directory).split(os.path.sep)[-1])
            repo_id = hf_api().create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = os.path.join(save_directory, config_file_name)

        self.to_json_file(output_config_file, use_diff=True, keys_to_pop=["compile_config"])
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str | os.PathLike,
        config_file_name: str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> "GenerationConfig":
        r"""
        Instantiate a [`GenerationConfig`] from a generation configuration file.

        Args:
            pretrained_model_name (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~GenerationConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"generation_config.json"`):
                Name of the generation configuration JSON file to be loaded from `pretrained_model_name`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the configuration files and override the cached versions if
                they exist.
            proxies (`dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `hf auth login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.
            kwargs (`dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from this pretrained model.

        Examples:

        ```python
        >>> from transformers import GenerationConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> generation_config = GenerationConfig.from_pretrained("openai-community/gpt2")

        >>> # E.g. config was saved using *save_pretrained('./test/saved_model/')*
        >>> generation_config.save_pretrained("./test/saved_model/")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/")

        >>> # You can also specify configuration names to your generation configuration file
        >>> generation_config.save_pretrained("./test/saved_model/", config_file_name="my_configuration.json")
        >>> generation_config = GenerationConfig.from_pretrained("./test/saved_model/", "my_configuration.json")

        >>> # If you'd like to try a minor variation to an existing configuration, you can also pass generation
        >>> # arguments to `.from_pretrained()`. Be mindful that typos and unused arguments will be ignored
        >>> generation_config, unused_kwargs = GenerationConfig.from_pretrained(
        ...     "openai-community/gpt2", top_k=1, foo=False, do_sample=True, return_unused_kwargs=True
        ... )
        >>> generation_config.top_k
        1

        >>> unused_kwargs
        {'foo': False}
        ```"""
        config_file_name = config_file_name if config_file_name is not None else GENERATION_CONFIG_NAME

        proxies = kwargs.pop("proxies", None)
        subfolder = kwargs.pop("subfolder", "")
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        commit_hash = kwargs.pop("_commit_hash", None)

        user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        config_path = os.path.join(pretrained_model_name, config_file_name)
        config_path = str(config_path)

        is_local = os.path.exists(config_path)
        if os.path.isfile(os.path.join(subfolder, config_path)):
            resolved_config_file = config_path
            is_local = True
        else:
            configuration_file = config_file_name
            try:
                resolved_config_file = cached_file(
                    pretrained_model_name,
                    configuration_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _commit_hash=commit_hash,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            except OSError:
                raise
            except Exception:
                raise OSError(
                    f"Can't load the configuration of '{pretrained_model_name}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            config_dict = cls._dict_from_json_file(resolved_config_file)
            config_dict["_commit_hash"] = commit_hash
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise OSError(f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file.")

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        if kwargs.get("_from_model_config", False):
            return cls.from_model_config(config_dict)
        elif kwargs.get("return_unused_kwargs") is True:
            config, unused_kwargs = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(config)  # Hash to detect whether the instance was modified
            return config, unused_kwargs
        else:
            config = cls.from_dict(config_dict, **kwargs)
            config._original_object_hash = hash(config)  # Hash to detect whether the instance was modified
            return config

    @classmethod
    def _dict_from_json_file(cls, json_file: str | os.PathLike):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any], **kwargs) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        kwargs.pop("_from_auto", None)
        kwargs.pop("_from_pipeline", None)
        if "_commit_hash" in kwargs and "_commit_hash" in config_dict:
            kwargs["_commit_hash"] = config_dict["_commit_hash"]

        config = cls(**{**config_dict, **kwargs})
        unused_kwargs = config.update(**kwargs)

        logger.info(f"Generate config {config}")
        if return_unused_kwargs:
            return config, unused_kwargs
        else:
            return config

    def dict_dtype_to_str(self, d: dict[str, Any]) -> None:
        """
        Checks whether the passed dictionary and its nested dicts have a *dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype") is not None and not isinstance(d["dtype"], str):
            d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self.dict_dtype_to_str(value)

    def to_diff_dict(self) -> dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        config_dict = self.to_dict()

        default_config_dict = GenerationConfig().to_dict()

        serializable_config_dict = {}

        for key, value in config_dict.items():
            if key not in default_config_dict or key == "transformers_version" or value != default_config_dict[key]:
                serializable_config_dict[key] = value

        self.dict_dtype_to_str(serializable_config_dict)
        return serializable_config_dict

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)

        if "_commit_hash" in output:
            del output["_commit_hash"]
        if "_original_object_hash" in output:
            del output["_original_object_hash"]

        output["transformers_version"] = __version__

        self.dict_dtype_to_str(output)
        return output

    def to_json_string(
        self, use_diff: bool = True, ignore_metadata: bool = False, keys_to_pop: list[str] | None = None
    ) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON string.
            ignore_metadata (`bool`, *optional*, defaults to `False`):
                Whether to ignore the metadata fields present in the instance
            keys_to_pop (`list[str]`, *optional*):
                Keys to pop from the config dictionary before serializing

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()

        if keys_to_pop is not None:
            for key in keys_to_pop:
                config_dict.pop(key, None)

        if ignore_metadata:
            for metadata_field in METADATA_FIELDS:
                config_dict.pop(metadata_field, None)

        def convert_keys_to_string(obj):
            if isinstance(obj, dict):
                return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys_to_string(item) for item in obj]
            else:
                return obj

        def convert_dataclass_to_dict(obj):
            if isinstance(obj, dict):
                return {key: convert_dataclass_to_dict(value) for key, value in obj.items()}
            elif is_dataclass(obj):
                if hasattr(obj, "to_dict"):
                    return obj.to_dict()
            else:
                return obj

        config_dict = convert_keys_to_string(config_dict)
        config_dict = convert_dataclass_to_dict(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(
        self, json_file_path: str | os.PathLike, use_diff: bool = True, keys_to_pop: list[str] | None = None
    ) -> None:
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `GenerationConfig()`
                is serialized to JSON file.
            keys_to_pop (`list[str]`, *optional*):
                Keys to pop from the config dictionary before serializing
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string(use_diff=use_diff, keys_to_pop=keys_to_pop))

    @classmethod
    def from_model_config(cls, model_config: Union["PreTrainedConfig", dict]) -> "GenerationConfig":
        """
        Instantiates a [`GenerationConfig`] from a [`PreTrainedConfig`]. This function is useful to convert legacy
        [`PreTrainedConfig`] objects, which may contain generation parameters, into a stand-alone [`GenerationConfig`].

        Args:
            model_config (`PreTrainedConfig | dict`):
                The model config that will be used to instantiate the generation config.

        Returns:
            [`GenerationConfig`]: The configuration object instantiated from those parameters.
        """
        config_dict = model_config.to_dict() if not isinstance(model_config, dict) else model_config
        config_dict.pop("_from_model_config", None)

        config_dict = {key: value for key, value in config_dict.items() if value is not None}
        generation_config = cls.from_dict(config_dict, return_unused_kwargs=False, _from_model_config=True)

        if isinstance(model_config, dict):
            decoder_possible_text_config_names = ("decoder", "generator", "text_config")
            for text_config_name in decoder_possible_text_config_names:
                if text_config := model_config.get(text_config_name):
                    model_config = text_config
                    break
        else:
            model_config = model_config.get_text_config(decoder=True)
            model_config = model_config.to_dict()

        default_generation_config = GenerationConfig()
        for attr in generation_config.to_dict():
            is_unset = getattr(generation_config, attr) == getattr(default_generation_config, attr)
            if attr in model_config and is_unset:
                setattr(generation_config, attr, model_config[attr])

        if not generation_config.return_dict_in_generate:
            if any(
                getattr(generation_config, extra_output_flag, False)
                for extra_output_flag in generation_config.extra_output_flags
            ):
                generation_config.return_dict_in_generate = True

        generation_config._original_object_hash = hash(generation_config)
        return generation_config

    def update(self, defaults_only=False, allow_custom_entries=False, **kwargs):
        """
        Updates attributes of this class instance with attributes from `kwargs` if they match existing attributes,
        returning all the unused kwargs.

        Args:
            defaults_only (`bool`, *optional*, defaults to `False`):
                Whether to update all keys in config with `kwargs` or only those that are set to `None` (i.e. default value).
            allow_custom_entries (`bool`, *optional*, defaults to `False`):
                Whether to allow updating custom entries into the config with `kwargs` if not present in the current config.
            kwargs (`dict[str, Any]`):
                Dictionary of attributes to tentatively update this class.

        Returns:
            `dict[str, Any]`: Dictionary containing all the key-value pairs that were not used to update the instance.
        """
        to_remove = []
        for key, value in kwargs.items():
            if allow_custom_entries and not hasattr(self, key):
                setattr(self, key, value)
                to_remove.append(key)
            elif hasattr(self, key):
                if not defaults_only or getattr(self, key) is None:
                    setattr(self, key, value)
                    to_remove.append(key)

        self.validate(user_set_attributes=set(to_remove))

        unused_kwargs = {key: value for key, value in kwargs.items() if key not in to_remove}
        return unused_kwargs


@dataclass
class BaseWatermarkingConfig(ABC):

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a BaseWatermarkingConfig instance from a dictionary of parameters.

        Args:
            config_dict (dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            BaseWatermarkingConfig: Instance of BaseWatermarkingConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    def to_json_file(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (Union[str, os.PathLike]): Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def __iter__(self):
        yield from copy.deepcopy(self.__dict__).items()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.

        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    def update(self, **kwargs):
        """
        Update the configuration attributes with new values.

        Args:
            **kwargs: Keyword arguments representing configuration attributes and their new values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def validate(self): ...

    @abstractmethod
    def construct_processor(self, vocab_size): ...


@dataclass
class WatermarkingConfig(BaseWatermarkingConfig):

    def __init__(
        self,
        greenlist_ratio: float = 0.25,
        bias: float = 2.0,
        hashing_key: int = 15485863,
        seeding_scheme: str = "lefthash",
        context_width: int = 1,
    ):
        self.greenlist_ratio = greenlist_ratio
        self.bias = bias
        self.hashing_key = hashing_key
        self.seeding_scheme = seeding_scheme
        self.context_width = context_width

    def validate(self):
        watermark_missing_arg_msg = (
            "Some of the keys in `watermarking_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.seeding_scheme not in ["selfhash", "lefthash"]:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="seeding_scheme",
                    correct_value="[`selfhash`, `lefthash`]",
                    found_value=self.seeding_scheme,
                ),
            )
        if not 0.0 <= self.greenlist_ratio <= 1.0:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="greenlist_ratio",
                    correct_value="in range between 0.0 and 1.0",
                    found_value=self.seeding_scheme,
                ),
            )
        if not self.context_width >= 1:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="context_width",
                    correct_value="a positive integer",
                    found_value=self.context_width,
                ),
            )

    def construct_processor(self, vocab_size: int, device) -> "WatermarkLogitsProcessor":
        return WatermarkLogitsProcessor(
            vocab_size=vocab_size,
            device=device,
            greenlist_ratio=self.greenlist_ratio,
            bias=self.bias,
            hashing_key=self.hashing_key,
            seeding_scheme=self.seeding_scheme,
            context_width=self.context_width,
        )


@dataclass
class SynthIDTextWatermarkingConfig(BaseWatermarkingConfig):

    def __init__(
        self,
        ngram_len: int,
        keys: list[int],
        context_history_size: int = 1024,
        sampling_table_seed: int = 0,
        sampling_table_size: int = 2**16,
        skip_first_ngram_calls: bool = False,
        debug_mode: bool = False,
    ):
        self.ngram_len = ngram_len
        self.keys = keys
        self.sampling_table_size = sampling_table_size
        self.sampling_table_seed = sampling_table_seed
        self.context_history_size = context_history_size
        self.skip_first_ngram_calls = skip_first_ngram_calls
        self.debug_mode = debug_mode

    def validate(self):
        watermark_missing_arg_msg = (
            "Some of the keys in `watermarking_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.sampling_table_size > 2**24:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="sampling_table_size",
                    correct_value="< 2**24",
                    found_value=self.sampling_table_size,
                ),
            )

    def construct_processor(self, vocab_size: int, device) -> "WatermarkLogitsProcessor":
        return SynthIDTextWatermarkLogitsProcessor(
            ngram_len=self.ngram_len,
            keys=self.keys,
            sampling_table_size=self.sampling_table_size,
            sampling_table_seed=self.sampling_table_seed,
            context_history_size=self.context_history_size,
            device=device,
            skip_first_ngram_calls=self.skip_first_ngram_calls,
            debug_mode=self.debug_mode,
        )


@dataclass
class CompileConfig:

    fullgraph: bool = False
    dynamic: bool | None = None
    backend: str | Callable = "inductor"
    mode: str = "reduce-overhead"
    options: dict | None = None
    _compile_all_devices = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes this instance to a Python dictionary."""
        return copy.deepcopy({key: value for key, value in self.__dict__.items() if key != "_compile_all_devices"})


@dataclass
class ContinuousBatchingConfig:

    block_size: int = 256

    num_blocks: int | None = None
    max_batch_tokens: int | None = None

    max_memory_percent: float | None = None

    max_requests_per_batch: int | None = None

    max_blocks_per_request: int | None = None

    allow_block_sharing: bool = True

    use_async_batching: bool | None = None

    use_cuda_graph: bool | tuple[bool, bool] | None = None

    q_padding_interval_size: int = 0
    kv_padding_interval_size: int = 0

    varlen_compile_config: CompileConfig | None = None
    decode_compile_config: CompileConfig | None = None
    default_compile_level: int = 0

    scheduler_type: str = "fifo"
    safety_margin: float | None = None

    return_logprobs: bool = False

    seed: int | None = None

    cpu_offload_space: float | None = 0.0
    cpu_offload_space_safety_threshold: float = 0.8

    max_queue_size: int = 0

    per_request_processors: bool = False
    drop_unsupported_processors: bool = True

    disable_nccl_graph_mixing: bool = True

    cpu_group_timeout: float | None = 300.0

    use_default_compile_configs: bool | None = None
    max_cached_graphs: int | None = None

    def __post_init__(self):
        graph_mixing_supported = os.environ.get("NCCL_GRAPH_MIXING_SUPPORT", "1") == "1"
        distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if self.disable_nccl_graph_mixing and graph_mixing_supported and distributed:
            logger.warning(
                "Setting NCCL_GRAPH_MIXING_SUPPORT = 0 because disable_nccl_graph_mixing is True and WORLD_SIZE > 1."
            )
            os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
        if self.use_default_compile_configs is not None:  # Deprecated in 5.11
            if self.use_default_compile_configs:
                level_msg = "setting default_compile_level to 3. Consider using a lower level for faster warmup time."
                self.default_compile_level = 3
            else:
                level_msg = "setting default_compile_level to 0."
                self.default_compile_level = 0
            logger.warning(
                "use_default_compile_configs is deprecated: please use default_compile_level instead. For backwards "
                f"compatibility, {level_msg}"
            )
        if self.max_cached_graphs is not None:  # Deprecated in 5.13
            logger.warning(
                "max_cached_graphs is deprecated: maximum number of graph is no longer an issue. Deprecated in 5.13."
            )

    @property
    def cuda_graph_booleans(self) -> tuple[bool, bool]:
        pass

    @property
    def fallback_max_blocks_per_request(self) -> int:
        pass
