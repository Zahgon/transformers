
import bisect
import copy
import functools
import inspect
import json
import os
import re
import sys
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict, TypeVar, Union

import numpy as np
import typing_extensions
from huggingface_hub import is_offline_mode
from huggingface_hub.dataclasses import validate_typed_dict
from huggingface_hub.errors import EntryNotFoundError

from .audio_utils import AudioInput, load_audio, make_list_of_audio
from .dynamic_module_utils import custom_object_save
from .feature_extraction_utils import BatchFeature
from .image_utils import ChannelDimension, ImageInput, is_vision_available, make_flat_list_of_images
from .tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from .utils import (
    AUDIO_TOKENIZER_NAME,
    CHAT_TEMPLATE_DIR,
    CHAT_TEMPLATE_FILE,
    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
    PROCESSOR_NAME,
    PushToHubMixin,
    TensorType,
    auto_docstring,
    cached_file,
    copy_func,
    direct_transformers_import,
    hf_api,
    is_torch_available,
    list_repo_templates,
    logging,
)
from .utils.chat_template_utils import _get_template_variables, render_jinja_template
from .utils.type_validators import (
    device_validator,
    image_size_validator,
    padding_validator,
    positive_any_number,
    positive_int,
    resampling_validator,
    tensor_type_validator,
    truncation_validator,
    video_metadata_validator,
)
from .video_utils import VideoInput, VideoMetadataType, make_batched_videos


if is_torch_available():
    import torch

    from .modeling_utils import PreTrainedAudioTokenizerBase

if is_vision_available():
    from .image_utils import PILImageResampling

logger = logging.get_logger(__name__)

SpecificProcessorType = TypeVar("SpecificProcessorType", bound="ProcessorMixin")

transformers_module = direct_transformers_import(Path(__file__).parent)


class _LazyAutoProcessorMapping(dict):

    _MAPPING_NAMES = {
        "image_processor": ("transformers.models.auto.image_processing_auto", "AutoImageProcessor"),
        "video_processor": ("transformers.models.auto.video_processing_auto", "AutoVideoProcessor"),
        "feature_extractor": ("transformers.models.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "audio_processor": ("transformers.models.auto.feature_extraction_auto", "AutoFeatureExtractor"),
        "tokenizer": ("transformers.models.auto.tokenization_auto", "AutoTokenizer"),
    }

    def __getitem__(self, key):
        if key not in self._MAPPING_NAMES:
            raise KeyError(key)
        module_name, attr_name = self._MAPPING_NAMES[key]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)

    def __contains__(self, key):
        return key in self._MAPPING_NAMES

    def keys(self):
        return self._MAPPING_NAMES.keys()


MODALITY_TO_AUTOPROCESSOR_MAPPING = _LazyAutoProcessorMapping()

MODALITY_TO_BASE_CLASS_MAPPING = {
    "audio_tokenizer": (
        "HiggsAudioV2TokenizerModel",
        "DacModel",
    ),  # TODO: @eustlb, to be replaced with PreTrainedAudioTokenizerBase
    "audio_processor": "FeatureExtractionMixin",
    "tokenizer": ("PreTrainedTokenizerBase", "MistralCommonBackend"),
    "feature_extractor": "FeatureExtractionMixin",
    "image_processor": "ImageProcessingMixin",
    "video_processor": "BaseVideoProcessor",
}


def _get_modality_for_attribute(attribute_name: str) -> str:
    """
    Get the canonical modality type for a given attribute name.

    For example:
    - "image_processor" -> "image_processor"
    - "encoder_image_processor" -> "image_processor"
    - "text_tokenizer" -> "tokenizer"
    - "my_feature_extractor" -> "feature_extractor"
    """
    for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING.keys():
        if modality in attribute_name:
            return modality
    raise ValueError(
        f"Cannot determine modality for attribute '{attribute_name}'. "
        f"Attribute name must contain one of: {list(MODALITY_TO_AUTOPROCESSOR_MAPPING.keys())}"
    )


if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack


class TextKwargs(TypedDict, total=False):

    text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None
    add_special_tokens: bool | None
    padding: Annotated[bool | str | PaddingStrategy | None, padding_validator()]
    truncation: Annotated[bool | str | TruncationStrategy | None, truncation_validator()]
    max_length: Annotated[int | None, positive_int()]
    stride: Annotated[int | None, positive_int()]
    is_split_into_words: bool | None
    pad_to_multiple_of: Annotated[int | None, positive_int()]
    return_token_type_ids: bool | None
    return_attention_mask: bool | None
    return_overflowing_tokens: bool | None
    return_special_tokens_mask: bool | None
    return_offsets_mapping: bool | None
    return_length: bool | None
    verbose: bool | None
    padding_side: Literal["left", "right"] | None
    return_mm_token_type_ids: bool | None
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]


class ImagesKwargs(TypedDict, total=False):

    do_convert_rgb: bool | None
    do_resize: bool | None
    size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    default_to_square: bool | None
    crop_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    resample: Annotated[Union["PILImageResampling", int] | None, resampling_validator()]
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | tuple[float, ...] | None
    image_std: float | list[float] | tuple[float, ...] | None
    do_pad: bool | None
    pad_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    do_center_crop: bool | None
    data_format: str | ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: Annotated[Union[str, "torch.device"] | None, device_validator()]
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]
    disable_grouping: bool | None
    image_seq_length: int | None


class VideosKwargs(TypedDict, total=False):

    do_convert_rgb: bool | None
    do_resize: bool | None
    size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    default_to_square: bool | None
    resample: Annotated[Union["PILImageResampling", int] | None, resampling_validator()]
    do_rescale: bool | None
    rescale_factor: float | None
    do_normalize: bool | None
    image_mean: float | list[float] | tuple[float, ...] | None
    image_std: float | list[float] | tuple[float, ...] | None
    do_center_crop: bool | None
    do_pad: bool | None
    crop_size: Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, image_size_validator()]
    data_format: str | ChannelDimension | None
    input_data_format: str | ChannelDimension | None
    device: Annotated[Union[str, "torch.device"] | None, device_validator()]
    do_sample_frames: bool | None
    video_metadata: Annotated[VideoMetadataType | None, video_metadata_validator()]
    fps: Annotated[int | float | None, positive_any_number()]
    num_frames: Annotated[int | None, positive_int()]
    return_metadata: bool | None
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]


class AudioKwargs(TypedDict, total=False):

    sampling_rate: Annotated[int | None, positive_int()]
    raw_speech: Union["np.ndarray", list[float], list["np.ndarray"], list[list[float]]] | None
    padding: Annotated[bool | str | PaddingStrategy | None, padding_validator()]
    max_length: Annotated[int | None, positive_int()]
    truncation: Annotated[bool | str | TruncationStrategy | None, truncation_validator()]
    pad_to_multiple_of: Annotated[int | None, positive_int()]
    return_attention_mask: bool | None
    return_tensors: Annotated[str | TensorType | None, tensor_type_validator()]
    load_audio_backend: str | None


class ProcessingKwargs(TypedDict, total=False):

    _defaults = {}

    text_kwargs: TextKwargs = {
        **TextKwargs.__annotations__,
    }
    images_kwargs: ImagesKwargs = {
        **ImagesKwargs.__annotations__,
    }
    videos_kwargs: VideosKwargs = {
        **VideosKwargs.__annotations__,
    }
    audio_kwargs: AudioKwargs = {
        **AudioKwargs.__annotations__,
    }


class TokenizerChatTemplateKwargs(TypedDict, total=False):

    tools: list[dict] | None = None
    documents: list[dict[str, str]] | None = None
    add_generation_prompt: bool | None = False
    continue_final_message: bool | str | None = False
    return_assistant_tokens_mask: bool | None = False
    reasoning_effort: str | None = None


class ProcessorChatTemplateKwargs(TokenizerChatTemplateKwargs, total=False):

    tokenize: bool | None = False
    return_dict: bool | None = False
    load_audio_from_video: bool | None = False


class AllKwargsForChatTemplate(TypedDict, total=False):

    processor_kwargs: ProcessingKwargs
    template_kwargs: ProcessorChatTemplateKwargs


@dataclass
class MultiModalData:

    num_image_tokens: list[int] | None = None
    num_video_tokens: list[int] | None = None
    num_audio_tokens: list[int] | None = None
    num_image_patches: list[int] | None = None

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")


@functools.lru_cache(maxsize=8)
def _merge_typed_dict(preprocessor_typed_dict: type, modality_typed_dict: type) -> type:
    return TypedDict(
        "merged_typed_dict",
        {**preprocessor_typed_dict.__annotations__, **modality_typed_dict.__annotations__},
        total=False,
    )


class ProcessorMixin(PushToHubMixin):

    tokenizer: Any
    feature_extractor: Any
    image_processor: Any
    video_processor: Any
    chat_template: str | dict[str, str] | None

    _auto_class = None
    valid_processor_kwargs = ProcessingKwargs
    skip_tensor_conversion = ["video_metadata", "text_replacement_offsets"]

    def __init__(self, *args, **kwargs):
        setattr(self, "chat_template", kwargs.pop("chat_template", None))

        if (audio_tokenizer := kwargs.pop("audio_tokenizer", None)) is not None:
            proper_class = self.check_argument_for_proper_class("audio_tokenizer", audio_tokenizer)
            if not (is_torch_available() and isinstance(audio_tokenizer, PreTrainedAudioTokenizerBase)):
                raise ValueError(
                    f"Tried to use `{proper_class}` for audio tokenization. However, this class is not"
                    " registered for audio tokenization."
                )
            setattr(self, "audio_tokenizer", audio_tokenizer)

        for key in kwargs:
            if key not in self.get_attributes():
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.get_attributes()):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.get_attributes()):
            raise ValueError(
                f"This processor requires {len(self.get_attributes())} arguments: {', '.join(self.get_attributes())}. Got "
                f"{len(args)} arguments instead."
            )

        for attribute_name, arg in kwargs.items():
            self.check_argument_for_proper_class(attribute_name, arg)
            setattr(self, attribute_name, arg)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        images, text, videos, audio = self.prepare_inputs_layout(
            images=images, text=text, videos=videos, audio=audio, **kwargs
        )
        self.validate_inputs(images=images, text=text, videos=videos, audio=audio, **kwargs)

        merged_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs if hasattr(self, "tokenizer") else {},
            **kwargs,
        )

        processed_images = processed_videos = processed_audio = {}
        images_replacements = videos_replacements = audio_replacements = []
        if images is not None and hasattr(self, "image_processor"):
            processed_images, images_replacements = self._process_images(images, **merged_kwargs["images_kwargs"])
        if videos is not None and hasattr(self, "video_processor"):
            processed_videos, videos_replacements = self._process_videos(videos, **merged_kwargs["videos_kwargs"])
        if audio is not None and hasattr(self, "feature_extractor"):
            processed_audio, audio_replacements = self._process_audio(audio, **merged_kwargs["audio_kwargs"])

        text_inputs = {}
        return_tensors = merged_kwargs["text_kwargs"].get("return_tensors", None)
        if getattr(self, "tokenizer", None) is not None and text is not None:
            return_mm_token_type_ids = merged_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
            return_text_replacement_offsets = merged_kwargs["text_kwargs"].pop(
                "return_text_replacement_offsets", False
            )

            text, text_replacement_offsets = self.get_text_with_replacements(
                text,
                images_replacements,
                videos_replacements,
                audio_replacements,
            )
            text_inputs = self.tokenizer(text, **merged_kwargs["text_kwargs"])
            self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video", "audio"])

            if return_text_replacement_offsets:
                text_inputs["text_replacement_offsets"] = text_replacement_offsets

            if return_mm_token_type_ids:
                text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])

        data = {**text_inputs, **processed_images, **processed_videos, **processed_audio}
        data = {k: v for k, v in data.items() if k not in self.unused_input_names}

        if not kwargs.get("return_metadata"):
            data.pop("video_metadata", None)

        return BatchFeature(data, tensor_type=return_tensors, skip_tensor_conversion=self.skip_tensor_conversion)

    def prepare_inputs_layout(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        pass

    def validate_inputs(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None,
        videos: VideoInput | None = None,
        audio: AudioInput | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        pass

    def _process_images(self, images: ImageInput, **kwargs):
        pass

    def _process_videos(self, videos: VideoInput, **kwargs):
        pass

    def _process_audio(self, audio: AudioInput, **kwargs):
        pass

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        raise NotImplementedError

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        raise NotImplementedError

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        raise NotImplementedError

    def get_text_with_replacements(
        self,
        text: list[str],
        images_replacements: list[str] = [],
        videos_replacements: list[str] = [],
        audio_replacements: list[str] = [],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        pass

    def create_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        pass

    @property
    def all_special_multimodal_tokens(self) -> list[str]:
        pass

    @property
    def image_token_ids(self) -> list[int | None]:
        pass

    @image_token_ids.setter
    def image_token_ids(self, value: list[int | None]):
        pass

    @property
    def video_token_ids(self) -> list[int | None]:
        pass

    @video_token_ids.setter
    def video_token_ids(self, value: list[int | None]):
        pass

    @property
    def audio_token_ids(self) -> list[int | None]:
        pass

    @audio_token_ids.setter
    def audio_token_ids(self, value: list[int | None]):
        pass

    def check_argument_for_proper_class(self, argument_name, argument):
        """
        Checks the passed argument's class against the expected transformers class. In case of an unexpected
        mismatch between expected and actual class, an error is raise. Otherwise, the proper retrieved class
        is returned.
        """
        if argument_name not in MODALITY_TO_BASE_CLASS_MAPPING:
            argument_name = _get_modality_for_attribute(argument_name)
        class_name = MODALITY_TO_BASE_CLASS_MAPPING.get(argument_name)
        if isinstance(class_name, tuple):
            proper_class = tuple(self.get_possibly_dynamic_module(n) for n in class_name if n is not None)
        else:
            proper_class = self.get_possibly_dynamic_module(class_name)

        if not isinstance(argument, proper_class):
            raise TypeError(
                f"Received a {type(argument).__name__} for argument {argument_name}, but a {class_name} was expected."
            )

        return proper_class

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        tokenizer_attributes = set()
        for attribute in self.__class__.get_attributes():
            if attribute in self.__dict__:
                modality = _get_modality_for_attribute(attribute)
                if modality == "tokenizer":
                    tokenizer_attributes.add(attribute)

        dict_to_copy = {k: v for k, v in self.__dict__.items() if k not in tokenizer_attributes}
        output = copy.deepcopy(dict_to_copy)

        sig = inspect.signature(self.__init__)
        attrs_to_save = list(sig.parameters) + self.__class__.get_attributes()
        attrs_to_save += ["auto_map"]

        if "chat_template" in output:
            del output["chat_template"]

        def cast_array_to_list(dictionary):
            """
            Numpy arrays are not serialiazable but can be in pre-processing dicts.
            This function casts arrays to list, recusring through the nested configs as well.
            """
            for key, value in dictionary.items():
                if isinstance(value, np.ndarray):
                    dictionary[key] = value.tolist()
                elif isinstance(value, dict):
                    dictionary[key] = cast_array_to_list(value)
            return dictionary

        if "audio_tokenizer" in output:
            audio_tokenizer_dict = {
                "audio_tokenizer_class": self.audio_tokenizer.__class__.__name__,
                "audio_tokenizer_name_or_path": self.audio_tokenizer.name_or_path,
            }
            output["audio_tokenizer"] = audio_tokenizer_dict

        output = {
            k: v.to_dict() if isinstance(v, PushToHubMixin) else v
            for k, v in output.items()
            if (
                k in attrs_to_save  # keep all attributes that have to be serialized
                and v.__class__.__name__ != "BeamSearchDecoderCTC"  # remove attributes with that are objects
            )
        }
        output = cast_array_to_list(output)
        output["processor_class"] = self.__class__.__name__

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.get_attributes()]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}\n\n{self.to_json_string()}"

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`]. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = hf_api().create_repo(repo_id, exist_ok=True, **kwargs).repo_id
            files_timestamps = self._get_files_timestamps(save_directory)
        if self._auto_class is not None:
            attrs = [getattr(self, attribute_name) for attribute_name in self.get_attributes()]
            configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a) for a in attrs]
            configs.append(self)
            custom_object_save(self, save_directory, config=configs)

        for attribute_name in self.get_attributes():
            attribute = getattr(self, attribute_name)

            modality = _get_modality_for_attribute(attribute_name)
            is_primary = attribute_name == modality
            if modality == "tokenizer":
                attribute._set_processor_class(self.__class__.__name__)
                if is_primary:
                    attribute.save_pretrained(save_directory)
                else:
                    attribute.save_pretrained(os.path.join(save_directory, attribute_name))
            elif attribute._auto_class is not None:
                custom_object_save(attribute, save_directory, config=attribute)

        if self._auto_class is not None:
            for attribute_name in self.get_attributes():
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        output_chat_template_file_jinja = os.path.join(save_directory, CHAT_TEMPLATE_FILE)
        chat_template_dir = os.path.join(save_directory, CHAT_TEMPLATE_DIR)

        if isinstance(self.chat_template, str):
            with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                f.write(self.chat_template)
            logger.info(f"chat template saved in {output_chat_template_file_jinja}")
        elif isinstance(self.chat_template, dict):
            for template_name, template in self.chat_template.items():
                if template_name == "default":
                    with open(output_chat_template_file_jinja, "w", encoding="utf-8") as f:
                        f.write(self.chat_template["default"])
                    logger.info(f"chat template saved in {output_chat_template_file_jinja}")
                else:
                    os.makedirs(chat_template_dir, exist_ok=True)
                    template_filepath = os.path.join(chat_template_dir, f"{template_name}.jinja")
                    if Path(template_filepath).resolve().parent != Path(chat_template_dir).resolve():
                        raise ValueError(f"Invalid chat template name: {template_name!r}")
                    with open(template_filepath, "w", encoding="utf-8") as f:
                        f.write(template)
                    logger.info(f"chat template saved in {template_filepath}")

        self.to_json_file(output_processor_file)
        logger.info(f"processor saved in {output_processor_file}")
        return_files = [output_processor_file]

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return return_files

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: str | os.PathLike, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processing_utils.ProcessingMixin`] using `from_args_and_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        audio_tokenizer_kwargs = copy.deepcopy(kwargs)

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_NAME)

        additional_chat_template_files = {}
        resolved_additional_chat_template_files = {}
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
            resolved_audio_tokenizer_file = None
            is_local = True
        else:
            if is_local:
                template_dir = Path(pretrained_model_name_or_path, CHAT_TEMPLATE_DIR)
                if template_dir.is_dir():
                    for template_file in template_dir.glob("*.jinja"):
                        template_name = template_file.stem
                        additional_chat_template_files[template_name] = f"{CHAT_TEMPLATE_DIR}/{template_file.name}"
            else:
                try:
                    for template in list_repo_templates(
                        pretrained_model_name_or_path,
                        local_files_only=local_files_only,
                        revision=revision,
                        cache_dir=cache_dir,
                        token=token,
                    ):
                        template = template.removesuffix(".jinja")
                        additional_chat_template_files[template] = f"{CHAT_TEMPLATE_DIR}/{template}.jinja"
                except EntryNotFoundError:
                    pass  # No template dir means no template files
            processor_file = PROCESSOR_NAME

            try:
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_raw_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    CHAT_TEMPLATE_FILE,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_additional_chat_template_files = {
                    template_name: cached_file(
                        pretrained_model_name_or_path,
                        template_file,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        user_agent=user_agent,
                        revision=revision,
                        subfolder=subfolder,
                        _raise_exceptions_for_missing_entries=False,
                    )
                    for template_name, template_file in additional_chat_template_files.items()
                }

                resolved_audio_tokenizer_file = cached_file(
                    pretrained_model_name_or_path,
                    AUDIO_TOKENIZER_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
            except OSError:
                raise
            except Exception:
                raise OSError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {PROCESSOR_NAME} file"
                )

        if resolved_chat_template_file is not None:
            with open(resolved_chat_template_file, encoding="utf-8") as reader:
                chat_template_json = json.loads(reader.read())
                chat_templates = {"default": chat_template_json["chat_template"]}
                if resolved_additional_chat_template_files:
                    raise ValueError(
                        "Cannot load chat template due to conflicting files - this checkpoint combines "
                        "a legacy chat_template.json file with separate template files, which is not "
                        "supported. To resolve this error, replace the legacy chat_template.json file "
                        "with a modern chat_template.jinja file."
                    )
        else:
            chat_templates = {
                template_name: open(template_file, "r", encoding="utf-8").read()
                for template_name, template_file in resolved_additional_chat_template_files.items()
            }
            if resolved_raw_chat_template_file is not None:
                with open(resolved_raw_chat_template_file, "r", encoding="utf-8") as reader:
                    chat_templates["default"] = reader.read()
        if isinstance(chat_templates, dict) and "default" in chat_templates and len(chat_templates) == 1:
            chat_templates = chat_templates["default"]  # Flatten when we just have a single template/file

        if resolved_processor_file is None:
            processor_dict = {}
        else:
            try:
                with open(resolved_processor_file, encoding="utf-8") as reader:
                    text = reader.read()
                processor_dict = json.loads(text)

            except json.JSONDecodeError:
                raise OSError(
                    f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
                )

        if is_local:
            logger.info(f"loading configuration file {resolved_processor_file}")
        else:
            logger.info(f"loading configuration file {processor_file} from cache at {resolved_processor_file}")

        if processor_dict.get("chat_template") is not None:
            logger.warning_once(
                "Chat templates should be in a 'chat_template.jinja' file but found key='chat_template' "
                "in the processor's config. Make sure to move your template to its own file."
            )
        elif chat_templates:
            processor_dict["chat_template"] = chat_templates

        if resolved_audio_tokenizer_file is not None or "audio_tokenizer" in processor_dict:
            if resolved_audio_tokenizer_file is not None:
                reader = open(resolved_audio_tokenizer_file, "r", encoding="utf-8")
                audio_tokenizer_dict = reader.read()
                audio_tokenizer_dict = json.loads(audio_tokenizer_dict)
            else:
                audio_tokenizer_dict = processor_dict["audio_tokenizer"]

            audio_tokenizer_class = cls.get_possibly_dynamic_module(audio_tokenizer_dict["audio_tokenizer_class"])
            audio_tokenizer_path = audio_tokenizer_dict["audio_tokenizer_name_or_path"]
            processor_dict["audio_tokenizer"] = audio_tokenizer_class.from_pretrained(
                audio_tokenizer_path, **audio_tokenizer_kwargs
            )

        return processor_dict, kwargs

    @classmethod
    def from_args_and_dict(cls, args, processor_dict: dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        for unused_kwarg in cls.get_attributes() + ["auto_map", "processor_class"]:
            processor_dict.pop(unused_kwarg, None)

        processor_dict.update(kwargs)

        accepted_args_and_kwargs = cls.__init__.__code__.co_varnames[: cls.__init__.__code__.co_argcount][1:]

        unused_kwargs, valid_kwargs = cls.validate_init_kwargs(
            processor_config=processor_dict, valid_kwargs=accepted_args_and_kwargs
        )

        args_to_update = {
            i: valid_kwargs.pop(arg)
            for i, arg in enumerate(accepted_args_and_kwargs)
            if (arg in valid_kwargs and i < len(args))
        }
        args = [args_to_update.get(i, arg) for i, arg in enumerate(args)]

        processor = cls(*args, **valid_kwargs)

        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, unused_kwargs
        else:
            return processor

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: ProcessingKwargs,
        tokenizer_init_kwargs: dict | None = None,
        **kwargs,
    ) -> dict[str, dict]:
        """
        Method to merge dictionaries of kwargs cleanly separated by modality within a Processor instance.
        The order of operations is as follows:
            1) kwargs passed as before have highest priority to preserve BC.
                ```python
                high_priority_kwargs = {"crop_size" = {"height": 222, "width": 222}, "padding" = "max_length"}
                processor(..., **high_priority_kwargs)
                ```
            2) kwargs passed as modality-specific kwargs have second priority. This is the recommended API.
                ```python
                processor(..., text_kwargs={"padding": "max_length"}, images_kwargs={"crop_size": {"height": 222, "width": 222}}})
                ```
            3) kwargs passed during instantiation of a modality processor have fourth priority.
                ```python
                tokenizer = tokenizer_class(..., {"padding": "max_length"})
                image_processor = image_processor_class(...)
                processor(tokenizer, image_processor) # will pass max_length unless overridden by kwargs at call
                ```
            4) defaults kwargs specified at processor level have lowest priority.
                ```python
                class MyProcessingKwargs(ProcessingKwargs, CommonKwargs, TextKwargs, ImagesKwargs, total=False):
                    _defaults = {
                        "text_kwargs": {
                            "padding": "max_length",
                            "max_length": 64,
                        },
                    }
                ```
        Args:
            ModelProcessorKwargs (`ProcessingKwargs`):
                Typed dictionary of kwargs specifically required by the model passed.
            tokenizer_init_kwargs (`Dict`, *optional*):
                Dictionary of kwargs the tokenizer was instantiated with and need to take precedence over defaults.

        Returns:
            output_kwargs (`Dict`):
                Dictionary of per-modality kwargs to be passed to each modality-specific processor.

        """
        kwargs = copy.deepcopy(kwargs)

        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
        }

        map_preprocessor_kwargs = {
            "text_kwargs": "tokenizer",
            "images_kwargs": "image_processor",
            "audio_kwargs": "feature_extractor",
            "videos_kwargs": "video_processor",
        }

        possible_modality_keywords = {"text", "audio", "videos", "images"}
        used_keys = set()

        for modality in default_kwargs:
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
            modality_valid_kwargs = set(ModelProcessorKwargs.__annotations__[modality].__annotations__)
            if modality in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[modality], None)
                preprocessor_valid_kwargs = (
                    getattr(preprocessor, "valid_kwargs", None) if preprocessor is not None else None
                )
                modality_valid_kwargs.update(
                    set(preprocessor_valid_kwargs.__annotations__ if preprocessor_valid_kwargs is not None else [])
                )
            for modality_key in modality_valid_kwargs:
                if tokenizer_init_kwargs is not None and modality_key in tokenizer_init_kwargs:
                    value = (
                        getattr(self.tokenizer, modality_key)
                        if hasattr(self.tokenizer, modality_key)
                        else tokenizer_init_kwargs[modality_key]
                    )
                    default_kwargs[modality][modality_key] = value
        output_kwargs.update(default_kwargs)

        common_kwargs = ModelProcessorKwargs._defaults.get("common_kwargs", {})
        common_kwargs.update(kwargs.get("common_kwargs", {}))
        if common_kwargs:
            for kwarg in output_kwargs.values():
                kwarg.update(common_kwargs)

        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality, output_kwarg in output_kwargs.items():
            modality_valid_kwargs = set(ModelProcessorKwargs.__annotations__[modality].__annotations__)
            if modality in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[modality], None)
                preprocessor_valid_kwargs = (
                    getattr(preprocessor, "valid_kwargs", None) if preprocessor is not None else None
                )
                modality_valid_kwargs.update(
                    set(preprocessor_valid_kwargs.__annotations__ if preprocessor_valid_kwargs is not None else [])
                )
            for modality_key in modality_valid_kwargs:
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed two times:\n"
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                    if kwarg_value == "__empty__" and modality_key in non_modality_kwargs:
                        kwarg_value = kwargs[modality_key]
                elif modality_key in kwargs:
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if not isinstance(kwarg_value, str) or kwarg_value != "__empty__":
                    output_kwarg[modality_key] = kwarg_value
                    used_keys.add(modality_key)

        if any(key in default_kwargs for key in kwargs):
            for modality, subdict in kwargs.items():
                if modality in default_kwargs:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            for key, kwarg in kwargs.items():
                if key not in used_keys and key not in possible_modality_keywords:
                    logger.warning_once(
                        f"Keyword argument `{key}` is not a valid argument for this processor and will be ignored."
                    )

        for key, typed_dict_obj in ModelProcessorKwargs.__annotations__.items():
            if key in map_preprocessor_kwargs:
                preprocessor = getattr(self, map_preprocessor_kwargs[key], None)
                if preprocessor is None or getattr(preprocessor, "valid_kwargs", None) is None:
                    continue
                preprocessor_typed_dict_obj = getattr(preprocessor, "valid_kwargs")
                typed_dict_obj = _merge_typed_dict(preprocessor_typed_dict_obj, typed_dict_obj)
            validate_typed_dict(typed_dict_obj, output_kwargs[key])
        return output_kwargs

    @classmethod
    def from_pretrained(
        cls: type[SpecificProcessorType],
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs,
    ) -> SpecificProcessorType:
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], image processor
        [`~image_processing_utils.ImageProcessingMixin`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        if token is not None:
            kwargs["token"] = token

        processor_dict, instantiation_kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)
        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, processor_dict, **kwargs)
        return cls.from_args_and_dict(args, processor_dict, **instantiation_kwargs)

    @classmethod
    def get_attributes(cls):
        args_in_init = inspect.signature(cls.__init__).parameters.keys()
        attributes = []
        for sub_processor_type in args_in_init:
            if sub_processor_type == "audio_tokenizer":
                continue
            if any(modality in sub_processor_type for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING.keys()):
                attributes.append(sub_processor_type)

        if not attributes:
            for attribute_name, value in cls.__dict__.items():
                if value is None or attribute_name == "audio_tokenizer_class" or not attribute_name.endswith("_class"):
                    continue
                inferred_attribute = attribute_name[: -len("_class")]
                if inferred_attribute == "audio_tokenizer":
                    continue
                if any(modality in inferred_attribute for modality in MODALITY_TO_AUTOPROCESSOR_MAPPING.keys()):
                    attributes.append(inferred_attribute)

        return attributes

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.



        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls, sub_processor_type, pretrained_model_name_or_path, subfolder="", **kwargs
    ):
        auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING["tokenizer"]
        is_primary = sub_processor_type == "tokenizer"

        if is_primary:
            tokenizer = auto_processor_class.from_pretrained(
                pretrained_model_name_or_path, subfolder=subfolder, **kwargs
            )
        else:
            tokenizer_subfolder = os.path.join(subfolder, sub_processor_type) if subfolder else sub_processor_type
            try:
                tokenizer = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=tokenizer_subfolder, **kwargs
                )
            except (OSError, ValueError):
                fallback_folder = "the root directory" if not subfolder else f"subfolder `{subfolder}`"
                logger.warning(
                    f"Could not load tokenizer from subfolder `{tokenizer_subfolder}`. "
                    f"Falling back to loading from {fallback_folder}. "
                    f"This behavior is deprecated and will be removed in a future version."
                )
                tokenizer = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                )
        return tokenizer

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, processor_dict=None, **kwargs):
        """
        Identify and instantiate the subcomponents of Processor classes, such as image processors, tokenizers,
        and feature extractors. This method inspects the processor's `__init__` signature to identify parameters
        that correspond to known modality types (image_processor, tokenizer, feature_extractor, etc.) or contain
        modality names in their attribute name.

        For tokenizers: Uses the appropriate Auto class (AutoTokenizer) to load via `.from_pretrained()`.
        Additional tokenizers (e.g., "decoder_tokenizer") are loaded from subfolders.

        For other sub-processors (image_processor, feature_extractor, etc.): Primary ones are loaded via
        Auto class. Additional ones are instantiated from the config stored in processor_config.json
        (passed as processor_dict).

        Args:
            pretrained_model_name_or_path: Path or model id to load from.
            processor_dict: Optional dict containing processor config (from processor_config.json).
                Required when loading additional non-tokenizer sub-processors.
        """
        args = []
        processor_dict = processor_dict if processor_dict is not None else {}
        subfolder = kwargs.pop("subfolder", "")

        sub_processors = cls.get_attributes()
        for sub_processor_type in sub_processors:
            modality = _get_modality_for_attribute(sub_processor_type)
            is_primary = sub_processor_type == modality

            if (
                "tokenizer" in sub_processor_type
            ):  # This is only necessary for the checkpoint in test_processing_mistral3.py which has no config.json and
                if "PixtralProcessor" in cls.__name__:
                    from .tokenization_utils_tokenizers import TokenizersBackend

                    tokenizer = TokenizersBackend.from_pretrained(
                        pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                    )
                else:
                    tokenizer = cls._load_tokenizer_from_pretrained(
                        sub_processor_type, pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                    )
                args.append(tokenizer)
            elif is_primary:
                auto_processor_class = MODALITY_TO_AUTOPROCESSOR_MAPPING[sub_processor_type]
                if hasattr(cls, sub_processor_type + "_class"):
                    sub_processor_class_name = getattr(cls, sub_processor_type + "_class")
                    logger.warning_once(
                        f"`{cls.__name__}` defines `{sub_processor_type}_class = '{sub_processor_class_name}'`, "
                        f"which is deprecated. Register the correct mapping in `{auto_processor_class.__name__}` instead.",
                    )
                    auto_processor_class = cls.get_possibly_dynamic_module(sub_processor_class_name)
                sub_processor = auto_processor_class.from_pretrained(
                    pretrained_model_name_or_path, subfolder=subfolder, **kwargs
                )
                args.append(sub_processor)

            elif sub_processor_type in processor_dict:
                sub_processor_config = processor_dict[sub_processor_type]
                if isinstance(sub_processor_config, dict):
                    type_key = f"{modality}_type"
                    class_name = sub_processor_config.get(type_key)
                    if class_name is None:
                        raise ValueError(
                            f"Cannot instantiate {sub_processor_type}: missing '{type_key}' in config. "
                            f"Config keys: {list(sub_processor_config.keys())}"
                        )
                    processor_class = cls.get_possibly_dynamic_module(class_name)
                    sub_processor = processor_class(**sub_processor_config)
                    args.append(sub_processor)
                else:
                    raise ValueError(
                        f"Expected dict for {sub_processor_type} in processor_config.json, "
                        f"got {type(sub_processor_config)}"
                    )
            else:
                raise ValueError(
                    f"Cannot find config for {sub_processor_type} in processor_config.json. "
                    f"Available keys: {list(processor_dict.keys())}"
                )

        return args

    @staticmethod
    def get_possibly_dynamic_module(module_name):
        if hasattr(transformers_module, module_name):
            return getattr(transformers_module, module_name)
        lookup_locations = [
            transformers_module.IMAGE_PROCESSOR_MAPPING,
            transformers_module.VIDEO_PROCESSOR_MAPPING,
            transformers_module.TOKENIZER_MAPPING,
            transformers_module.FEATURE_EXTRACTOR_MAPPING,
            transformers_module.MODEL_FOR_AUDIO_TOKENIZATION_MAPPING,
        ]
        for lookup_location in lookup_locations:
            for custom_class in lookup_location._extra_content.values():
                if isinstance(custom_class, tuple):
                    for custom_subclass in custom_class:
                        if custom_subclass is not None and custom_subclass.__name__ == module_name:
                            return custom_subclass
                elif custom_class is not None and custom_class.__name__ == module_name:
                    return custom_class
        raise ValueError(
            f"Could not find module {module_name} in `transformers`. If this is a custom class, "
            f"it should be registered using the relevant `AutoClass.register()` function so that "
            f"other functions can find it!"
        )

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError(f"Cannot batch decode text: {self.__class__.__name__} has no tokenizer.")
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError(f"Cannot decode text: {self.__class__.__name__} has no tokenizer.")
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def unused_input_names(self) -> list[str]:
        pass

    @property
    def model_input_names(self) -> list[str]:
        pass

    @staticmethod
    def validate_init_kwargs(processor_config, valid_kwargs):
        kwargs_from_config = set(processor_config.keys())
        valid_kwargs_set = set(valid_kwargs)

        unused_keys = kwargs_from_config - valid_kwargs_set
        valid_keys = kwargs_from_config & valid_kwargs_set

        unused_kwargs = {k: processor_config[k] for k in unused_keys} if unused_keys else {}
        valid_kwargs = {k: processor_config[k] for k in valid_keys} if valid_keys else {}

        return unused_kwargs, valid_kwargs

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]] | list[list[dict[str, str]]],
        chat_template: str | None = None,
        tools: list[dict] | None = None,
        documents: list[dict[str, str]] | None = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool | str = False,
        return_assistant_tokens_mask: bool = False,
        tokenize: bool = False,
        return_tensors: str | TensorType | None = None,
        return_dict: bool = False,
        load_audio_from_video: bool = False,
        processor_kwargs: dict | None = None,
        **kwargs,
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """
        processor_kwargs = processor_kwargs or {}

        if chat_template is None:
            if isinstance(self.chat_template, dict) and "default" in self.chat_template:
                chat_template = self.chat_template["default"]
            elif isinstance(self.chat_template, dict):
                raise ValueError(
                    'The processor has multiple chat templates but none of them are named "default". You need to specify'
                    " which one to use by passing the `chat_template` argument. Available templates are: "
                    f"{', '.join(self.chat_template.keys())}"
                )
            elif self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "Cannot use apply_chat_template because this processor does not have a chat template."
                )
        else:
            if isinstance(self.chat_template, dict) and chat_template in self.chat_template:
                chat_template = self.chat_template[chat_template]
            else:
                pass

        template_kwargs = _get_template_variables(chat_template)
        processor_kwargs_from_kwargs = {k: v for k, v in kwargs.items() if k not in template_kwargs}
        if processor_kwargs_from_kwargs:
            logger.warning(
                "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`"
            )
            processor_kwargs = processor_kwargs_from_kwargs

        is_tokenizers_fast = False
        if hasattr(self, "tokenizer"):
            if hasattr(self.tokenizer, "backend"):
                is_tokenizers_fast = self.tokenizer.backend == "tokenizers"
            else:
                is_tokenizers_fast = self.tokenizer.__class__.__name__.endswith("Fast")

        if continue_final_message:
            if add_generation_prompt:
                raise ValueError(
                    "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."
                )
            if return_assistant_tokens_mask:
                raise ValueError("continue_final_message is not compatible with return_assistant_tokens_mask.")

        if return_assistant_tokens_mask:
            if not is_tokenizers_fast:
                raise ValueError(
                    "`return_assistant_tokens_mask` is not possible with slow tokenizers. Make sure you have `tokenizers` installed. "
                    "If the error persists, open an issue to support a Fast tokenizer for your model."
                )
            else:
                processor_kwargs["return_offsets_mapping"] = (
                    True  # force offset mapping so we can infer token boundaries
                )

        sampling_rate = kwargs.get("sampling_rate", processor_kwargs.get("sampling_rate"))
        if sampling_rate is None:
            if hasattr(self, "feature_extractor") and hasattr(self.feature_extractor, "sampling_rate"):
                sampling_rate = self.feature_extractor.sampling_rate
            else:
                sampling_rate = 16_000

        load_audio_backend = kwargs.get("load_audio_backend", processor_kwargs.get("load_audio_backend"))
        if load_audio_backend is None:
            default_audio_kwargs = self.valid_processor_kwargs._defaults.get("audio_kwargs", {})
            load_audio_backend = default_audio_kwargs.get("load_audio_backend", "auto")

        if isinstance(conversation, (list, tuple)) and (
            isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "content")
        ):
            is_batched = True
            conversations = conversation
        else:
            is_batched = False
            conversations = [conversation]

        for conversation_idx, conversation in enumerate(conversations):
            for message in conversation:
                if not isinstance(message.get("content"), list):
                    continue
                new_content = []
                for content in message["content"]:
                    if isinstance(content, dict) and content.get("type") == "image_url" and "image_url" in content:
                        image_url_info = content["image_url"]
                        url = image_url_info.get("url", "") if isinstance(image_url_info, dict) else image_url_info
                        new_content.append({"type": "image", "url": url})
                    else:
                        new_content.append(content)
                message["content"] = new_content

        if tokenize:
            batch_images, batch_videos = [], []
            batch_audios = []
            for conversation in conversations:
                images, videos = [], []
                for message in conversation:
                    content = message.get("content") or []
                    if isinstance(content, str):
                        continue
                    visuals = [
                        content_block for content_block in content if content_block["type"] in ["image", "video"]
                    ]
                    audio_fnames = [
                        content_block[key]
                        for content_block in content
                        for key in ["audio", "url", "path"]
                        if key in content_block and content_block["type"] == "audio"
                    ]
                    image_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["image", "url", "path", "base64"]
                        if key in vision_info and vision_info["type"] == "image"
                    ]
                    images.extend(image_fnames)
                    video_fnames = [
                        vision_info[key]
                        for vision_info in visuals
                        for key in ["video", "url", "path"]
                        if key in vision_info and vision_info["type"] == "video"
                    ]
                    videos.extend(video_fnames)

                    if not load_audio_from_video:
                        for fname in audio_fnames:
                            batch_audios.append(
                                load_audio(fname, sampling_rate=sampling_rate, backend=load_audio_backend)
                            )
                    else:
                        for fname in video_fnames:
                            message["content"].append({"type": "audio"})
                            batch_audios.append(
                                load_audio(fname, sampling_rate=sampling_rate, backend=load_audio_backend)
                            )

                batch_images.append(images)
                batch_videos.append(videos)

        template_kwargs = {**self.tokenizer.special_tokens_map, **kwargs}
        prompt, generation_indices = render_jinja_template(
            conversations=conversations,
            tools=tools,
            documents=documents,
            chat_template=chat_template,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            continue_final_message=continue_final_message,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

        if not is_batched:
            prompt = prompt[0]

        if tokenize:
            single_prompt = prompt[0] if is_batched else prompt
            if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
                processor_kwargs["add_special_tokens"] = False

            if "do_sample_frames" not in processor_kwargs and (
                processor_kwargs.get("fps") is not None or processor_kwargs.get("num_frames") is not None
            ):
                processor_kwargs["do_sample_frames"] = True

            if return_tensors:
                processor_kwargs["return_tensors"] = return_tensors

            images_exist = any((im is not None) for im_list in batch_images for im in im_list)
            videos_exist = any((vid is not None) for vid_list in batch_videos for vid in vid_list)
            out = self(
                text=prompt,
                images=batch_images if images_exist else None,
                videos=batch_videos if videos_exist else None,
                audio=batch_audios or None,
                **processor_kwargs,
            )

            if return_dict:
                if return_assistant_tokens_mask:
                    assistant_masks = []
                    offset_mapping = out.pop("offset_mapping")
                    input_ids = out["input_ids"]
                    for i in range(len(input_ids)):
                        current_mask = [0] * len(input_ids[i])
                        offsets = offset_mapping[i]
                        offset_starts = [start for start, end in offsets]
                        for assistant_start_char, assistant_end_char in generation_indices[i]:
                            start_pos = bisect.bisect_left(offset_starts, assistant_start_char)
                            end_pos = bisect.bisect_left(offset_starts, assistant_end_char)

                            if not (
                                start_pos >= 0
                                and start_pos < len(offsets)
                                and offsets[start_pos][0] <= assistant_start_char < offsets[start_pos][1]
                            ):
                                continue
                            if end_pos > len(input_ids[i]):
                                end_pos = len(input_ids[i])
                            for token_id in range(start_pos, end_pos or len(input_ids[i])):
                                current_mask[token_id] = 1
                        assistant_masks.append(current_mask)
                    out["assistant_masks"] = assistant_masks
                    out.convert_to_tensors(tensor_type=return_tensors)
                return out
            else:
                return out["input_ids"]
        return prompt

    def parse_response(
        self,
        response: "str | list[int] | list[str] | list[list[int]] | np.ndarray | torch.Tensor",
        schema: list | dict | None = None,
        *,
        prefix: "str | list[int] | list[str] | list[list[int]] | np.ndarray | torch.Tensor | None" = None,
    ):
        """
        Converts an output string created by generating text from a model into a parsed message dictionary.
        This method is intended for use with chat models, and will read the tokenizer's `response_template`
        attribute (preferred) or the legacy `response_schema` attribute to control parsing. Either can be
        overridden by passing a `schema` argument directly.

        Accepts either a single sequence (returning a single message `dict`) or a batch (returning a `list` of
        message dicts, one per item).

        Args:
            response (`str`, token ids, 1D/2D tensor, or a list of these):
                The output generated by the model, either decoded text or token ids, as a single sequence or a
                batch.
            schema (`Union[list, dict]`, *optional*):
                A response template (preferred, new-style) or legacy response schema dict. If not provided, the
                tokenizer's `response_template` or `response_schema` attribute is used (in that order).
            prefix (`str`, token ids, 1D/2D tensor, or a list of these):
                The prompt that came before generation. Many chat templates pre-write part of the message, so
                this is needed to parse correctly. For a batched `response`, pass either a single prefix
                (broadcast to every item) or one prefix per item. Only supported with new-style templates.
        """
        if not hasattr(self, "tokenizer"):
            raise ValueError("Can't use parse_response on a processor class without a tokenizer!")
        return self.tokenizer.parse_response(response, schema, prefix=prefix)

    def post_process_multimodal_output(
        self, generated_outputs, skip_special_tokens=True, generation_mode=None, **kwargs
    ):
        """
        Post-process the output of a multimodal model to return the requested modality output.
        If the model cannot generated the requested modality, an error will be raised.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            generation_mode (`str`, *optional*):
                Generation mode indicated which modality to output and can be one of `["text", "image", "audio"]`.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        if generation_mode is not None and generation_mode != "text":
            raise ValueError(
                f"{self.__class__.__name__} got an unexpected generation_mode={generation_mode}. Supported options are only [`text`]"
            )
        return self.post_process_image_text_to_text(
            generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs
        )

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=True, **kwargs):
        """
        Post-process the output of a vlm to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `decode` method.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)

    def _check_special_mm_tokens(self, text: list[str], text_inputs: "BatchFeature", modalities: list[str]):
        pass


ProcessorMixin.push_to_hub = copy_func(ProcessorMixin.push_to_hub)
if ProcessorMixin.push_to_hub.__doc__ is not None:
    ProcessorMixin.push_to_hub.__doc__ = ProcessorMixin.push_to_hub.__doc__.format(
        object="processor", object_class="AutoProcessor", object_files="processor files"
    )


def prepare_prompt_input(
    inputs: str | list[str] | None,
    batch_size: int,
    input_name: str = "inputs",
) -> list[str | None]:
    pass
