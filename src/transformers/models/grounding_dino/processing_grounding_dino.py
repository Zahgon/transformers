
import warnings
from typing import TYPE_CHECKING

from ...image_transforms import center_to_corners_format
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import BatchEncoding, PreTokenizedInput, TextInput
from ...utils import TensorType, auto_docstring, is_torch_available


if is_torch_available():
    import torch

if TYPE_CHECKING:
    from .modeling_grounding_dino import GroundingDinoObjectDetectionOutput


AnnotationType = dict[str, int | str | list[dict]]


def get_phrases_from_posmap(posmaps, input_ids):
    pass


def _is_list_of_candidate_labels(text) -> bool:
    pass


def _merge_candidate_labels_text(text: list[str]) -> str:
    pass


class DictWithDeprecationWarning(dict):
    message = (
        "The key `labels` is will return integer ids in `GroundingDinoProcessor.post_process_grounded_object_detection` "
        "output since v4.51.0. Use `text_labels` instead to retrieve string object names."
    )

    def __getitem__(self, key):
        if key == "labels":
            warnings.warn(self.message, FutureWarning)
        return super().__getitem__(key)

    def get(self, key, *args, **kwargs):
        if key == "labels":
            warnings.warn(self.message, FutureWarning)
        return super().get(key, *args, **kwargs)


class GroundingDinoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": False,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": True,
            "return_length": False,
            "verbose": True,
        }
    }


@auto_docstring
class GroundingDinoProcessor(ProcessorMixin):
    valid_processor_kwargs = GroundingDinoProcessorKwargs

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[GroundingDinoProcessorKwargs],
    ) -> BatchEncoding:
        if text is not None:
            text = self._preprocess_input_text(text)
        return super().__call__(images=images, text=text, **kwargs)

    def _preprocess_input_text(self, text):
        pass

    def post_process_grounded_object_detection(
        self,
        outputs: "GroundingDinoObjectDetectionOutput",
        input_ids: TensorType | None = None,
        threshold: float = 0.25,
        text_threshold: float = 0.25,
        target_sizes: TensorType | list[tuple] | None = None,
        text_labels: list[list[str]] | None = None,
    ):
        pass


__all__ = ["GroundingDinoProcessor"]
