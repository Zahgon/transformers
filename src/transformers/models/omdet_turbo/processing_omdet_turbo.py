
from typing import TYPE_CHECKING

from ...feature_extraction_utils import BatchFeature
from ...image_transforms import center_to_corners_format
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, TextKwargs, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import (
    TensorType,
    auto_docstring,
    is_torch_available,
    is_torchvision_available,
)
from ...utils.import_utils import requires


if TYPE_CHECKING:
    from .modeling_omdet_turbo import OmDetTurboObjectDetectionOutput


class OmDetTurboTextKwargs(TextKwargs, total=False):

    task: str | list[str] | TextInput | PreTokenizedInput | None


if is_torch_available():
    import torch


if is_torchvision_available():
    from torchvision.ops.boxes import batched_nms


class OmDetTurboProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: OmDetTurboTextKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 77,
            "stride": 0,
            "return_overflowing_tokens": False,
            "return_special_tokens_mask": False,
            "return_offsets_mapping": False,
            "return_token_type_ids": False,
            "return_length": False,
            "verbose": True,
            "task": None,
        },
    }


def clip_boxes(box, box_size: tuple[int, int]):
    pass


def compute_score(boxes):
    pass


def _post_process_boxes_for_image(
    boxes: "torch.Tensor",
    scores: "torch.Tensor",
    labels: "torch.Tensor",
    image_num_classes: int,
    image_size: tuple[int, int],
    threshold: float,
    nms_threshold: float,
    max_num_det: int | None = None,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    pass


@requires(backends=("vision", "torchvision"))
@auto_docstring
class OmDetTurboProcessor(ProcessorMixin):
    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: list[str] | list[list[str]] | None = None,
        **kwargs: Unpack[OmDetTurboProcessorKwargs],
    ) -> BatchFeature:
        if images is None or text is None:
            raise ValueError("You have to specify both `images` and `text`")

        output_kwargs = self._merge_kwargs(
            OmDetTurboProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if isinstance(text, str):
            text = text.strip(" ").split(",")

        if not (len(text) and isinstance(text[0], (list, tuple))):
            text = [text]

        task = output_kwargs["text_kwargs"].pop("task", None)
        if task is None:
            task = [f"Detect {', '.join(text_single)}." for text_single in text]
        elif not isinstance(task, (list, tuple)):
            task = [task]

        encoding_image_processor = self.image_processor(images, **output_kwargs["images_kwargs"])
        tasks_encoding = self.tokenizer(text=task, **output_kwargs["text_kwargs"])

        classes = text

        classes_structure = torch.tensor([len(class_single) for class_single in classes], dtype=torch.long)
        classes_flattened = [class_single for class_batch in classes for class_single in class_batch]
        classes_encoding = self.tokenizer(text=classes_flattened, **output_kwargs["text_kwargs"])

        encoding = BatchFeature()
        encoding.update({f"tasks_{key}": value for key, value in tasks_encoding.items()})
        encoding.update({f"classes_{key}": value for key, value in classes_encoding.items()})
        encoding.update({"classes_structure": classes_structure})
        encoding.update(encoding_image_processor)

        return encoding

    @property
    def model_input_names(self):
        pass

    def _get_default_image_size(self) -> tuple[int, int]:
        pass

    def post_process_grounded_object_detection(
        self,
        outputs: "OmDetTurboObjectDetectionOutput",
        text_labels: list[str] | list[list[str]] | None = None,
        threshold: float = 0.3,
        nms_threshold: float = 0.5,
        target_sizes: TensorType | list[tuple] | None = None,
        max_num_det: int | None = None,
    ):
        pass


__all__ = ["OmDetTurboProcessor"]
