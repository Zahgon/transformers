
import numpy as np

from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class Glm4vProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_token_type_ids": False,
            "return_mm_token_type_ids": True,
        },
        "videos_kwargs": {"return_metadata": True},
    }


@auto_docstring
class Glm4vProcessor(ProcessorMixin):
    valid_processor_kwargs = Glm4vProcessorKwargs

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = "<|image|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.video_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
        self.video_end_id = tokenizer.convert_tokens_to_ids("<|end_of_video|>")

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    def replace_video_token(self, video_inputs: dict, video_idx: int) -> str:
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        pass

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        pass

    def create_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        pass

    def replace_frame_token_id(self, timestamp_sec, num_image_tokens: int = 1):
        pass


__all__ = ["Glm4vProcessor"]
