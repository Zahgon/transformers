from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class MiniMaxM3VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "videos_kwargs": {"do_resize": False, "return_metadata": True},
    }


@auto_docstring
class MiniMaxM3VLProcessor(ProcessorMixin):

    valid_processor_kwargs = MiniMaxM3VLProcessorKwargs

    IMAGE_TOKEN = "]<]image[>["
    VIDEO_TOKEN = "]<]video[>["
    VISION_START_TOKEN = "]<]start of image[>["
    VISION_END_TOKEN = "]<]end of image[>["

    def __init__(self, image_processor=None, tokenizer=None, video_processor=None, chat_template=None, **kwargs):
        self.image_token = self.IMAGE_TOKEN
        self.video_token = self.VIDEO_TOKEN
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN) if tokenizer else None
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.VIDEO_TOKEN) if tokenizer else None
        super().__init__(image_processor, tokenizer, video_processor, chat_template=chat_template)
        self.vision_start_token_id = tokenizer.convert_tokens_to_ids(self.VISION_START_TOKEN) if tokenizer else None
        self.vision_end_token_id = tokenizer.convert_tokens_to_ids(self.VISION_END_TOKEN) if tokenizer else None

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


__all__ = ["MiniMaxM3VLProcessor"]
