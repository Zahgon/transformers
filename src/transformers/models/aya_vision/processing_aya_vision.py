

from ...processing_utils import BatchFeature, MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring


class AyaVisionProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding_side": "left",
            "padding": True,
            "return_mm_token_type_ids": False,
        },
        "images_kwargs": {
            "crop_to_patches": True,
        },
    }


@auto_docstring
class AyaVisionProcessor(ProcessorMixin):
    valid_processor_kwargs = AyaVisionProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size: int = 28,
        img_size: int = 364,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        downsample_factor: int = 1,
        start_of_img_token="<|START_OF_IMG|>",
        end_of_img_token="<|END_OF_IMG|>",
        img_patch_token="<|IMG_PATCH|>",
        img_line_break_token="<|IMG_LINE_BREAK|>",
        tile_token="TILE",
        tile_global_token="TILE_GLOBAL",
        chat_template=None,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*, defaults to 28):
            The size of image patches for tokenization.
        img_size (`int`, *optional*, defaults to 364):
            The size of the image to be tokenized. This should correspond to the size given to the image processor.
        image_token (`str`, *optional*, defaults to `"<image>"`):
            The token to be used to represent an image in the text.
        downsample_factor (`int`, *optional*, defaults to 1):
            The factor by which to scale the patch size.
        start_of_img_token (`str`, *optional*, defaults to `"<|START_OF_IMG|>"`):
            The token to be used to represent the start of an image in the text.
        end_of_img_token (`str`, *optional*, defaults to `"<|END_OF_IMG|>"`):
            The token to be used to represent the end of an image in the text.
        img_patch_token (`str`, *optional*, defaults to `"<|IMG_PATCH|>"`):
            The token to be used to represent an image patch in the text.
        img_line_break_token (`str`, *optional*, defaults to `"<|IMG_LINE_BREAK|>"`):
            The token to be used to represent a line break in the text.
        tile_token (`str`, *optional*, defaults to `"TILE"`):
            The token to be used to represent an image patch in the text.
        tile_global_token (`str`, *optional*, defaults to `"TILE_GLOBAL"`):
            The token to be used to represent the cover image in the text.
        """
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.image_token = image_token
        self.patch_size = patch_size * downsample_factor
        self.img_size = img_size

        self.start_of_img_token = start_of_img_token
        self.end_of_img_token = end_of_img_token
        self.img_patch_token = img_patch_token
        self.img_line_break_token = img_line_break_token
        self.tile_token = tile_token
        self.tile_global_token = tile_global_token
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.img_patch_token)

    @property
    def image_token_ids(self) -> list[int]:
        pass

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    def _check_special_mm_tokens(self, text: list[str], text_inputs: "BatchFeature", modalities: list[str]):
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    @property
    def unused_input_names(self) -> list[str]:
        pass


__all__ = ["AyaVisionProcessor"]
