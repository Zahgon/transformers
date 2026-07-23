
from ...image_utils import get_image_size, to_numpy_array
from ...processing_utils import (
    MultiModalData,
    ProcessingKwargs,
    ProcessorMixin,
)
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class LlavaProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_mm_token_type_ids": False, "return_text_replacement_offsets": False},
    }


@auto_docstring
class LlavaProcessor(ProcessorMixin):
    valid_processor_kwargs = LlavaProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        chat_template=None,
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        num_additional_image_tokens=0,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        num_additional_image_tokens (`int`, *optional*, defaults to 0):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
        """
        self.patch_size = patch_size
        self.num_additional_image_tokens = num_additional_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = tokenizer.encode(self.image_token, add_special_tokens=False)[0]
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass


__all__ = ["LlavaProcessor"]
