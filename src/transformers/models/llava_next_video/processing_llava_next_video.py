
import numpy as np

from ...image_processing_utils import select_best_resolution
from ...image_utils import get_image_size, to_numpy_array
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


class LlavaNextVideoProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "common_kwargs": {
            "return_tensors": "pt",
        },
    }


@auto_docstring
class LlavaNextVideoProcessor(ProcessorMixin):
    valid_processor_kwargs = LlavaNextVideoProcessorKwargs

    def __init__(
        self,
        video_processor=None,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        patch_size=None,
        vision_feature_select_strategy=None,
        video_token="<video>",
        image_token="<image>",
        num_additional_image_tokens=0,
        **kwargs,
    ):
        r"""
        patch_size (`int`, *optional*):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Should be same as in model's config
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
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
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
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
        super().__init__(video_processor, image_processor, tokenizer, chat_template=chat_template)

    def replace_image_token(self, image_inputs: dict | None = None, image_idx: int = 0) -> str:
        pass

    def replace_video_token(self, video_inputs: dict | None = None, video_idx: int = 0) -> str:
        pass

    def _get_number_of_features(self, orig_height: int, orig_width: int, height: int, width: int) -> int:
        pass

    def _get_unpadded_features(self, height, width, patches_height, patches_width, scale_height, scale_width):
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass


__all__ = ["LlavaNextVideoProcessor"]
