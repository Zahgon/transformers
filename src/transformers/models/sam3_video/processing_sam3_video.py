from typing import Union

from ...utils import is_torch_available


if is_torch_available():
    import torch

from ...utils import is_torchvision_available


if is_torchvision_available():
    from torchvision.ops import masks_to_boxes

from ...image_utils import ImageInput
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import TensorType, auto_docstring
from ...utils.import_utils import requires
from ...video_utils import VideoInput
from .modeling_sam3_video import Sam3VideoInferenceSession


@requires(backends=("torch", "torchvision"))
@auto_docstring
class Sam3VideoProcessor(ProcessorMixin):
    def __init__(
        self,
        image_processor,
        video_processor,
        tokenizer,
        target_size: int | None = None,
        **kwargs,
    ):
        r"""
        target_size (`int`, *optional*):
            The target size (target_size, target_size) to which the image will be resized.
        """
        super().__init__(image_processor, video_processor, tokenizer, **kwargs)
        self.target_size = target_size if target_size is not None else self.image_processor.size["height"]

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        segmentation_maps: ImageInput | None = None,
        original_sizes: list[list[float]] | torch.Tensor | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ) -> BatchEncoding:
        r"""
        images (`ImageInput`, *optional*):
            The image(s) to process.
        segmentation_maps (`ImageInput`, *optional*):
            The segmentation maps to process (optional, for image processor).
        original_sizes (`list[list[float]]`, `torch.Tensor`, *optional*):
            The original sizes of the images. Only used when images is not provided.

        Returns:
            A [`BatchEncoding`] with the following fields:
            - `pixel_values` (`torch.Tensor`): The processed image(s).
            - `original_sizes` (`list[list[float]]`): The original sizes of the images.
            - `labels` (`torch.Tensor`, *optional*): The processed segmentation maps (if provided).
        """
        if images is not None:
            encoding_image_processor = self.image_processor(
                images,
                segmentation_maps=segmentation_maps,
                return_tensors=return_tensors,
                **kwargs,
            )
        elif original_sizes is not None:
            if isinstance(original_sizes, torch.Tensor):
                original_sizes = original_sizes.cpu().tolist()
            encoding_image_processor = BatchEncoding({"original_sizes": original_sizes}, tensor_type=return_tensors)
        else:
            raise ValueError("Either images or original_sizes must be provided")

        original_sizes = encoding_image_processor["original_sizes"]
        if images is not None and len(original_sizes) != 1 and len(original_sizes) != len(images):
            raise ValueError(
                "original_sizes must be of length 1 or len(images). If you are passing a single image, you must pass a single original_size."
            )

        return encoding_image_processor

    def add_text_prompt(self, inference_session: Sam3VideoInferenceSession, text: str | list[str]):
        pass

    def init_video_session(
        self,
        video: VideoInput | None = None,
        inference_device: Union[str, "torch.device"] = "cpu",
        inference_state_device: Union[str, "torch.device"] | None = None,
        processing_device: Union[str, "torch.device"] | None = None,
        video_storage_device: Union[str, "torch.device"] | None = None,
        max_vision_features_cache_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        pass

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks

    def _apply_object_wise_non_overlapping_constraints(
        self,
        pred_masks,
        obj_scores,
        background_value=-10.0,
        prompt_ids=None,
    ):
        pass

    def _apply_object_wise_non_overlapping_constraints_impl(self, pred_masks, obj_scores, background_value=-10.0):
        pass

    def postprocess_outputs(
        self,
        inference_session,
        model_outputs,
        original_sizes: list[list[float]] | torch.Tensor | None = None,
    ):
        pass


__all__ = ["Sam3VideoProcessor"]
