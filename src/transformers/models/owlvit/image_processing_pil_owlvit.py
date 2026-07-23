
from typing import TYPE_CHECKING

from ...image_processing_backends import PilBackend
from ...image_transforms import center_to_corners_format
from ...image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD, PILImageResampling
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import TensorType, auto_docstring, logging
from ...utils.import_utils import requires


def _upcast(t):
    import torch

    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()


def _scale_boxes(boxes, target_sizes):
    """
    Scale batch of bounding boxes to the target sizes.

    Args:
        boxes (`torch.Tensor` of shape `(batch_size, num_boxes, 4)`):
            Bounding boxes to scale. Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (`list[tuple[int, int]]` or `torch.Tensor` of shape `(batch_size, 2)`):
            Target sizes to scale the boxes to. Each target size is expected to be in (height, width) format.

    Returns:
        `torch.Tensor` of shape `(batch_size, num_boxes, 4)`: Scaled bounding boxes.
    """
    import torch

    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        raise TypeError("`target_sizes` must be a list, tuple or torch.Tensor")

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its (x1, y1, x2, y2) coordinates.

    Args:
        boxes (`torch.FloatTensor` of shape `(number_of_boxes, 4)`):
            Boxes for which the area will be computed. They are expected to be in (x1, y1, x2, y2) format with `0 <= x1
            < x2` and `0 <= y1 < y2`.
    Returns:
        `torch.FloatTensor`: a tensor containing the area for each box.
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1, boxes2):
    import torch

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


if TYPE_CHECKING:
    from .modeling_owlvit import OwlViTObjectDetectionOutput


logger = logging.get_logger(__name__)


@auto_docstring
class OwlViTImageProcessorPil(PilBackend):
    resample = PILImageResampling.BICUBIC
    image_mean = OPENAI_CLIP_MEAN
    image_std = OPENAI_CLIP_STD
    size = {"height": 768, "width": 768}
    default_to_square = True
    crop_size = {"height": 768, "width": 768}
    do_resize = True
    do_center_crop = False
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True
    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        if "rescale" in kwargs:
            rescale_val = kwargs.pop("rescale")
            kwargs["do_rescale"] = rescale_val

        super().__init__(**kwargs)

    @requires(backends=("torch",))
    def post_process_object_detection(
        self,
        outputs: "OwlViTObjectDetectionOutput",
        threshold: float = 0.1,
        target_sizes: TensorType | list[tuple] | None = None,
    ):
        """
        Converts the raw output of [`OwlViTForObjectDetection`] into final bounding boxes in (top_left_x, top_left_y,
        bottom_right_x, bottom_right_y) format.

        Args:
            outputs ([`OwlViTObjectDetectionOutput`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.1):
                Score threshold to keep object detection predictions.
            target_sizes (`torch.Tensor` or `list[tuple[int, int]]`, *optional*):
                Tensor of shape `(batch_size, 2)` or list of tuples (`tuple[int, int]`) containing the target size
                `(height, width)` of each image in the batch. If unset, predictions will not be resized.

        Returns:
            `list[Dict]`: A list of dictionaries, each dictionary containing the following keys:
            - "scores": The confidence scores for each predicted box on the image.
            - "labels": Indexes of the classes predicted by the model on the image.
            - "boxes": Image bounding boxes in (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.
        """
        import torch

        batch_logits, batch_boxes = outputs.logits, outputs.pred_boxes
        batch_size = len(batch_logits)

        if target_sizes is not None and len(target_sizes) != batch_size:
            raise ValueError("Make sure that you pass in as many target sizes as images")

        batch_class_logits = torch.max(batch_logits, dim=-1)
        batch_scores = torch.sigmoid(batch_class_logits.values)
        batch_labels = batch_class_logits.indices

        batch_boxes = center_to_corners_format(batch_boxes)

        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for scores, labels, boxes in zip(batch_scores, batch_labels, batch_boxes):
            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            results.append({"scores": scores, "labels": labels, "boxes": boxes})

        return results

    @requires(backends=("torch",))
    def post_process_image_guided_detection(self, outputs, threshold=0.0, nms_threshold=0.3, target_sizes=None):
        pass


__all__ = ["OwlViTImageProcessorPil"]
