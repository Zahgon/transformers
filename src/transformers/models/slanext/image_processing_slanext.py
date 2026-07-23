

import torch
import torchvision.transforms.v2.functional as tvF

from ...image_processing_backends import TorchvisionBackend
from ...image_processing_utils import BatchFeature
from ...image_transforms import group_images_by_shape, reorder_images
from ...image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, SizeDict
from ...processing_utils import ImagesKwargs, Unpack
from ...utils import auto_docstring, is_torchdynamo_compiling, logging
from ...utils.generic import TensorType
from ...utils.import_utils import requires


logger = logging.get_logger(__name__)


@auto_docstring
@requires(backends=("torch",))
class SLANeXtImageProcessor(TorchvisionBackend):
    resample = 2  # PILImageResampling.BILINEAR
    image_mean = IMAGENET_DEFAULT_MEAN
    image_std = IMAGENET_DEFAULT_STD
    size = {"height": 512, "width": 512}
    pad_size = {"height": 512, "width": 512}
    do_convert_rgb = True
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_pad = True

    def _resize(
        self,
        image: "torch.Tensor",
        size: SizeDict,
    ) -> "torch.Tensor":
        batch_size, channels, height, width = image.shape
        image = image.view(batch_size * channels, height, width)

        device = image.device

        scale = max(size.height, size.width) / max(height, width)
        target_height = round(height * scale)
        target_width = round(width * scale)

        target_col = torch.arange(target_width, dtype=torch.float32, device=device)
        src_col = (target_col + 0.5) * (float(width) / float(target_width)) - 0.5
        src_col_floor = src_col.floor().to(torch.int32)
        src_col_frac = src_col - src_col_floor.float()
        src_col_frac = torch.where(src_col_floor < 0, torch.zeros_like(src_col_frac), src_col_frac)
        src_col_floor = torch.where(src_col_floor < 0, torch.zeros_like(src_col_floor), src_col_floor)
        src_col_frac = torch.where(src_col_floor >= width - 1, torch.ones_like(src_col_frac), src_col_frac)
        src_col_floor = torch.where(
            src_col_floor >= width - 1, torch.full_like(src_col_floor, width - 2), src_col_floor
        )
        weight_right = (src_col_frac * 2048 + 0.5).floor().to(torch.int32)  # round-to-nearest
        weight_left = 2048 - weight_right  # (target_w,)
        target_row = torch.arange(target_height, dtype=torch.float32, device=device)
        src_row = (target_row + 0.5) * (float(height) / float(target_height)) - 0.5
        src_row_floor = src_row.floor().to(torch.int32)
        src_row_frac = src_row - src_row_floor.float()
        src_row_frac = torch.where(src_row_floor < 0, torch.zeros_like(src_row_frac), src_row_frac)
        src_row_floor = torch.where(src_row_floor < 0, torch.zeros_like(src_row_floor), src_row_floor)
        src_row_frac = torch.where(src_row_floor >= height - 1, torch.ones_like(src_row_frac), src_row_frac)
        src_row_floor = torch.where(
            src_row_floor >= height - 1, torch.full_like(src_row_floor, height - 2), src_row_floor
        )
        weight_bottom = (src_row_frac * 2048 + 0.5).floor().to(torch.int32)
        weight_top = 2048 - weight_bottom  # (target_h,)

        image_uint8 = image.clamp(0, 255).to(torch.uint8)  # (C, H, W)
        image_int32 = image_uint8.to(torch.int32)  # (C, H, W)
        col_left = src_col_floor.long()  # (target_w,)
        col_right = (src_col_floor + 1).long()  # (target_w,)  safe: src_col_floor <= width-2
        row_top = src_row_floor.long()  # (target_h,)
        row_bottom = (src_row_floor + 1).long()  # (target_h,)
        pixel_top_left = image_int32[:, row_top[:, None], col_left[None, :]]
        pixel_top_right = image_int32[:, row_top[:, None], col_right[None, :]]
        pixel_bottom_left = image_int32[:, row_bottom[:, None], col_left[None, :]]
        pixel_bottom_right = image_int32[:, row_bottom[:, None], col_right[None, :]]
        weight_bottom_3d = weight_bottom.view(1, target_height, 1)
        weight_top_3d = weight_top.view(1, target_height, 1)
        weight_right_3d = weight_right.view(1, 1, target_width)
        weight_left_3d = weight_left.view(1, 1, target_width)
        interp = weight_top_3d * (
            weight_left_3d * pixel_top_left + weight_right_3d * pixel_top_right
        ) + weight_bottom_3d * (weight_left_3d * pixel_bottom_left + weight_right_3d * pixel_bottom_right)
        interp = (interp + (1 << 21)) >> 22
        result = interp.clamp(0, 255).to(torch.uint8)  # (B*C, target_h, target_w)

        return result.view(batch_size, channels, target_height, target_width).to(dtype=image.dtype)

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        size: SizeDict,
        resample: "tvF.InterpolationMode | int | None",
        do_center_crop: bool,
        crop_size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        do_pad: bool | None,
        pad_size: SizeDict | None,
        disable_grouping: bool | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        if resample is not None and not is_torchdynamo_compiling():
            logger.warning_once("Resampling is not supported in SLANeXt")

        grouped_images, grouped_images_index = group_images_by_shape(images, disable_grouping=disable_grouping)
        resized_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_resize:
                stacked_images = self._resize(image=stacked_images, size=size)
            resized_images_grouped[shape] = stacked_images
        resized_images = reorder_images(resized_images_grouped, grouped_images_index)

        grouped_images, grouped_images_index = group_images_by_shape(resized_images, disable_grouping=disable_grouping)
        processed_images_grouped = {}
        for shape, stacked_images in grouped_images.items():
            if do_center_crop:
                stacked_images = self.center_crop(stacked_images, crop_size)
            stacked_images = self.rescale_and_normalize(
                stacked_images, do_rescale, rescale_factor, do_normalize, image_mean, image_std
            )
            processed_images_grouped[shape] = stacked_images
        processed_images = reorder_images(processed_images_grouped, grouped_images_index)

        if do_pad:
            processed_images = self.pad(processed_images, pad_size=pad_size, disable_grouping=disable_grouping)

        return BatchFeature(data={"pixel_values": processed_images}, tensor_type=return_tensors)

    def __init__(self, **kwargs: Unpack[ImagesKwargs]):
        super().__init__(**kwargs)
        self.init_decoder()

    def init_decoder(self):
        """
        Initialize the decoder vocabulary for table structure recognition.

        Builds a character dictionary mapping HTML table structure tokens (e.g., `<thead>`, `<tr>`, `<td>`, colspan/
        rowspan attributes) to integer indices. The dictionary includes special `"sos"` (start-of-sequence) and
        `"eos"` (end-of-sequence) tokens. Merged `<td></td>` tokens are used in place of standalone `<td>` tokens
        when applicable.
        """
        dict_character = [
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            "<tr>",
            "</tr>",
            "<td>",
            "<td",
            ">",
            "</td>",
        ]
        dict_character += [f' colspan="{i + 2}"' for i in range(19)]
        dict_character += [f' rowspan="{i + 2}"' for i in range(19)]

        if "<td></td>" not in dict_character:
            dict_character.append("<td></td>")
        if "<td>" in dict_character:
            dict_character.remove("<td>")

        dict_character = ["sos"] + dict_character + ["eos"]
        self.dict = {char: i for i, char in enumerate(dict_character)}
        self.character = dict_character
        self.td_token = ["<td>", "<td", "<td></td>"]
        self.bos_id = self.dict["sos"]
        self.eos_id = self.dict["eos"]

    def post_process_table_recognition(self, outputs):
        pass


__all__ = ["SLANeXtImageProcessor"]
