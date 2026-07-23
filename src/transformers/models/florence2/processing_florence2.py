import re
from typing import Any

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available, logging


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class Florence2ProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False, "return_mm_token_type_ids": False, "return_text_replacement_offsets": False},
    }


@auto_docstring
class Florence2Processor(ProcessorMixin):
    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        num_additional_image_tokens: int = 0,
        post_processor_config: dict | None = None,
        **kwargs,
    ):
        r"""
        num_additional_image_tokens (`int`, *optional*, defaults to 0):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
        post_processor_config (`dict`,  *optional*, defaults to 0):
            Task-specific parsing rules for [`Florence2PostProcessor`], e.g. regex patterns,
            thresholds, or banned tokens.
        """
        self.tasks_answer_post_processing_type = {
            "<OCR>": "pure_text",
            "<OCR_WITH_REGION>": "ocr",
            "<CAPTION>": "pure_text",
            "<DETAILED_CAPTION>": "pure_text",
            "<MORE_DETAILED_CAPTION>": "pure_text",
            "<OD>": "description_with_bboxes",
            "<DENSE_REGION_CAPTION>": "description_with_bboxes",
            "<CAPTION_TO_PHRASE_GROUNDING>": "phrase_grounding",
            "<REFERRING_EXPRESSION_SEGMENTATION>": "polygons",
            "<REGION_TO_SEGMENTATION>": "polygons",
            "<OPEN_VOCABULARY_DETECTION>": "description_with_bboxes_or_polygons",
            "<REGION_TO_CATEGORY>": "pure_text",
            "<REGION_TO_DESCRIPTION>": "pure_text",
            "<REGION_TO_OCR>": "pure_text",
            "<REGION_PROPOSAL>": "bboxes",
        }

        self.task_prompts_without_inputs = {
            "<OCR>": "What is the text in the image?",
            "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
            "<CAPTION>": "What does the image describe?",
            "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
            "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
            "<OD>": "Locate the objects with category name in the image.",
            "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
            "<REGION_PROPOSAL>": "Locate the region proposals in the image.",
        }

        self.task_prompts_with_input = {
            "<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
            "<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {input} in the image with mask",
            "<REGION_TO_SEGMENTATION>": "What is the polygon mask of region {input}",
            "<OPEN_VOCABULARY_DETECTION>": "Locate {input} in the image.",
            "<REGION_TO_CATEGORY>": "What is the region {input}?",
            "<REGION_TO_DESCRIPTION>": "What does the region {input} describe?",
            "<REGION_TO_OCR>": "What text is in the region {input}?",
        }

        self.num_image_tokens = image_processor.image_seq_length
        self.num_additional_image_tokens = num_additional_image_tokens
        self.post_processor_config = post_processor_config
        self.post_processor = Florence2PostProcessor(config=post_processor_config, tokenizer=tokenizer)
        self.image_token = tokenizer.image_token
        self.image_token_id = tokenizer.image_token_id

        super().__init__(image_processor, tokenizer, **kwargs)

    def _construct_prompts(self, text: str | list[str]) -> list[str]:
        pass

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | None = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        **kwargs: Unpack[Florence2ProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is None and text is None:
            raise ValueError("You have to specify at least one of `images` or `text`.")

        output_kwargs = self._merge_kwargs(
            Florence2ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        if images is not None:
            image_inputs = self.image_processor(images, **output_kwargs["images_kwargs"])

        if text is None:
            logger.warning_once("You are using Florence-2 without a text prefix.")
            text = [""] * (1 if not isinstance(images, list) else len(images))
        elif isinstance(text, str):
            text = [text]

        if not isinstance(text, list) or not all(isinstance(token, str) for token in text):
            raise ValueError("`text` must be a string or list of strings.")

        if isinstance(images, list) and len(images) != len(text):
            raise ValueError(f"Number of images ({len(images)}) must match number of texts ({len(text)}).")

        prompt_strings = self._construct_prompts(text)

        if image_inputs.get("pixel_values") is not None:
            expanded_image_prompts = []
            for sample in prompt_strings:
                sample = (
                    self.image_token * self.num_image_tokens
                    + self.tokenizer.bos_token
                    + sample
                    + self.tokenizer.eos_token
                )
                expanded_image_prompts.append(sample)
            prompt_strings = expanded_image_prompts

        output_kwargs["text_kwargs"].pop("add_special_tokens", None)
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
        text_inputs = self.tokenizer(
            prompt_strings, **output_kwargs["text_kwargs"], add_special_tokens=False, return_tensors=None
        )
        self._check_special_mm_tokens(prompt_strings, text_inputs, modalities=["image"])

        if return_mm_token_type_ids:
            text_inputs["mm_token_type_ids"] = self.create_mm_token_type_ids(text_inputs["input_ids"])
        return BatchFeature(data={**image_inputs, **text_inputs}, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        pass

    def _get_num_multimodal_tokens(self, image_sizes=None, **kwargs):
        pass

    def post_process_image_text_to_text(self, generated_outputs, skip_special_tokens=False, **kwargs):
        """
        Post-processes the output of `FuyuForConditionalGeneration` to only return the text output.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                containing the token ids of the generated sequences.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text output.
        """
        return self.batch_decode(generated_outputs, skip_special_tokens=skip_special_tokens, **kwargs)

    def post_process_generation(self, text=None, sequence=None, task=None, image_size=None) -> dict[str, Any]:
        """
        Post-process generation outputs based on the task.

        Args:
            text (`str`, *optional*):
                Generated text.
            sequence (`Union[List[int], torch.Tensor]`, *optional*):
                Generated token sequence.
            task (`str`, *optional*):
                The task for post-processing.
            image_size (`Tuple[int, int]`, *optional*):
                Image size for dequantization.

        Returns:
            `Dict[str, Any]`: Post-processed results keyed by task.
        """
        if task is None:
            raise ValueError("`task` must be provided for post-processing.")

        post_proc_type = self.tasks_answer_post_processing_type.get(task, "pure_text")
        parsed = self.post_processor(
            text=text,
            sequence=sequence,
            image_size=image_size,
            parse_tasks=[post_proc_type],
        )[post_proc_type]

        if post_proc_type == "pure_text":
            final_answer = parsed.replace("<s>", "").replace("</s>", "").strip()
        elif post_proc_type in ["description_with_bboxes", "bboxes"]:
            bboxes = [inst["bbox"] for inst in parsed]
            labels = [inst["cat_name"] for inst in parsed]
            final_answer = {"bboxes": bboxes, "labels": labels}
            if parsed and "score" in parsed[0]:
                final_answer["scores"] = [inst["score"] for inst in parsed]
        elif post_proc_type == "ocr":
            quad_boxes = [inst["quad_box"] for inst in parsed]
            labels = [inst["text"] for inst in parsed]
            final_answer = {"quad_boxes": quad_boxes, "labels": labels}
        elif post_proc_type == "phrase_grounding":
            bboxes = []
            labels = []
            for inst in parsed:
                for bbox in inst["bbox"]:
                    bboxes.append(bbox)
                    labels.append(inst["cat_name"])
            final_answer = {"bboxes": bboxes, "labels": labels}
        elif post_proc_type in ["description_with_polygons", "polygons"]:
            polygons = [inst["polygons"] for inst in parsed]
            labels = [inst["cat_name"] for inst in parsed]
            final_answer = {"polygons": polygons, "labels": labels}
        elif post_proc_type == "description_with_bboxes_or_polygons":
            bboxes = []
            bboxes_labels = []
            polygons = []
            polygons_labels = []
            for inst in parsed:
                label = inst["cat_name"]
                if "polygons" in inst:
                    polygons.append(inst["polygons"])
                    polygons_labels.append(label)
                else:
                    bboxes.append(inst["bbox"])
                    bboxes_labels.append(label)
            final_answer = {
                "bboxes": bboxes,
                "bboxes_labels": bboxes_labels,
                "polygons": polygons,
                "polygons_labels": polygons_labels,
            }
        else:
            raise ValueError(f"Unknown post-processing type: {post_proc_type}")

        return {task: final_answer}


class Florence2PostProcessor:

    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.parse_task_config = config or {}
        self.banned_grounding_tokens = set(
            self.parse_task_config.get("phrase_grounding", {}).get("banned_grounding_tokens", [])
        )
        self.all_special_tokens = set(self.tokenizer.all_special_tokens)
        self.quantize_bins = (1000, 1000)

    def quantize(self, locations: "torch.Tensor", size: tuple[int, int]) -> "torch.Tensor":
        """
        Quantize locations.

        Args:
            locations (`torch.Tensor`):
                Tensor of shape (N, 4) for boxes or (N, 2) for points/coordinates.
            size (`tuple[int, int]`):
                Original image size (width, height).

        Returns:
            `torch.Tensor`: Quantized locations as integers.
        """
        bins_w, bins_h = self.quantize_bins
        size_w, size_h = size
        per_bin_w = size_w / bins_w
        per_bin_h = size_h / bins_h

        if locations.shape[-1] == 4:  # Bounding boxes: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = locations.split(1, dim=-1)
            q_xmin = (xmin / per_bin_w).floor().clamp(0, bins_w - 1)
            q_ymin = (ymin / per_bin_h).floor().clamp(0, bins_h - 1)
            q_xmax = (xmax / per_bin_w).floor().clamp(0, bins_w - 1)
            q_ymax = (ymax / per_bin_h).floor().clamp(0, bins_h - 1)
            return torch.cat([q_xmin, q_ymin, q_xmax, q_ymax], dim=-1).int()

        elif locations.shape[-1] == 2:  # Points/coordinates: [x, y]
            x, y = locations.split(1, dim=-1)
            q_x = (x / per_bin_w).floor().clamp(0, bins_w - 1)
            q_y = (y / per_bin_h).floor().clamp(0, bins_h - 1)
            return torch.cat([q_x, q_y], dim=-1).int()

        else:
            raise ValueError(f"Unsupported location shape: last dim must be 2 or 4, got {locations.shape[-1]}.")

    def dequantize(self, locations: "torch.Tensor", size: tuple[int, int]) -> "torch.Tensor":
        """
        Dequantize locations back to original scale.

        Args:
            locations (`torch.Tensor`):
                Quantized tensor of shape (N, 4) for boxes or (N, 2) for points/coordinates.
            size (`tuple[int, int]`):
                Original image size (width, height).

        Returns:
            `torch.Tensor`: Dequantized locations as floats.
        """
        bins_w, bins_h = self.quantize_bins
        size_w, size_h = size
        per_bin_w = size_w / bins_w
        per_bin_h = size_h / bins_h

        if locations.shape[-1] == 4:  # Bounding boxes
            xmin, ymin, xmax, ymax = locations.split(1, dim=-1)
            dq_xmin = (xmin + 0.5) * per_bin_w
            dq_ymin = (ymin + 0.5) * per_bin_h
            dq_xmax = (xmax + 0.5) * per_bin_w
            dq_ymax = (ymax + 0.5) * per_bin_h
            return torch.cat([dq_xmin, dq_ymin, dq_xmax, dq_ymax], dim=-1).int()

        elif locations.shape[-1] == 2:  # Points/coordinates
            x, y = locations.split(1, dim=-1)
            dq_x = (x + 0.5) * per_bin_w
            dq_y = (y + 0.5) * per_bin_h
            return torch.cat([dq_x, dq_y], dim=-1).int()

        else:
            raise ValueError(f"Unsupported location shape: last dim must be 2 or 4, got {locations.shape[-1]}.")

    def decode_with_spans(self, token_ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
        pass

    def parse_ocr_from_text_and_spans(
        self, text: str, pattern: str | None, image_size: tuple[int, int], area_threshold: float = 0.0
    ) -> list[dict[str, Any]]:
        pass

    def parse_phrase_grounding_from_text_and_spans(
        self, text: str, image_size: tuple[int, int]
    ) -> list[dict[str, Any]]:
        pass

    def _find_matched_token_indices(self, cur_span: tuple[int, int], token_spans: list[tuple[int, int]]) -> list[int]:
        pass

    def parse_description_with_bboxes_from_text_and_spans(
        self,
        text: str,
        image_size: tuple[int, int],
        allow_empty_phrase: bool = False,
    ) -> list[dict[str, Any]]:
        pass

    def parse_description_with_polygons_from_text_and_spans(
        self,
        text: str,
        image_size: tuple[int, int],
        allow_empty_phrase: bool = False,
        polygon_sep_token: str = "<sep>",
        polygon_start_token: str = "<poly>",
        polygon_end_token: str = "</poly>",
        with_box_at_start: bool = False,
    ) -> list[dict[str, Any]]:
        pass

    def __call__(self, text=None, sequence=None, image_size=None, parse_tasks=None) -> dict[str, Any]:
        """
        Process model output and parse into task-specific results.

        Args:
            text (`Optional[str]`, *optional*):
                Generated text. Either this or `sequence` must be provided.
            sequence (`Optional[Union[list[int], torch.Tensor]]`, *optional*):
                Token sequence. Either this or `text` must be provided.
            image_size (`Optional[tuple[int, int]]`, *optional*):
                Image size (width, height) required for dequantization.
            parse_tasks (`Optional[Union[str, list[str]]]`, *optional*):
                Specific tasks to parse. If None, parse all supported tasks.

        Returns:
            `dict[str, Any]`: Parsed results for each task, including the raw 'text'.
        """
        if parse_tasks is not None:
            parse_tasks = [parse_tasks] if isinstance(parse_tasks, str) else parse_tasks
            for task in parse_tasks:
                if task not in self.parse_task_config.keys():
                    raise ValueError(f"Unsupported parse task: {task}")

        if (text is None and sequence is None) or (text is not None and sequence is not None):
            raise ValueError("Exactly one of 'text' or 'sequence' must be provided.")

        if sequence is not None:
            if isinstance(sequence, torch.Tensor):
                sequence = sequence.tolist()
            sequence = sequence[1:] if sequence[0] == self.tokenizer.bos_token_id else sequence  # Skip BOS if present
            text, _ = self.decode_with_spans(sequence)

        parsed_dict = {"text": text}

        tasks_to_parse = parse_tasks or self.parse_task_config.keys()
        for task in tasks_to_parse:
            config = self.parse_task_config[task]
            pattern = config.get("PATTERN")

            if task == "ocr":
                parsed_dict["ocr"] = self.parse_ocr_from_text_and_spans(
                    text, pattern=pattern, image_size=image_size, area_threshold=config.get("AREA_THRESHOLD", 0.0)
                )
            elif task == "phrase_grounding":
                parsed_dict["phrase_grounding"] = self.parse_phrase_grounding_from_text_and_spans(
                    text, image_size=image_size
                )
            elif task == "pure_text":
                parsed_dict["pure_text"] = text
            elif task == "description_with_bboxes":
                parsed_dict["description_with_bboxes"] = self.parse_description_with_bboxes_from_text_and_spans(
                    text, image_size=image_size
                )
            elif task == "description_with_polygons":
                parsed_dict["description_with_polygons"] = self.parse_description_with_polygons_from_text_and_spans(
                    text, image_size=image_size
                )
            elif task == "polygons":
                parsed_dict["polygons"] = self.parse_description_with_polygons_from_text_and_spans(
                    text, image_size=image_size, allow_empty_phrase=True
                )
            elif task == "bboxes":
                parsed_dict["bboxes"] = self.parse_description_with_bboxes_from_text_and_spans(
                    text, image_size=image_size, allow_empty_phrase=True
                )
            elif task == "description_with_bboxes_or_polygons":
                if "<poly>" in text:
                    instances = self.parse_description_with_polygons_from_text_and_spans(text, image_size=image_size)
                else:
                    instances = self.parse_description_with_bboxes_from_text_and_spans(text, image_size=image_size)
                parsed_dict["description_with_bboxes_or_polygons"] = instances
            else:
                raise ValueError(f"task {task} is not supported")

        return parsed_dict


__all__ = ["Florence2Processor"]
