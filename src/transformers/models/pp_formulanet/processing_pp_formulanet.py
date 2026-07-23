
import re

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...utils import auto_docstring, logging


logger = logging.get_logger(__name__)


@auto_docstring
class PPFormulaNetProcessor(ProcessorMixin):

    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

        self._text_reg = re.compile(r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})")
        self._macro_pattern = re.compile(r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})")
        self._protected_macros = {"\\operatorname", "\\mathrm", "\\text", "\\mathbf"}

        letter = r"[a-zA-Z]"
        noletter = r"[\W_^\d]"
        self._rule_noletter_noletter = re.compile(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter))
        self._rule_noletter_letter = re.compile(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter))
        self._rule_letter_noletter = re.compile(r"(%s)\s+?(%s)" % (letter, noletter))

        self._chinese_text_wrapping_pattern = re.compile(r"\\text\s*{([^{}]*[\u4e00-\u9fff]+[^{}]*)}")

    @auto_docstring
    def __call__(
        self,
        images: ImageInput,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchFeature:
        r"""
        images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
            The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
            tensor. Both channels-first and channels-last formats are supported.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            ProcessingKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
        return BatchFeature({**image_inputs})

    def post_process_generation(self, text: str) -> str:
        """Post-processes a string by fixing text and normalizing it.

        Args:
            text (str): String to post-process.

        Returns:
            str: Post-processed string.
        """
        text = self.remove_chinese_text_wrapping(text)
        try:
            from ftfy import fix_text

            text = fix_text(text)
        except ImportError:
            logger.warning_once(
                "ftfy is not installed, skipping fix_text. "
                "Output may contain unnormalized unicode, extra spaces, or escaped artifacts"
            )
        text = self.normalize(text)
        return text

    def normalize(self, text: str) -> str:
        """Normalizes a string by removing unnecessary spaces."""
        names = []
        for x in self._text_reg.findall(text):
            matches = self._macro_pattern.findall(x[0])
            for m in matches:
                if m not in self._protected_macros and m.strip() != "":
                    text = text.replace(m, m + "XXXXXXX")
                    text = text.replace(" ", "")
                    names.append(text)

        if names:
            text = self._text_reg.sub(lambda match: str(names.pop(0)), text)

        new_text = text
        while True:
            text = new_text
            new_text = self._rule_noletter_noletter.sub(r"\1\2", text)
            new_text = self._rule_noletter_letter.sub(r"\1\2", new_text)
            new_text = self._rule_letter_noletter.sub(r"\1\2", new_text)
            if new_text == text:
                break

        return new_text.replace("XXXXXXX", " ")

    def remove_chinese_text_wrapping(self, formula: str) -> str:
        def replacer(match):
            pass

        replaced_formula = self._chinese_text_wrapping_pattern.sub(replacer, formula)
        return replaced_formula.replace('"', "")

    def post_process(self, generated_outputs, skip_special_tokens=True, **kwargs):
        pass


__all__ = ["PPFormulaNetProcessor"]
