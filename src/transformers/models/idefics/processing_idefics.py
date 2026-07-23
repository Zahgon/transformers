
from urllib.parse import urlparse

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import (
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
)
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, is_torch_available


if is_torch_available():
    import torch


IMAGE_TOKEN = "<image>"


class IdeficsTextKwargs(TextKwargs, total=False):

    add_eos_token: bool | None
    add_end_of_utterance_token: bool | None


class IdeficsProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: IdeficsTextKwargs
    _defaults = {
        "text_kwargs": {
            "add_special_tokens": False,
            "padding": "longest",
            "add_eos_token": False,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


def incremental_to_binary_attention_mask(incremental_mask, return_tensors, num_classes=-1):
    pass


def image_attention_mask_for_packed_input_ids(input_ids, tokenizer, return_tensors):
    pass


def image_attention_mask_for_packed_input_ids_pt(input_ids, tokenizer):
    pass


def is_url(string):
    pass


@auto_docstring
class IdeficsProcessor(ProcessorMixin):
    def __init__(self, image_processor, tokenizer=None, image_size=224, add_end_of_utterance_token=None, **kwargs):
        r"""
        image_size (int, *optional*, defaults to 224):
            The size of the image to be processed.
        add_end_of_utterance_token (bool, *optional*, defaults to None):
            Whether to add the end of utterance token to the text.
        """
        super().__init__(image_processor, tokenizer)
        self.image_token_id = (
            tokenizer.image_token_id
            if hasattr(tokenizer, "image_token")
            else tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        )

        self.default_image_dims = (
            self.image_processor.image_num_channels,
            self.image_processor.image_size,
            self.image_processor.image_size,
        )

        self.tokenizer_was_trained_with_end_of_utterance_token = (
            "<end_of_utterance>" in self.tokenizer.special_tokens_map.get("additional_special_tokens", [])
        )

    @auto_docstring
    def __call__(
        self,
        images: ImageInput | list[ImageInput] | str | list[str] | list[list[str]] = None,
        text: TextInput
        | PreTokenizedInput
        | list[TextInput]
        | list[PreTokenizedInput]
        | list[list[TextInput]]
        | list[list[PreTokenizedInput]] = None,
        **kwargs: Unpack[IdeficsProcessorKwargs],
    ) -> BatchFeature:
        r"""
        Returns:
            a dict with entries: `input_ids`, `attention_mask`, `pixel_values`, `image_attention_mask` which can be
            directly passed to `model.generate`

            Detailed explanation:

            Each entry in `text` is either a text to be passed as is or an image that will be processed.

            An image can be either an image object (`PIL.Image`) or a url from which the image can be retrieved.

        When the processor encounters an image it'll inject `<fake_token_around_image><image><fake_token_around_image>`
        entry into the prompt.

        Example:

        ```python
        checkpoint = "HuggingFaceM4/idefics-9b"
        processor = AutoProcessor.from_pretrained(checkpoint)
        url = "https://hips.hearstapps.com/hmg-prod/images/cute-photos-of-cats-in-grass-1593184777.jpg"
        img = processor.image_processor.fetch_images([url])[0]

        prompts = [
            "User:",
            img,
            "Describe this image.\nAssistant: An image of two kittens in grass.\n",
            "User:",
            "https://hips.hearstapps.com/hmg-prod/images/dog-puns-1581708208.jpg",
            "Describe this image.\nAssistant:",
        ]

        inputs = processor(text=prompts, return_tensors="pt")
        generated_ids = model.generate(**inputs, max_length=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        ```

        In this example the `prompts` will be converted into:

        ```
        <s>User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
        Assistant: An image of two kittens in grass.
        User:<fake_token_around_image><image><fake_token_around_image>Describe this image.
        Assistant:'
        ```

        and the two images will be massaged using [`IdeficsImageProcessor.__call__`] method and placed inside the
        `pixel_values` dict entry of the return value.

        This example also exemplifies that images can be passed as objects or as text urls. It can be seen that the
        first image is passed as object and the second one as a url.

        To do training do:

        ```python
        image_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (w, h), scale=(0.9, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.image_mean, std=self.image_std),
            ]
        )
        inputs = processor(text=prompts, transform=image_transform, return_tensors="pt")
        ```

        In order to help debug prompt generation enable `debug=True` which will show you what's happening.

        """
        if images is None and text is None:
            raise ValueError("You need to specify either `text` or `images` and `text`.")

        if images is None:
            prompts = text
        elif text is not None:
            if not isinstance(images, (list, tuple)):
                images = [images]
            if isinstance(text, str):
                text = [text]
            if isinstance(text, (list, tuple)) and len(text) != len(images):
                raise ValueError(
                    "When providing both images and text arguments, the number of text prompts should be the same as the number of images."
                    "If you want to have several images per prompt, images should be nested as such: images=[[img1, img2], [img3, img4], ...] for text=[prompt1, prompt2, ...]."
                )
            if not all(isinstance(i, str) for i in text):
                raise ValueError("When using the image-text-to-text behavior, the prompts should only contain text.")
            if isinstance(images[0], (list, tuple)):
                prompts = [[sample, *image_list] for image_list, sample in zip(images, text)]
            else:
                prompts = list(zip(images, text))

        output_kwargs = self._merge_kwargs(
            IdeficsProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        add_eos_token = output_kwargs["text_kwargs"].pop("add_eos_token", False)
        add_end_of_utterance_token = output_kwargs["text_kwargs"].pop("add_end_of_utterance_token", None)

        if add_end_of_utterance_token is None:
            add_end_of_utterance_token = self.tokenizer_was_trained_with_end_of_utterance_token
        if not any(isinstance(i, (list, tuple)) for i in prompts):
            prompts = [prompts]

        fake_token = "<fake_token_around_image>"
        image_token = "<image>"
        end_of_utterance_token = "<end_of_utterance>"

        def image_tokens(last_was_image):
            pass

        all_prompts = []
        all_images = []
        for sample in prompts:
            full_text = f"{self.tokenizer.bos_token}"

            image_objects = []
            last_was_image = False
            last_was_text = False
            for i, item in enumerate(sample):
                if i > 0:
                    last_was_text = bool(not last_was_image)

                if isinstance(item, str):
                    item = item.strip(" ")
                    if is_url(item):
                        image = self.image_processor.fetch_images(item)
                        full_text += image_tokens(last_was_image)
                        image_objects.append(image)
                        last_was_image = True
                    else:
                        if add_end_of_utterance_token and last_was_text:
                            full_text += end_of_utterance_token
                        full_text += item
                        last_was_image = False
                else:
                    full_text += image_tokens(last_was_image)
                    image_objects.append(item)
                    last_was_image = True

            if add_eos_token:
                full_text += self.tokenizer.eos_token

            if len(image_objects) > 0:
                image_objects = self.image_processor(image_objects, **output_kwargs["images_kwargs"])

            all_prompts.append(full_text)
            all_images.append(image_objects)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", "pt")
        text_encoding = self.tokenizer(all_prompts, **output_kwargs["text_kwargs"])
        all_texts = text_encoding["input_ids"]
        all_attention_masks = text_encoding["attention_mask"]

        max_num_images = max(len(x) for x in all_images)
        max_num_images = max(1, max_num_images)

        at_least_one_image = sum(len(x) for x in all_images) > 0
        output_input_ids = []
        output_images = []
        output_attention_masks = []

        for text_single, attention_mask, extracted_images in zip(all_texts, all_attention_masks, all_images):
            padded_input_ids = text_single
            image_count = padded_input_ids.count(self.image_token_id)
            local_max_num_images = min(image_count, max_num_images)
            current_images = extracted_images[:local_max_num_images]

            if len(current_images) > 0:
                if return_tensors == "pt":
                    padded_image_tensor = torch.zeros(max_num_images, *current_images.size()[1:])
                    padded_image_tensor[: current_images.size(0)] = current_images
            else:
                if return_tensors == "pt":
                    padded_image_tensor = torch.zeros(max_num_images, *self.default_image_dims)

            output_images.append(padded_image_tensor)
            if return_tensors == "pt":
                output_input_ids.append(torch.tensor(padded_input_ids))
                output_attention_masks.append(torch.tensor(attention_mask))

        if return_tensors == "pt":
            output_input_ids = torch.stack(output_input_ids)
            output_images = torch.stack(output_images)
            output_attention_masks = torch.stack(output_attention_masks)

        if at_least_one_image:
            image_attention_mask, _ = image_attention_mask_for_packed_input_ids(
                output_input_ids, self.tokenizer, return_tensors
            )
            image_attention_mask = incremental_to_binary_attention_mask(
                image_attention_mask, return_tensors, num_classes=max_num_images
            )
        else:
            if return_tensors == "pt":
                image_attention_mask = torch.zeros(
                    output_input_ids.shape[0], output_input_ids.shape[1], 1, dtype=torch.bool
                )
        return BatchFeature(
            data={
                "input_ids": output_input_ids,
                "attention_mask": output_attention_masks,
                "pixel_values": output_images,
                "image_attention_mask": image_attention_mask,
            }
        )

    @property
    def model_input_names(self):
        pass


__all__ = ["IdeficsProcessor"]
