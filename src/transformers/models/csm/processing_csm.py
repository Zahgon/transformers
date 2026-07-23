
import math
from pathlib import Path
from typing import Any

import numpy as np

from ...utils import auto_docstring, is_soundfile_available, is_torch_available
from ...utils.import_utils import requires, requires_backends


if is_torch_available():
    import torch

if is_soundfile_available():
    import soundfile as sf

from ...audio_utils import AudioInput, make_list_of_audio
from ...feature_extraction_utils import BatchFeature
from ...processing_utils import AudioKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput


class CsmAudioKwargs(AudioKwargs, total=False):

    encoded_length_kwargs: dict[str, Any] | None


class CsmProcessorKwargs(ProcessingKwargs, total=False):
    audio_kwargs: CsmAudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        },
        "audio_kwargs": {
            "encoded_length_kwargs": {
                "kernel_sizes": [7, 3, 1, 8, 3, 1, 10, 3, 1, 12, 3, 1, 16, 3, 4],
                "strides": [1, 1, 1, 4, 1, 1, 5, 1, 1, 6, 1, 1, 8, 1, 2],
                "dilations": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                "use_causal_conv": True,
            },
            "sampling_rate": 24000,
        },
        "common_kwargs": {"return_tensors": "pt"},
    }


@requires(backends=("torch",))
@auto_docstring
class CsmProcessor(ProcessorMixin):
    def __init__(
        self,
        feature_extractor,
        tokenizer,
        chat_template=None,
    ):
        if not hasattr(tokenizer, "audio_token"):
            self.audio_token = "<|AUDIO|>"
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)
        else:
            self.audio_token = tokenizer.audio_token
            self.audio_token_id = tokenizer.audio_token_id

        if not hasattr(tokenizer, "audio_eos_token"):
            self.audio_eos_token = "<|audio_eos|>"
            self.audio_eos_token_id = tokenizer.convert_tokens_to_ids(self.audio_eos_token)
        else:
            self.audio_eos_token = tokenizer.audio_eos_token
            self.audio_eos_token_id = tokenizer.audio_eos_token_id

        super().__init__(feature_extractor, tokenizer, chat_template=chat_template)

    @staticmethod
    def _get_encoded_length(audio_length, kernel_sizes=None, strides=None, dilations=None, use_causal_conv=None):
        pass

    def save_audio(
        self,
        audio: AudioInput,
        saving_path: str | Path | list[str | Path],
        **kwargs: Unpack[CsmProcessorKwargs],
    ):
        pass

    @auto_docstring
    def __call__(
        self,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None,
        audio: AudioInput | None = None,
        output_labels: bool | None = False,
        depth_decoder_labels_ratio: float | None = 1.0,
        **kwargs: Unpack[CsmProcessorKwargs],
    ):
        r"""
        output_labels (bool, *optional*, default=False):
            Whether to return labels for training. Indices will be in `[config.audio_token_id, -100, -101]`.
            - `config.audio_token_id` indicates an audio frame (considering sequence length elements as frames)
            - `-100` will be ignored in the loss computation
            - `-101` indicates the audio frame will be used only for the backbone model (using the first codebook token as labels)
        depth_decoder_labels_ratio (float, *optional*, default=1.0):
            The ratio of audio frames to keep for the depth decoder labels.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **input_values** -- List of audio values to be fed to a model. Returned when `audio` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **labels** -- List of labels for the audio frames. Returned when `output_labels=True`.
        """

        output_kwargs = self._merge_kwargs(
            CsmProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        text_kwargs = output_kwargs["text_kwargs"]
        audio_kwargs = output_kwargs["audio_kwargs"]
        return_tensors = text_kwargs.get("return_tensors", None)
        if return_tensors != "pt":
            raise ValueError(f"{self.__class__.__name__} only supports `return_tensors='pt'`.")

        if isinstance(text, str):
            text = [text]
        elif not (isinstance(text, (list, tuple)) and all(isinstance(t, str) for t in text)):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")
        n_audio_in_text = [t.count(self.audio_token) for t in text]

        n_audio = 0
        if audio is not None:
            audio = make_list_of_audio(audio)
            n_audio = len(audio)

        if sum(n_audio_in_text) > 0 and n_audio != sum(n_audio_in_text):
            if audio is None:
                raise ValueError("No audio were provided, but there are audio tokens in the prompt")
            else:
                raise ValueError(
                    f"The number of audio tokens in each text ({n_audio_in_text}) should be the same as the "
                    f"number of provided audios ({n_audio})."
                )

        if audio is not None:
            encoded_length_kwargs = audio_kwargs.pop("encoded_length_kwargs", {})
            num_audio_tokens_list = [
                self._get_encoded_length(audio_array.shape[-1], **encoded_length_kwargs) for audio_array in audio
            ]
            num_audio_tokens_list_copy = num_audio_tokens_list.copy()

            expanded_text = []
            for sample in text:
                replace_str = []
                while self.audio_token in sample:
                    num_audio_tokens = num_audio_tokens_list_copy.pop(0)
                    expanded_audio_token = self.audio_token * num_audio_tokens

                    replace_str.append(expanded_audio_token)
                    sample = sample.replace(self.audio_token, "<placeholder>", 1)

                while "<placeholder>" in sample:
                    sample = sample.replace("<placeholder>", replace_str.pop(0), 1)
                expanded_text.append(sample)

            text = expanded_text

        encoding = self.tokenizer(text, **text_kwargs)
        data = {}
        data.update(encoding)

        if audio is not None:
            audio_kwargs.pop("return_attention_mask", None)  # not supported by the feature extractor

            concatenated_audio, input_values_cutoffs = [], []
            offset = 0
            for n_audio in n_audio_in_text:
                if n_audio == 0:
                    concatenated_audio.append(np.zeros(0))
                    input_values_cutoffs.append(torch.tensor([-1]))
                else:
                    concatenated_audio.append(
                        np.concatenate(
                            [
                                el.cpu().numpy() if isinstance(el, torch.Tensor) else el
                                for el in audio[offset : offset + n_audio]
                            ],
                            axis=-1,
                        )
                    )
                    input_values_cutoffs.append(
                        torch.tensor([el.shape[-1] for el in audio[offset : offset + n_audio]]).cumsum(dim=-1)
                    )
                    offset += n_audio

            audio_inputs = self.feature_extractor(concatenated_audio, **audio_kwargs)
            audio_inputs.pop("padding_mask", None)  # not applicable here
            data.update(audio_inputs)

            max_len = max(cut_idxs.shape[-1] for cut_idxs in input_values_cutoffs)
            input_values_cutoffs = [
                torch.nn.functional.pad(cut_idxs, (0, max_len - cut_idxs.shape[-1]), value=-1)
                for cut_idxs in input_values_cutoffs
            ]
            data["input_values_cutoffs"] = torch.stack(input_values_cutoffs, dim=0)

        if output_labels:
            audio_frame_idxs = (data["input_ids"] == self.audio_token_id).nonzero()
            n_audio_frames = audio_frame_idxs.shape[0]

            if depth_decoder_labels_ratio <= 1.0:
                rand_idxs = torch.randperm(n_audio_frames)[: int(n_audio_frames * (1 - depth_decoder_labels_ratio))]
                skip_frames_idxs = audio_frame_idxs[rand_idxs]
            else:
                skip_frames_idxs = audio_frame_idxs

            labels = torch.where(
                (data["input_ids"] == self.audio_token_id) | (data["input_ids"] == self.audio_eos_token_id),
                data["input_ids"],
                -100,
            )
            labels[skip_frames_idxs[:, 0], skip_frames_idxs[:, 1]] = -101

            data["labels"] = labels

        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def model_input_names(self):
        pass


__all__ = ["CsmProcessor"]
