from collections import UserDict
from typing import Any

import httpx
import numpy as np

from ..utils import (
    add_end_docstrings,
    logging,
)
from .audio_classification import ffmpeg_read
from .base import Pipeline, build_pipeline_init_args


logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_feature_extractor=True, has_tokenizer=True))
class ZeroShotAudioClassificationPipeline(Pipeline):

    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = True
    _load_tokenizer = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, audios: np.ndarray | bytes | str | dict, **kwargs: Any) -> list[dict[str, Any]]:
        """
        Assign labels to the audio(s) passed as inputs.

        Args:
            audios (`str`, `list[str]`, `np.array` or `list[np.array]`):
                The pipeline handles three types of inputs:
                - A string containing a http link pointing to an audio
                - A string containing a local path to an audio
                - An audio loaded in numpy
            candidate_labels (`list[str]`):
                The candidate labels for this audio. They will be formatted using *hypothesis_template*.
            hypothesis_template (`str`, *optional*, defaults to `"This is a sound of {}"`):
                The format used in conjunction with *candidate_labels* to attempt the audio classification by
                replacing the placeholder with the candidate_labels. Pass "{}" if *candidate_labels* are
                already formatted.
        Return:
            A list of dictionaries containing one entry per proposed label. Each dictionary contains the
            following keys:
            - **label** (`str`) -- One of the suggested *candidate_labels*.
            - **score** (`float`) -- The score attributed by the model to that label. It is a value between
                0 and 1, computed as the `softmax` of `logits_per_audio`.
        """
        return super().__call__(audios, **kwargs)

    def _sanitize_parameters(self, **kwargs):
        preprocess_params = {}
        if "candidate_labels" in kwargs:
            preprocess_params["candidate_labels"] = kwargs["candidate_labels"]
        if "hypothesis_template" in kwargs:
            preprocess_params["hypothesis_template"] = kwargs["hypothesis_template"]

        return preprocess_params, {}, {}

    def preprocess(self, audio, candidate_labels=None, hypothesis_template="This is a sound of {}."):
        if isinstance(audio, str):
            if audio.startswith("http://") or audio.startswith("https://"):
                audio = httpx.get(audio, follow_redirects=True).content
            else:
                with open(audio, "rb") as f:
                    audio = f.read()

        if isinstance(audio, bytes):
            audio = ffmpeg_read(audio, self.feature_extractor.sampling_rate)

        if not isinstance(audio, np.ndarray):
            raise TypeError("We expect a numpy ndarray as input")
        if len(audio.shape) != 1:
            raise ValueError("We expect a single channel audio input for ZeroShotAudioClassificationPipeline")

        inputs = self.feature_extractor(
            [audio], sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt"
        )
        inputs = inputs.to(self.dtype)
        inputs["candidate_labels"] = candidate_labels
        sequences = [hypothesis_template.format(x) for x in candidate_labels]
        text_inputs = self.tokenizer(sequences, return_tensors="pt", padding=True)
        inputs["text_inputs"] = [text_inputs]
        return inputs

    def _forward(self, model_inputs):
        candidate_labels = model_inputs.pop("candidate_labels")
        text_inputs = model_inputs.pop("text_inputs")
        if isinstance(text_inputs[0], UserDict):
            text_inputs = text_inputs[0]
        else:
            text_inputs = text_inputs[0][0]

        outputs = self.model(**text_inputs, **model_inputs)

        model_outputs = {
            "candidate_labels": candidate_labels,
            "logits": outputs.logits_per_audio,
        }
        return model_outputs

    def postprocess(self, model_outputs):
        candidate_labels = model_outputs.pop("candidate_labels")
        logits = model_outputs["logits"][0]

        probs = logits.softmax(dim=0)
        scores = probs.tolist()

        result = [
            {"score": score, "label": candidate_label}
            for score, candidate_label in sorted(zip(scores, candidate_labels), key=lambda x: -x[0])
        ]
        return result
