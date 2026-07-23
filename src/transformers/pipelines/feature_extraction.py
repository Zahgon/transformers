from typing import Any

from ..utils import add_end_docstrings
from .base import GenericTensor, Pipeline, build_pipeline_init_args


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True, supports_binary_output=False),
    r"""
        tokenize_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the tokenizer.
        return_tensors (`bool`, *optional*):
            If `True`, returns a tensor according to the specified framework, otherwise returns a list.""",
)
class FeatureExtractionPipeline(Pipeline):

    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation

        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, **tokenize_kwargs) -> dict[str, GenericTensor]:
        model_inputs = self.tokenizer(inputs, return_tensors="pt", **tokenize_kwargs)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, return_tensors=False):
        if return_tensors:
            return model_outputs[0]
        return model_outputs[0].tolist()

    def __call__(self, *args: str | list[str], **kwargs: Any) -> Any | list[Any]:
        """
        Extract the features of the input(s) text.

        Args:
            args (`str` or `list[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)
