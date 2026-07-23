
from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring
from ..auto.configuration_auto import AutoConfig


@auto_docstring(checkpoint="")
@strict
class RagConfig(PreTrainedConfig):

    model_type = "rag"
    has_no_defaults_at_init = True

    vocab_size: int | None = None
    is_encoder_decoder: bool = True
    prefix: str | None = None
    bos_token_id: int | None = None
    pad_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    decoder_start_token_id: int | None = None
    title_sep: str = " / "
    doc_sep: str = " // "
    n_docs: int = 5
    max_combined_length: int = 300
    retrieval_vector_size: int = 768
    retrieval_batch_size: int = 8
    dataset: str = "wiki_dpr"
    dataset_split: str = "train"
    index_name: str = "compressed"
    index_path: str | None = None
    passages_path: str | None = None
    use_dummy_dataset: bool = False
    reduce_loss: bool = False
    label_smoothing: float = 0.0
    do_deduplication: bool = True
    exclude_bos_score: bool = False
    do_marginalize: bool = False
    output_retrieved: bool = False
    use_cache: bool = True
    dataset_revision: str | None = None

    def __post_init__(self, **kwargs):
        if "question_encoder" not in kwargs or "generator" not in kwargs:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because not both `question_encoder` and"
                f" `generator` sub-configurations are passed, but only {kwargs}"
            )

        question_encoder_config = kwargs.pop("question_encoder")
        question_encoder_model_type = question_encoder_config.pop("model_type")
        decoder_config = kwargs.pop("generator")
        decoder_model_type = decoder_config.pop("model_type")

        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)

        super().__post_init__(**kwargs)

    @classmethod
    def from_question_encoder_generator_configs(
        cls, question_encoder_config: PreTrainedConfig, generator_config: PreTrainedConfig, **kwargs
    ) -> PreTrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)


__all__ = ["RagConfig"]
