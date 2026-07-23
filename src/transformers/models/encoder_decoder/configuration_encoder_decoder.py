

from huggingface_hub.dataclasses import strict

from ...configuration_utils import PreTrainedConfig
from ...utils import auto_docstring, logging
from ..auto import AutoConfig


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="")
@strict
class EncoderDecoderConfig(PreTrainedConfig):

    model_type = "encoder-decoder"
    sub_configs = {"encoder": AutoConfig, "decoder": AutoConfig}
    has_no_defaults_at_init = True

    pad_token_id: int | None = None
    decoder_start_token_id: int | None = None
    is_encoder_decoder: bool | None = True

    def __post_init__(self, **kwargs):
        if "encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError(
                f"A configuration of type {self.model_type} cannot be instantiated because not both `encoder` and"
                f" `decoder` sub-configurations are passed, but only {kwargs}"
            )

        encoder_config = kwargs.pop("encoder")
        encoder_model_type = encoder_config.pop("model_type")
        decoder_config = kwargs.pop("decoder")
        decoder_model_type = decoder_config.pop("model_type")

        self.encoder = AutoConfig.for_model(encoder_model_type, **encoder_config)
        self.decoder = AutoConfig.for_model(decoder_model_type, **decoder_config)
        super().__post_init__(**kwargs)

    @classmethod
    def from_encoder_decoder_configs(
        cls, encoder_config: PreTrainedConfig, decoder_config: PreTrainedConfig, **kwargs
    ) -> PreTrainedConfig:
        r"""
        Instantiate a [`EncoderDecoderConfig`] (or a derived class) from a pre-trained encoder model configuration and
        decoder model configuration.

        Returns:
            [`EncoderDecoderConfig`]: An instance of a configuration object
        """
        logger.info("Set `config.is_decoder=True` and `config.add_cross_attention=True` for decoder_config")
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True

        return cls(encoder=encoder_config.to_dict(), decoder=decoder_config.to_dict(), **kwargs)


__all__ = ["EncoderDecoderConfig"]
