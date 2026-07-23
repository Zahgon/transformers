
import torch
from huggingface_hub.dataclasses import strict
from torch import nn

from ...modeling_outputs import BaseModelOutputWithPooling
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring
from ...utils.generic import merge_with_config_defaults
from ...utils.output_capturing import capture_outputs
from ..granite_speech.configuration_granite_speech import GraniteSpeechConfig, GraniteSpeechEncoderConfig
from ..granite_speech.modeling_granite_speech import (
    GraniteSpeechCTCEncoder,
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechModel,
    GraniteSpeechPreTrainedModel,
)


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-plus")
@strict
class GraniteSpeechPlusEncoderConfig(GraniteSpeechEncoderConfig):

    cat_hidden_layers: list[int] | None = None


@auto_docstring(checkpoint="ibm-granite/granite-speech-4.1-2b-plus")
@strict
class GraniteSpeechPlusConfig(GraniteSpeechConfig):

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        if self.encoder_config.cat_hidden_layers is not None:
            for idx in self.encoder_config.cat_hidden_layers:
                if idx < 0 or idx >= self.encoder_config.num_layers:
                    raise ValueError(
                        f"cat_hidden_layers index {idx} is out of range [0, {self.encoder_config.num_layers})."
                    )
        if self.encoder_config.cat_hidden_layers is not None:
            num_concat = len(self.encoder_config.cat_hidden_layers) + 1
            if self.projector_config.encoder_hidden_size != self.encoder_config.hidden_dim * num_concat:
                raise ValueError(
                    f"projector encoder_hidden_size {self.projector_config.encoder_hidden_size} "
                    f"must equal encoder hidden_dim * {num_concat} = "
                    f"{self.encoder_config.hidden_dim * num_concat}."
                )


class GraniteSpeechPlusPreTrainedModel(GraniteSpeechPreTrainedModel): ...


class GraniteSpeechPlusModel(GraniteSpeechModel): ...


class GraniteSpeechPlusCTCEncoder(GraniteSpeechCTCEncoder):
    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPooling:
        hidden_states = self.input_linear(hidden_states)
        cat_layers = set(self.config.cat_hidden_layers or [])
        exported_hidden_states = []

        if 0 in cat_layers:
            exported_hidden_states.append(hidden_states)

        for idx, layer in enumerate(self.layers, start=1):
            hidden_states = layer(hidden_states, attention_dists=self.attention_dists)

            if idx in cat_layers:
                exported_hidden_states.append(hidden_states)

            if idx == self.num_layers // 2:
                hidden_states_mid = hidden_states.clone()
                hidden_states_mid = self.out(hidden_states_mid)
                hidden_states += self.out_mid(nn.Softmax(dim=-1)(hidden_states_mid))

        if exported_hidden_states:
            hidden_states = torch.cat([*exported_hidden_states, hidden_states], dim=-1)

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


@auto_docstring(
    custom_intro="""
    The Granite Speech Plus model, a Granite Speech variant whose projector consumes the concatenation of the
    encoder's final hidden states with an arbitrary subset of its intermediate hidden states.
    """
)
class GraniteSpeechPlusForConditionalGeneration(GraniteSpeechForConditionalGeneration): ...


__all__ = [
    "GraniteSpeechPlusConfig",
    "GraniteSpeechPlusEncoderConfig",
    "GraniteSpeechPlusModel",
    "GraniteSpeechPlusCTCEncoder",
    "GraniteSpeechPlusForConditionalGeneration",
    "GraniteSpeechPlusPreTrainedModel",
]
