
from dataclasses import dataclass

import torch

from .cache_utils import Cache, EncoderDecoderCache
from .utils import ModelOutput


@dataclass
class BaseModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithNoAttention(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithPooling(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithPoolingAndNoAttention(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithPast(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithCrossAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithPoolingAndCrossAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    past_key_values: Cache | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithPastAndCrossAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class MoEModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_probs: tuple[torch.FloatTensor] | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
class MoeModelOutputWithPast(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
class MoeCausalLMOutputWithPast(ModelOutput):

    loss: torch.FloatTensor | None = None
    aux_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
class MoEModelOutputWithPastAndCrossAttentions(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    router_probs: tuple[torch.FloatTensor] | None = None
    router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
class Seq2SeqModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Seq2SeqMoEModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    decoder_router_logits: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
class CausalLMOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class CausalLMOutputWithPast(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class CausalLMOutputWithCrossAttentions(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class SequenceClassifierOutputWithPast(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class MaskedLMOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Seq2SeqLMOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Seq2SeqMoEOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    encoder_z_loss: torch.FloatTensor | None = None
    decoder_z_loss: torch.FloatTensor | None = None
    encoder_aux_loss: torch.FloatTensor | None = None
    decoder_aux_loss: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    decoder_router_logits: tuple[torch.FloatTensor] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_router_logits: tuple[torch.FloatTensor] | None = None


@dataclass
class NextSentencePredictorOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Seq2SeqSequenceClassifierOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class MultipleChoiceModelOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class TokenClassifierOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class QuestionAnsweringModelOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    start_logits: torch.FloatTensor | None = None
    end_logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Seq2SeqQuestionAnsweringModelOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    start_logits: torch.FloatTensor | None = None
    end_logits: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class SemanticSegmenterOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ImageClassifierOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ImageClassifierOutputWithNoAttention(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class DepthEstimatorOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    predicted_depth: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class ImageSuperResolutionOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    reconstruction: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Wav2Vec2BaseModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    extract_features: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class XVectorOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    embeddings: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BackboneOutput(ModelOutput):

    feature_maps: tuple[torch.FloatTensor] | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class BaseModelOutputWithPoolingAndProjection(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    pooler_output: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    projection_state: tuple[torch.FloatTensor] | None = None


@dataclass
class Seq2SeqSpectrogramOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    spectrogram: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None


@dataclass
class Seq2SeqTSModelOutput(ModelOutput):

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    loc: torch.FloatTensor | None = None
    scale: torch.FloatTensor | None = None
    static_features: torch.FloatTensor | None = None


@dataclass
class Seq2SeqTSPredictionOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    params: tuple[torch.FloatTensor, ...] | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    loc: torch.FloatTensor | None = None
    scale: torch.FloatTensor | None = None
    static_features: torch.FloatTensor | None = None


@dataclass
class SampleTSPredictionOutput(ModelOutput):

    sequences: torch.FloatTensor | None = None


@dataclass
class MaskedImageModelingOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    reconstruction: torch.FloatTensor | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


MoECausalLMOutputWithPast = MoeCausalLMOutputWithPast
