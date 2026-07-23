
import collections
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch
from torch import nn
from torch.nn import BCELoss

from .. import initialization as init
from ..configuration_utils import PreTrainedConfig
from ..modeling_utils import PreTrainedModel
from ..utils import ModelOutput, logging
from .logits_process import SynthIDTextWatermarkLogitsProcessor, WatermarkLogitsProcessor


if TYPE_CHECKING:
    from .configuration_utils import WatermarkingConfig

logger = logging.get_logger(__name__)


@dataclass
class WatermarkDetectorOutput:

    num_tokens_scored: np.ndarray | None = None
    num_green_tokens: np.ndarray | None = None
    green_fraction: np.ndarray | None = None
    z_score: np.ndarray | None = None
    p_value: np.ndarray | None = None
    prediction: np.ndarray | None = None
    confidence: np.ndarray | None = None


class WatermarkDetector:

    def __init__(
        self,
        model_config: "PreTrainedConfig",
        device: str,
        watermarking_config: Union["WatermarkingConfig", dict],
        ignore_repeated_ngrams: bool = False,
        max_cache_size: int = 128,
    ):
        if not isinstance(watermarking_config, dict):
            watermarking_config = watermarking_config.to_dict()

        self.bos_token_id = (
            model_config.bos_token_id if not model_config.is_encoder_decoder else model_config.decoder_start_token_id
        )
        self.greenlist_ratio = watermarking_config["greenlist_ratio"]
        self.ignore_repeated_ngrams = ignore_repeated_ngrams
        self.processor = WatermarkLogitsProcessor(
            vocab_size=model_config.vocab_size, device=device, **watermarking_config
        )

        self._get_ngram_score_cached = lru_cache(maxsize=max_cache_size)(self._get_ngram_score)

    def _get_ngram_score(self, prefix: torch.LongTensor, target: int):
        pass

    def _score_ngrams_in_passage(self, input_ids: torch.LongTensor):
        pass

    def _compute_z_score(self, green_token_count: np.ndarray, total_num_tokens: np.ndarray) -> np.ndarray:
        pass

    def _compute_pval(self, x, loc=0, scale=1):
        pass

    def __call__(
        self,
        input_ids: torch.LongTensor,
        z_threshold: float = 3.0,
        return_dict: bool = False,
    ) -> WatermarkDetectorOutput | np.ndarray:
        """
                Args:
                input_ids (`torch.LongTensor`):
                    The watermark generated text. It is advised to remove the prompt, which can affect the detection.
                z_threshold (`Dict`, *optional*, defaults to `3.0`):
                    Changing this threshold will change the sensitivity of the detector. Higher z threshold gives less
                    sensitivity and vice versa for lower z threshold.
                return_dict (`bool`,  *optional*, defaults to `False`):
                    Whether to return `~generation.WatermarkDetectorOutput` or not. If not it will return boolean predictions,
        ma
                Return:
                    [`~generation.WatermarkDetectorOutput`] or `np.ndarray`: A [`~generation.WatermarkDetectorOutput`]
                    if `return_dict=True` otherwise a `np.ndarray`.

        """

        if input_ids[0, 0] == self.bos_token_id:
            input_ids = input_ids[:, 1:]

        if input_ids.shape[-1] - self.processor.context_width < 1:
            raise ValueError(
                f"Must have at least `1` token to score after the first "
                f"min_prefix_len={self.processor.context_width} tokens required by the seeding scheme."
            )

        num_tokens_scored, green_token_count = self._score_ngrams_in_passage(input_ids)
        z_score = self._compute_z_score(green_token_count, num_tokens_scored)
        prediction = z_score > z_threshold

        if return_dict:
            p_value = self._compute_pval(z_score)
            confidence = 1 - p_value

            return WatermarkDetectorOutput(
                num_tokens_scored=num_tokens_scored,
                num_green_tokens=green_token_count,
                green_fraction=green_token_count / num_tokens_scored,
                z_score=z_score,
                p_value=p_value,
                prediction=prediction,
                confidence=confidence,
            )
        return prediction


class BayesianDetectorConfig(PreTrainedConfig):

    def __init__(self, watermarking_depth: int | None = None, base_rate: float = 0.5, **kwargs):
        self.watermarking_depth = watermarking_depth
        self.base_rate = base_rate
        self.model_name = None
        self.watermarking_config = None

        super().__init__(**kwargs)

    def set_detector_information(self, model_name, watermarking_config):
        pass


@dataclass
class BayesianWatermarkDetectorModelOutput(ModelOutput):

    loss: torch.FloatTensor | None = None
    posterior_probabilities: torch.FloatTensor | None = None


class BayesianDetectorWatermarkedLikelihood(nn.Module):

    def __init__(self, watermarking_depth: int):
        """Initializes the model parameters."""
        super().__init__()
        self.watermarking_depth = watermarking_depth
        self.beta = torch.nn.Parameter(-2.5 + 0.001 * torch.randn(1, 1, watermarking_depth))
        self.delta = torch.nn.Parameter(0.001 * torch.randn(1, 1, self.watermarking_depth, watermarking_depth))

    def _compute_latents(self, g_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the unique token probability distribution given g-values.

        Args:
            g_values (`torch.Tensor` of shape `(batch_size, seq_len, watermarking_depth)`):
                PRF values.

        Returns:
            p_one_unique_token and p_two_unique_tokens, both of shape
            [batch_size, seq_len, watermarking_depth]. p_one_unique_token[i,t,l]
            gives the probability of there being one unique token in a tournament
            match on layer l, on timestep t, for batch item i.
            p_one_unique_token[i,t,l] + p_two_unique_token[i,t,l] = 1.
        """

        x = torch.repeat_interleave(torch.unsqueeze(g_values, dim=-2), self.watermarking_depth, axis=-2)

        x = torch.tril(x, diagonal=-1)

        logits = (self.delta[..., None, :] @ x.type(self.delta.dtype)[..., None]).squeeze() + self.beta

        p_two_unique_tokens = torch.sigmoid(logits)
        p_one_unique_token = 1 - p_two_unique_tokens
        return p_one_unique_token, p_two_unique_tokens

    def forward(self, g_values: torch.Tensor) -> torch.Tensor:
        """Computes the likelihoods P(g_values|watermarked).

        Args:
            g_values (`torch.Tensor` of shape `(batch_size, seq_len, watermarking_depth)`):
                g-values (values 0 or 1)

        Returns:
            p(g_values|watermarked) of shape [batch_size, seq_len, watermarking_depth].
        """
        p_one_unique_token, p_two_unique_tokens = self._compute_latents(g_values)

        return 0.5 * ((g_values + 0.5) * p_two_unique_tokens + p_one_unique_token)


class BayesianDetectorModel(PreTrainedModel):

    config: BayesianDetectorConfig
    base_model_prefix = "model"

    def __init__(self, config):
        super().__init__(config)

        self.watermarking_depth = config.watermarking_depth
        self.base_rate = config.base_rate
        self.likelihood_model_watermarked = BayesianDetectorWatermarkedLikelihood(
            watermarking_depth=self.watermarking_depth
        )
        self.prior = torch.nn.Parameter(torch.tensor([self.base_rate]))

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Parameter):
            init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_posterior(
        self,
        likelihoods_watermarked: torch.Tensor,
        likelihoods_unwatermarked: torch.Tensor,
        mask: torch.Tensor,
        prior: float,
    ) -> torch.Tensor:
        """
        Compute posterior P(w|g) given likelihoods, mask and prior.

        Args:
            likelihoods_watermarked (`torch.Tensor` of shape `(batch, length, depth)`):
                Likelihoods P(g_values|watermarked) of g-values under watermarked model.
            likelihoods_unwatermarked (`torch.Tensor` of shape `(batch, length, depth)`):
                Likelihoods P(g_values|unwatermarked) of g-values under unwatermarked model.
            mask (`torch.Tensor` of shape `(batch, length)`):
                A binary array indicating which g-values should be used. g-values with mask value 0 are discarded.
            prior (`float`):
                the prior probability P(w) that the text is watermarked.

        Returns:
            Posterior probability P(watermarked|g_values), shape [batch].
        """
        mask = torch.unsqueeze(mask, dim=-1)
        prior = torch.clamp(prior, min=1e-5, max=1 - 1e-5)
        log_likelihoods_watermarked = torch.log(torch.clamp(likelihoods_watermarked, min=1e-30, max=float("inf")))
        log_likelihoods_unwatermarked = torch.log(torch.clamp(likelihoods_unwatermarked, min=1e-30, max=float("inf")))
        log_odds = log_likelihoods_watermarked - log_likelihoods_unwatermarked

        relative_surprisal_likelihood = torch.einsum("i...->i", log_odds * mask)

        relative_surprisal_prior = torch.log(prior) - torch.log(1 - prior)

        relative_surprisal = relative_surprisal_prior + relative_surprisal_likelihood

        return torch.sigmoid(relative_surprisal)

    def forward(
        self,
        g_values: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        loss_batch_weight=1,
        return_dict=False,
    ) -> BayesianWatermarkDetectorModelOutput:
        """
        Computes the watermarked posterior P(watermarked|g_values).

        Args:
            g_values (`torch.Tensor` of shape `(batch_size, seq_len, watermarking_depth, ...)`):
                g-values (with values 0 or 1)
            mask:
                A binary array shape [batch_size, seq_len] indicating which g-values should be used. g-values with mask
                value 0 are discarded.

        Returns:
            p(watermarked | g_values), of shape [batch_size].
        """

        likelihoods_watermarked = self.likelihood_model_watermarked(g_values)
        likelihoods_unwatermarked = 0.5 * torch.ones_like(g_values)
        out = self._compute_posterior(
            likelihoods_watermarked=likelihoods_watermarked,
            likelihoods_unwatermarked=likelihoods_unwatermarked,
            mask=mask,
            prior=self.prior,
        )

        loss = None
        if labels is not None:
            loss_fct = BCELoss()
            loss_unwweight = torch.sum(self.likelihood_model_watermarked.delta**2)
            loss_weight = loss_unwweight * loss_batch_weight
            loss = loss_fct(torch.clamp(out, 1e-5, 1 - 1e-5), labels) + loss_weight

        if not return_dict:
            return (out,) if loss is None else (out, loss)

        return BayesianWatermarkDetectorModelOutput(loss=loss, posterior_probabilities=out)


class SynthIDTextWatermarkDetector:

    def __init__(
        self,
        detector_module: BayesianDetectorModel,
        logits_processor: SynthIDTextWatermarkLogitsProcessor,
        tokenizer: Any,
    ):
        self.detector_module = detector_module
        self.logits_processor = logits_processor
        self.tokenizer = tokenizer

    def __call__(self, tokenized_outputs: torch.Tensor):
        eos_token_mask = self.logits_processor.compute_eos_token_mask(
            input_ids=tokenized_outputs,
            eos_token_id=self.tokenizer.eos_token_id,
        )[:, self.logits_processor.ngram_len - 1 :]

        context_repetition_mask = self.logits_processor.compute_context_repetition_mask(
            input_ids=tokenized_outputs,
        )

        combined_mask = context_repetition_mask * eos_token_mask

        g_values = self.logits_processor.compute_g_values(
            input_ids=tokenized_outputs,
        )
        return self.detector_module(g_values, combined_mask)
