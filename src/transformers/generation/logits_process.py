
import inspect
import math
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from .._typing import WhisperGenerationConfigLike
from ..utils import add_start_docstrings
from ..utils.logging import get_logger


if TYPE_CHECKING:
    from ..generation.configuration_utils import GenerationConfig

logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class LogitsProcessor:

    supports_continuous_batching: bool | None = None

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)

        return scores


class MinLengthLogitsProcessor(LogitsProcessor):

    supports_continuous_batching: bool = False

    def __init__(self, min_length: int, eos_token_id: int | list[int] | torch.Tensor, device: str = "cpu"):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        eos_token_mask = torch.isin(vocab_tensor, self.eos_token_id)
        scores_processed = scores.clone()
        if input_ids.shape[-1] < self.min_length:
            scores_processed = torch.where(eos_token_mask, -math.inf, scores)
        return scores_processed


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):

    supports_continuous_batching = False

    def __init__(
        self,
        prompt_length_to_skip: int,
        min_new_tokens: int,
        eos_token_id: int | list[int] | torch.Tensor,
        device: str = "cpu",
    ):
        for arg_name, arg_value in [
            ("prompt_length_to_skip", prompt_length_to_skip),
            ("min_new_tokens", min_new_tokens),
        ]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(f"`{arg_name}` has to be a positive integer, but is {arg_value}")

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)

        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
        scores_processed = scores.clone()
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        eos_token_mask = torch.isin(vocab_tensor, self.eos_token_id)
        if new_tokens_length < self.min_new_tokens:
            scores_processed = torch.where(eos_token_mask, -math.inf, scores)

        return scores_processed


class TemperatureLogitsWarper(LogitsProcessor):

    supports_continuous_batching = True

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        self.temperature = temperature

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores / self.temperature
        return scores_processed


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):

    supports_continuous_batching = False

    def __init__(self, penalty: float, prompt_ignore_length: int | None = None):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        if prompt_ignore_length is not None and (
            not isinstance(prompt_ignore_length, int) or prompt_ignore_length < 0
        ):
            raise ValueError(f"`prompt_ignore_length` has to be a positive integer, but is {prompt_ignore_length}")

        self.penalty = penalty
        self.prompt_ignore_length = prompt_ignore_length
        self.logits_indices = None
        self.cu_seq_lens_q = None

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.prompt_ignore_length:
            input_ids = input_ids[:, self.prompt_ignore_length :]

        if scores.dim() == 3:
            if self.logits_indices is not None and self.cu_seq_lens_q is not None:
                last_positions = self.logits_indices
                last_scores = scores[0, last_positions, :]

                token_mask = torch.zeros_like(last_scores, dtype=torch.bool)
                cu_seq_lens = self.cu_seq_lens_q
                lengths = cu_seq_lens[1:] - cu_seq_lens[:-1]
                seq_indices = torch.repeat_interleave(torch.arange(len(lengths), device=input_ids.device), lengths)
                token_mask[seq_indices, input_ids] = True

                penalty_scores = torch.where(last_scores < 0, last_scores * self.penalty, last_scores / self.penalty)
                scores[0, last_positions, :] = torch.where(token_mask, penalty_scores, last_scores)
            else:
                batch_size, seq_len, vocab_size = scores.shape
                last_scores = scores[:, -1, :]
                token_mask = torch.zeros_like(last_scores, dtype=torch.bool)
                if input_ids.dim() == 1:
                    unique_tokens = torch.unique(input_ids)
                    token_mask.scatter_(1, unique_tokens.unsqueeze(0), True)
                else:
                    token_mask.scatter_(1, input_ids, True)
                penalty_scores = torch.where(last_scores < 0, last_scores * self.penalty, last_scores / self.penalty)
                scores[:, -1, :] = torch.where(token_mask, penalty_scores, last_scores)
            return scores

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)

        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        scores_processed = scores.scatter(1, input_ids, score)
        return scores_processed


class EncoderRepetitionPenaltyLogitsProcessor(LogitsProcessor):

    supports_continuous_batching: bool = False

    def __init__(self, penalty: float, encoder_input_ids: torch.LongTensor):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = 1 / penalty
        self.encoder_input_ids = encoder_input_ids

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, self.encoder_input_ids)

        score = torch.where(score < 0, score * self.penalty, score / self.penalty)

        scores_processed = scores.scatter(1, self.encoder_input_ids, score)
        return scores_processed


class TopPLogitsWarper(LogitsProcessor):

    supports_continuous_batching = True

    def __init__(self, top_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class TopKLogitsWarper(LogitsProcessor):

    supports_continuous_batching = True

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep  # used for CB processor initialization

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class TopHLogitsWarper(LogitsProcessor):

    def __init__(self, top_h: float, filter_value: float = -float("Inf")):
        super().__init__()

        if not (0 < top_h <= 1):
            raise ValueError("`top_h` must be in the range (0, 1].")

        self.top_n = 100

        self.top_h = top_h
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Filters logits using Top-H sampling.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input token IDs.
            scores (`torch.FloatTensor` of shape `(batch_size, vocab_size)`):
                Raw logits from the model.

        Return:
            `torch.FloatTensor` of shape `(batch_size, vocab_size)`:
                Processed logits where invalid tokens are masked with `-inf`.
        """
        batch_size, vocab_size = scores.shape
        device = scores.device
        keep_mask = torch.zeros((batch_size, vocab_size), dtype=torch.bool, device=device)
        top_n = min(self.top_n, vocab_size)

        top_logits, top_idx = torch.topk(scores, top_n, dim=-1, largest=True, sorted=True)

        dist = torch.distributions.Categorical(logits=top_logits)
        probs = dist.probs
        log_probs = torch.log(probs)  # dist.log_prob(idx)

        tau = (dist.entropy() * self.top_h).unsqueeze(-1)

        entropy_terms = -probs * log_probs
        cumulative_entropy = torch.cumsum(entropy_terms, dim=-1)

        selection_mask = cumulative_entropy <= tau
        selection_mask[:, 0] = True

        keep_mask.scatter_(dim=1, index=top_idx, src=selection_mask)

        scores_processed = scores.clone()
        scores_processed[~keep_mask] = self.filter_value
        return scores_processed


class MinPLogitsWarper(LogitsProcessor):

    def __init__(self, min_p: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not (0 <= min_p <= 1.0):
            raise ValueError(f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.min_p = min_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores, dim=-1)
        top_probs = probs.amax(dim=-1, keepdim=True)
        scaled_min_p = self.min_p * top_probs
        tokens_to_remove = probs < scaled_min_p

        k = min(self.min_tokens_to_keep, probs.shape[-1])
        sorted_indices = torch.topk(probs, k, dim=-1).indices
        tokens_to_remove.scatter_(-1, sorted_indices, False)

        scores_processed = scores.masked_fill(tokens_to_remove, self.filter_value)
        return scores_processed


class TypicalLogitsWarper(LogitsProcessor):

    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        mass = float(mass)
        if not (mass > 0 and mass < 1):
            raise ValueError(f"`typical_p` has to be a float > 0 and < 1, but is {mass}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind.clamp_(max=sorted_scores.shape[-1] - 1)
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class EpsilonLogitsWarper(LogitsProcessor):

    def __init__(self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`epsilon_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = epsilon
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probabilities = scores.softmax(dim=-1)
        indices_to_remove = probabilities < self.epsilon

        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


class EtaLogitsWarper(LogitsProcessor):

    def __init__(
        self, epsilon: float, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, device: str = "cpu"
    ):
        epsilon = float(epsilon)
        if epsilon <= 0 or epsilon >= 1:
            raise ValueError(f"`eta_cutoff` has to be a float > 0 and < 1, but is {epsilon}")

        min_tokens_to_keep = int(min_tokens_to_keep)
        if min_tokens_to_keep < 1:
            raise ValueError(
                f"`min_tokens_to_keep` has to be a strictly positive integer, but is {min_tokens_to_keep}"
            )

        self.epsilon = torch.tensor(epsilon, device=device)
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probabilities = scores.softmax(dim=-1)
        entropy = torch.distributions.Categorical(logits=scores).entropy()
        eta = torch.min(self.epsilon, torch.sqrt(self.epsilon) * torch.exp(-entropy))[..., None]
        indices_to_remove = probabilities < eta

        top_k = min(self.min_tokens_to_keep, scores.size(-1))  # Safety check
        indices_to_remove = indices_to_remove & (scores < torch.topk(scores, top_k)[0][..., -1, None])

        scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores_processed


def _get_ngrams(ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int):
    """
    Assume ngram_size=2 and prev_input_ids=tensor([[40, 2883, 2712, 4346]]). The output of generated ngrams look like
    this {(40,): [2883], (2883,): [2712], (2712,): [4346]}.

    Args:
        ngram_size (`int`):
            The number sequential tokens taken as a group which may only occur once before being banned.
        prev_input_ids (`torch.Tensor`):
           Generated token ids for the current hypothesis.
        num_hypos (`int`):
            The number of hypotheses for which n-grams need to be generated.

    Returns:
        generated_ngrams (`dict`):
            Dictionary of generated ngrams.
    """
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]
    return generated_ngrams


def _get_generated_ngrams(banned_ngrams, prev_input_ids, ngram_size, cur_len):
    pass


def _calc_banned_ngram_tokens(
    ngram_size: int, prev_input_ids: torch.Tensor, num_hypos: int, cur_len: int
) -> list[Iterable[int]]:
    pass


class NoRepeatNGramLogitsProcessor(LogitsProcessor):

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]
        scores_processed = scores.clone()
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed


class EncoderNoRepeatNGramLogitsProcessor(LogitsProcessor):

    def __init__(self, encoder_ngram_size: int, encoder_input_ids: torch.LongTensor):
        if not isinstance(encoder_ngram_size, int) or encoder_ngram_size <= 0:
            raise ValueError(
                f"`encoder_ngram_size` has to be a strictly positive integer, but is {encoder_ngram_size}"
            )
        self.ngram_size = encoder_ngram_size
        if len(encoder_input_ids.shape) == 1:
            encoder_input_ids = encoder_input_ids.unsqueeze(0)
        self.batch_size = encoder_input_ids.shape[0]
        self.generated_ngrams = _get_ngrams(encoder_ngram_size, encoder_input_ids, self.batch_size)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        num_hypos = scores.shape[0]
        num_beams = num_hypos // self.batch_size
        cur_len = input_ids.shape[-1]
        scores_processed = scores.clone()
        banned_batch_tokens = [
            _get_generated_ngrams(
                self.generated_ngrams[hypo_idx // num_beams], input_ids[hypo_idx], self.ngram_size, cur_len
            )
            for hypo_idx in range(num_hypos)
        ]

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed


class SequenceBiasLogitsProcessor(LogitsProcessor):

    def __init__(self, sequence_bias: list[list[list[int] | float]]):
        self.sequence_bias: Any = sequence_bias
        self._validate_arguments()
        self._convert_list_arguments_into_dict()

        self.length_1_bias = None
        self.prepared_bias_variables = False

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.prepared_bias_variables:
            self._prepare_bias_variables(scores)

        bias = torch.zeros_like(scores)

        bias += self.length_1_bias

        for sequence_ids, sequence_bias in self.sequence_bias.items():
            if len(sequence_ids) == 1:  # the sequence is of length 1, already applied
                continue
            if len(sequence_ids) > input_ids.shape[1]:  # the sequence is longer than the context, ignore
                continue
            prefix_length = len(sequence_ids) - 1
            last_token = sequence_ids[-1]
            matching_rows = torch.eq(
                input_ids[:, -prefix_length:],
                torch.tensor(sequence_ids[:-1], dtype=input_ids.dtype, device=input_ids.device),
            ).prod(dim=1)
            bias[:, last_token] += torch.where(
                matching_rows.bool(),
                torch.tensor(sequence_bias, device=input_ids.device),
                torch.tensor(0.0, device=input_ids.device),
            )

        scores_processed = scores + bias
        return scores_processed

    def _prepare_bias_variables(self, scores: torch.FloatTensor):
        pass

    def _validate_arguments(self):
        sequence_bias = self.sequence_bias
        if not isinstance(sequence_bias, dict) and not isinstance(sequence_bias, list) or len(sequence_bias) == 0:
            raise ValueError(
                f"`sequence_bias` has to be a non-empty dictionary, or non-empty list of lists but is {sequence_bias}."
            )
        if isinstance(sequence_bias, dict) and any(
            not isinstance(sequence_ids, tuple) for sequence_ids in sequence_bias
        ):
            raise ValueError(f"`sequence_bias` has to be a dict with tuples as keys, but is {sequence_bias}.")
        if isinstance(sequence_bias, dict) and any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in sequence_ids)
            or len(sequence_ids) == 0
            for sequence_ids in sequence_bias
        ):
            raise ValueError(
                f"Each key in `sequence_bias` has to be a non-empty tuple of positive integers, but is "
                f"{sequence_bias}."
            )

        def all_token_bias_pairs_are_valid(sequence):
            return (
                isinstance(sequence[0], list)
                and all(isinstance(token_id, (int, np.integer)) and token_id > 0 for token_id in sequence[0])
                and isinstance(sequence[1], float)
            )

        if isinstance(sequence_bias, list) and any(
            (not all_token_bias_pairs_are_valid(sequence)) or len(sequence) == 0 for sequence in sequence_bias
        ):
            raise ValueError(
                f"Each element in `sequence_bias` has to be a non-empty list of lists of positive integers and float, but is "
                f"{sequence_bias}."
            )
        if isinstance(sequence_bias, dict) and any(not isinstance(bias, float) for bias in sequence_bias.values()):
            raise ValueError(f"`sequence_bias` has to be a dict with floats as values, but is {sequence_bias}.")

    def _convert_list_arguments_into_dict(self):
        """BC: we used to accept `dict{tuple of tokens: float}` directly, now we expect a list"""
        if isinstance(self.sequence_bias, list):
            temp_sequence = self.sequence_bias
            self.sequence_bias = {tuple(sublist[0]): sublist[1] for sublist in temp_sequence}


class NoBadWordsLogitsProcessor(SequenceBiasLogitsProcessor):

    def __init__(self, bad_words_ids: list[list[int]], eos_token_id: int | list[int] | torch.Tensor | None = None):
        self.bad_word_ids = bad_words_ids
        self._validate_arguments()

        if eos_token_id is not None:
            if not isinstance(eos_token_id, torch.Tensor):
                if isinstance(eos_token_id, int):
                    eos_token_id = [eos_token_id]
                eos_token_id = torch.tensor(eos_token_id)

            eos_token_id_list = eos_token_id.tolist()  # convert to python list before
            bad_words_ids = list(
                filter(lambda bad_token_seq: all(bad_token_seq != [i] for i in eos_token_id_list), bad_words_ids)
            )
        sequence_bias = {tuple(sequence): float("-inf") for sequence in bad_words_ids}
        super().__init__(sequence_bias=sequence_bias)

    def _validate_arguments(self):
        bad_words_ids = self.bad_word_ids
        if not isinstance(bad_words_ids, list) or len(bad_words_ids) == 0:
            raise ValueError(f"`bad_words_ids` has to be a non-empty list, but is {bad_words_ids}.")
        if any(not isinstance(bad_word_ids, list) for bad_word_ids in bad_words_ids):
            raise ValueError(f"`bad_words_ids` has to be a list of lists, but is {bad_words_ids}.")
        if any(
            any((not isinstance(token_id, (int, np.integer)) or token_id < 0) for token_id in bad_word_ids)
            for bad_word_ids in bad_words_ids
        ):
            raise ValueError(
                f"Each list in `bad_words_ids` has to be a list of positive integers, but is {bad_words_ids}."
            )


class PrefixConstrainedLogitsProcessor(LogitsProcessor):

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], list[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -math.inf)
        batch_size = input_ids.shape[0] // self._num_beams

        for batch_id in range(batch_size):
            for beam_id in range(self._num_beams):
                sent = input_ids[batch_id * self._num_beams + beam_id]
                prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                if len(prefix_allowed_tokens) == 0:
                    raise ValueError(
                        f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                        f"This means that the constraint is unsatisfiable. Please check your implementation"
                        f"of `prefix_allowed_tokens_fn` "
                    )
                mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        scores_processed = scores + mask
        return scores_processed


class ForcedBOSTokenLogitsProcessor(LogitsProcessor):

    def __init__(self, bos_token_id: int):
        self.bos_token_id = bos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        scores_processed = scores
        if cur_len == 1:
            scores_processed = torch.full_like(scores, -math.inf)
            scores_processed[:, self.bos_token_id] = 0
        return scores_processed


class ForcedEOSTokenLogitsProcessor(LogitsProcessor):

    def __init__(self, max_length: int, eos_token_id: int | list[int] | torch.Tensor, device: str = "cpu"):
        self.max_length = max_length

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_id = eos_token_id

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        scores_processed = scores
        if cur_len == self.max_length - 1:
            scores_processed = torch.full_like(scores, -math.inf)
            scores_processed[:, self.eos_token_id] = 0
        return scores_processed


class InfNanRemoveLogitsProcessor(LogitsProcessor):

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = torch.where(scores != scores, 0.0, scores)

        scores_processed = torch.where(scores == float("inf"), torch.finfo(scores.dtype).max, scores_processed)
        scores_processed = torch.where(scores == -float("inf"), torch.finfo(scores.dtype).min, scores_processed)

        return scores_processed


class ExponentialDecayLengthPenalty(LogitsProcessor):

    def __init__(
        self,
        exponential_decay_length_penalty: tuple[int, float],
        eos_token_id: int | list[int] | torch.Tensor,
        input_ids_seq_length: int,
    ):
        self.regulation_start = exponential_decay_length_penalty[0] + input_ids_seq_length
        self.regulation_factor = exponential_decay_length_penalty[1]

        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)
        self.eos_token_id = eos_token_id

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        self.eos_token_id = self.eos_token_id.to(scores.device)
        penalties = torch.zeros_like(scores)
        scores_processed = scores
        if cur_len > self.regulation_start:
            penalty_idx = cur_len - self.regulation_start
            penalty = torch.abs(scores[:, self.eos_token_id]) * (pow(self.regulation_factor, penalty_idx) - 1)
            penalties[:, self.eos_token_id] = penalty
            scores_processed = scores + penalties
        return scores_processed


class LogitNormalization(LogitsProcessor):

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores.log_softmax(dim=-1)
        return scores_processed


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):

    def __init__(self, begin_suppress_tokens, begin_index, device: str = "cpu"):
        self.begin_suppress_tokens = torch.tensor(list(begin_suppress_tokens), device=device)
        self.begin_index = begin_index

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        suppress_token_mask = torch.isin(vocab_tensor, self.begin_suppress_tokens.to(scores.device))
        scores_processed = scores
        if input_ids.shape[-1] == self.begin_index:
            scores_processed = torch.where(suppress_token_mask, -float("inf"), scores)

        return scores_processed


class SuppressTokensLogitsProcessor(LogitsProcessor):

    def __init__(self, suppress_tokens, device: str = "cpu"):
        self.suppress_tokens = torch.tensor(list(suppress_tokens), device=device)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        suppress_token_mask = torch.isin(vocab_tensor, self.suppress_tokens.to(scores.device))
        scores = torch.where(suppress_token_mask, -float("inf"), scores)
        return scores


class WhisperTimeStampLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        generate_config: "GenerationConfig",
        begin_index: int,
        _detect_timestamp_from_logprob: bool | None = None,
    ):  # support for the kwargs
        whisper_generate_config = cast(WhisperGenerationConfigLike, generate_config)
        self.no_timestamps_token_id = whisper_generate_config.no_timestamps_token_id
        self.timestamp_begin = whisper_generate_config.no_timestamps_token_id + 1
        self.eos_token_id = generate_config.eos_token_id or generate_config.bos_token_id

        self._detect_timestamp_from_logprob = (
            _detect_timestamp_from_logprob
            if _detect_timestamp_from_logprob is not None
            else getattr(generate_config, "_detect_timestamp_from_logprob", True)
        )
        self.begin_index = begin_index
        if begin_index is None:
            raise ValueError(
                "`forced_decoder_ids` is deprecated in favor of `task` and `language` and, as such, `begin_index` "
                "must be provided to `WhisperTimeStampLogitsProcessor`. The previous default value of `begin_index` "
                "was `len(generate_config.forced_decoder_ids)`"
            )

        self.max_initial_timestamp_index = getattr(generate_config, "max_initial_timestamp_index", None)

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores.clone()
        scores_processed[:, self.no_timestamps_token_id] = -float("inf")

        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index :]
            seq = list(sampled_tokens.tolist())

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    scores_processed[k, self.timestamp_begin :] = -float("inf")
                else:  # cannot be normal text tokens
                    scores_processed[k, : self.eos_token_id] = -float("inf")

            timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    timestamp_last = timestamps[-1] + 1

                scores_processed[k, self.timestamp_begin : timestamp_last] = -float("inf")

        if input_ids.shape[1] == self.begin_index:
            scores_processed[:, : self.timestamp_begin] = -float("inf")

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores_processed[:, last_allowed + 1 :] = -float("inf")

        logprobs = torch.nn.functional.log_softmax(scores_processed.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores_processed[k, : self.timestamp_begin] = -float("inf")

        return scores_processed


class WhisperNoSpeechDetection(LogitsProcessor):

    def __init__(self, no_speech_token: int, begin_index: int, scores_is_logprobs: bool = False):
        self.no_speech_token = no_speech_token
        self.start_of_trans_offset = begin_index

        self.begin_index = begin_index
        self._no_speech_prob = [0.0]
        self.is_scores_logprobs = scores_is_logprobs

        self.model: Any = None
        self.inputs: dict[str, Any] | None = None

    def set_model(self, model):
        self.model = model

    def set_inputs(self, inputs):
        self.inputs = {**self.model.prepare_inputs_for_generation(**inputs), **inputs}
        self.inputs["input_features"] = self.inputs.pop("inputs")

        if "input_ids" not in inspect.signature(self.model.forward).parameters:
            self.inputs.pop("input_ids", None)

    @property
    def no_speech_prob(self):
        pass

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        is_scores_logprobs = self.is_scores_logprobs

        if input_ids.shape[1] == self.begin_index:
            if self.start_of_trans_offset > 1:
                with torch.no_grad():
                    logits = self.model(**self.inputs).logits

                no_speech_index = self.begin_index - self.start_of_trans_offset
                no_speech_scores = logits[:, no_speech_index]
                is_scores_logprobs = False
            else:
                no_speech_scores = scores

            if is_scores_logprobs:
                probs = no_speech_scores.exp()
            else:
                probs = no_speech_scores.float().softmax(dim=-1)

            self._no_speech_prob = probs[:, self.no_speech_token]

        return scores


class ClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):

    def __init__(self, guidance_scale):
        if guidance_scale > 1:
            self.guidance_scale = guidance_scale
        else:
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        unguided_bsz = scores.shape[0] // 2
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        scores_processed = uncond_logits + (cond_logits - uncond_logits) * self.guidance_scale
        return scores_processed


class AlternatingCodebooksLogitsProcessor(LogitsProcessor):

    def __init__(self, input_start_len: int, semantic_vocab_size: int, codebook_size: int):
        if not isinstance(input_start_len, int) or input_start_len < 0:
            raise ValueError(f"`input_starting_length` has to be a non-negative integer, but is {input_start_len}")

        self.input_start_len = input_start_len
        self.semantic_vocab_size = semantic_vocab_size
        self.codebook_size = codebook_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        curr_len = input_ids.shape[-1]

        is_first_codebook = ((curr_len - self.input_start_len) % 2) == 0

        scores_processed = scores.clone()
        if is_first_codebook:
            scores_processed[:, : self.semantic_vocab_size] = -float("inf")
            scores_processed[:, self.semantic_vocab_size + self.codebook_size :] = -float("inf")
        else:
            scores_processed[:, : self.semantic_vocab_size + self.codebook_size] = -float("inf")

        return scores_processed


class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: torch.LongTensor | None = None,
        unconditional_attention_mask: torch.LongTensor | None = None,
        use_cache: bool = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
        pass

    def __call__(self, input_ids, scores):
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        logits = self.get_unconditional_logits(input_ids)

        unconditional_logits = torch.nn.functional.log_softmax(logits[:, -1], dim=-1)
        scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return scores_processed


class BarkEosPrioritizerLogitsProcessor(LogitsProcessor):

    def __init__(self, eos_token_id: int | list[int] | torch.Tensor, min_eos_p: float, device: str = "cpu"):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id, device=device)
        self.eos_token_id = eos_token_id

        if torch.is_floating_point(eos_token_id) or (eos_token_id < 0).any():
            raise ValueError(f"`eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        if min_eos_p is not None and min_eos_p <= 0:
            raise ValueError(f"`min_eos_p` has to be a positive float, but is {min_eos_p}")
        self.min_eos_p = min_eos_p

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_processed = scores
        if self.min_eos_p:
            probs = torch.nn.functional.softmax(scores.float(), dim=-1)
            early_stop_scores = torch.ones_like(scores) * -float("inf")
            early_stop_scores[:, self.eos_token_id] = scores[:, self.eos_token_id]

            do_early_stop = probs[:, self.eos_token_id] > self.min_eos_p
            do_early_stop = torch.any(do_early_stop, dim=1, keepdim=True)
            scores_processed = torch.where(do_early_stop, early_stop_scores, scores)

        return scores_processed


class WatermarkLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        vocab_size,
        device,
        greenlist_ratio: float = 0.25,
        bias: float = 2.0,
        hashing_key: int = 15485863,
        seeding_scheme: str = "lefthash",
        context_width: int = 1,
    ):
        if seeding_scheme not in ["selfhash", "lefthash"]:
            raise ValueError(f"seeding_scheme has to be one of [`selfhash`, `lefthash`], but found {seeding_scheme}")
        if greenlist_ratio >= 1.0 or greenlist_ratio <= 0.0:
            raise ValueError(
                f"greenlist_ratio has be in range between 0.0 and 1.0, exclusively. but found {greenlist_ratio}"
            )

        self.vocab_size = vocab_size
        self.greenlist_size = int(self.vocab_size * greenlist_ratio)
        self.bias = bias
        self.seeding_scheme = seeding_scheme
        self.rng = torch.Generator(device=device)
        self.hash_key = hashing_key
        self.context_width = context_width

        self.rng.manual_seed(hashing_key)
        self.table_size = 1_000_003
        self.fixed_table = torch.randperm(self.table_size, generator=self.rng, device=device)

    def set_seed(self, input_seq: torch.LongTensor):
        input_seq = input_seq[-self.context_width :]
        if self.seeding_scheme == "selfhash":
            a = self.fixed_table[input_seq % self.table_size] + 1
            b = self.fixed_table[input_seq[-1] % self.table_size] + 1
            seed = (self.hash_key * a * b).min().item()
        else:
            seed = self.hash_key * input_seq[-1].item()
        self.rng.manual_seed(seed % (2**64 - 1))

    def _get_greenlist_ids(self, input_seq: torch.LongTensor) -> torch.LongTensor:
        pass

    def _score_rejection_sampling(self, input_seq: torch.LongTensor, scores: torch.FloatTensor) -> torch.LongTensor:
        pass

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.context_width:
            logger.warning(
                f"`input_ids` should have at least `{self.context_width}` tokens but has {input_ids.shape[-1]}. "
                "The seeding will be skipped for this generation step!"
            )
            return scores

        scores_processed = scores.clone()
        for b_idx, input_seq in enumerate(input_ids):
            if self.seeding_scheme == "selfhash":
                greenlist_ids = self._score_rejection_sampling(input_seq, scores[b_idx])
            else:
                greenlist_ids = self._get_greenlist_ids(input_seq)
            scores_processed[b_idx, greenlist_ids] = scores_processed[b_idx, greenlist_ids] + self.bias

        return scores_processed


class SynthIDTextWatermarkState:

    def __init__(
        self,
        batch_size: int,
        ngram_len: int,
        context_history_size: int,
        device: torch.device,
    ):
        """Initializes the state.

        Args:
            batch_size (`int`): Batch size.
            ngram_len (`int`): Ngram length.
            context_history_size (`int`): Size of the tensor to keep track of seen contexts.
            device (`int`): Device to use.
        """
        self.context = torch.zeros(
            (batch_size, ngram_len - 1),
            dtype=torch.int64,
            device=device,
        )
        self.context_history = torch.zeros(
            (batch_size, context_history_size),
            dtype=torch.int64,
            device=device,
        )
        self.num_calls = 0


class SynthIDTextWatermarkLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        ngram_len: int,
        keys: list[int],
        sampling_table_size: int,
        sampling_table_seed: int,
        context_history_size: int,
        device: torch.device,
        skip_first_ngram_calls: bool = False,
        debug_mode: bool = False,
    ):
        self.ngram_len = ngram_len
        self.keys = torch.tensor(keys, device=device)

        generator = torch.Generator(device=device).manual_seed(sampling_table_seed)
        self.sampling_table = torch.randint(
            low=0,
            high=2,
            size=(sampling_table_size,),
            generator=generator,
            device=device,
        )
        self.context_history_size = context_history_size
        self.device = device
        self.state = None
        self.skip_first_ngram_calls = skip_first_ngram_calls
        self.debug_mode = debug_mode

    def _init_state(self, batch_size: int):
        pass

    def update_scores(self, scores: torch.FloatTensor, g_values: torch.FloatTensor) -> torch.FloatTensor:
        pass

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self._check_input_ids_shape(input_ids)
        batch_size, vocab_size = scores.shape

        if self.debug_mode:
            scores = torch.ones_like(scores)

        all_indices = torch.arange(vocab_size, device=self.device).unsqueeze(0).expand(batch_size, -1)

        if self.state is None:
            self._init_state(batch_size)
        else:
            self.state.context = torch.concat(
                (self.state.context, input_ids[:, -1:]),
                dim=1,
            )
            self.state.context = self.state.context[:, 1:]

        if self.state is None:
            raise ValueError("self.state can't be None! Call `self._init_state` to initialize the state.")

        self.state.num_calls += 1

        if self.skip_first_ngram_calls and self.state.num_calls < self.ngram_len:
            return scores

        ngram_keys, hash_result_with_just_context = self._compute_keys(self.state.context, all_indices)

        g_values = self.sample_g_values(ngram_keys)

        updated_scores = self.update_scores(scores, g_values)

        hash_result_with_just_context = hash_result_with_just_context[:, None]
        is_repeated_context = (self.state.context_history == hash_result_with_just_context).any(
            dim=1,
            keepdim=True,
        )
        self.state.context_history = torch.concat(
            (hash_result_with_just_context, self.state.context_history),
            dim=1,
        )[:, :-1]

        updated_watermarked_scores = torch.where(
            is_repeated_context,
            input=scores,
            other=updated_scores,
        )
        return updated_watermarked_scores

    def accumulate_hash(
        self,
        current_hash: torch.LongTensor,
        data: torch.LongTensor,
        multiplier: int = 6364136223846793005,
        increment: int = 1,
    ) -> torch.LongTensor:
        pass

    def compute_ngram_keys(self, ngrams: torch.LongTensor) -> torch.LongTensor:
        pass

    def _compute_keys(
        self, n_minus_1_grams: torch.LongTensor, indices: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        pass

    def sample_g_values(self, ngram_keys: torch.LongTensor) -> torch.LongTensor:
        pass

    def _check_input_ids_shape(self, input_ids: torch.LongTensor):
        pass

    def compute_g_values(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        pass

    def compute_context_repetition_mask(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        pass

    def compute_eos_token_mask(self, input_ids: torch.LongTensor, eos_token_id: int) -> torch.LongTensor:
        pass

    def expected_mean_g_value(self, vocab_size: int, coinflip_prob: float = 0.5) -> float:
        pass


class DiaClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):

    def __init__(self, guidance_scale: float, guidance_top_k: int | None = None):
        if guidance_scale > 1:
            self.guidance_scale = guidance_scale
        else:
            raise ValueError(
                "Require guidance scale >1 to use the classifier free guidance processor, got guidance scale "
                f"{guidance_scale}."
            )

        self.guidance_top_k = guidance_top_k
        if self.guidance_top_k is not None and self.guidance_top_k < 1:
            raise ValueError(
                f"`guidance_top_k` has to be a strictly positive integer if given, but is {self.guidance_top_k}"
            )

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[0] != 2 * input_ids.shape[0]:
            raise ValueError(
                f"Logits should have twice the batch size of the input ids, the first half of batches corresponding to "
                f"the conditional inputs, and the second half of batches corresponding to the unconditional inputs. Got "
                f"batch size {scores.shape[0]} for the logits and {input_ids.shape[0]} for the input ids."
            )
        unguided_bsz = scores.shape[0] // 2
        cond_logits, uncond_logits = scores.split(unguided_bsz, dim=0)
        scores_processed = cond_logits + (cond_logits - uncond_logits) * self.guidance_scale

        if self.guidance_top_k is not None:
            _, top_k_indices = torch.topk(scores_processed, k=self.guidance_top_k, dim=-1)
            top_k_mask = torch.ones_like(scores_processed, dtype=torch.bool)
            top_k_mask = top_k_mask.scatter(dim=-1, index=top_k_indices, value=False)
            scores_processed = cond_logits.masked_fill(top_k_mask, -float("inf"))

        return scores_processed


class DiaEOSChannelFilterLogitsProcessor(LogitsProcessor):

    def __init__(self, num_channels: int, eos_token_id: int):
        if num_channels < 1:
            raise ValueError(f"Audio codebooks need at least one channel, but found {num_channels} channels.")
        if eos_token_id < 1:
            raise ValueError(f"Expected `eos_token_id` to be a positive integer, found {eos_token_id} instead.")

        self.num_channels = num_channels
        self.eos_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.reshape(-1, self.num_channels, scores.shape[-1])

        scores[:, 1:, self.eos_id :] = torch.full_like(
            scores[:, 1:, self.eos_id :],
            fill_value=-float("inf"),
        )
        scores[:, 0, self.eos_id + 1 :] = torch.full_like(
            scores[:, 0, self.eos_id + 1 :],
            fill_value=-float("inf"),
        )

        scores = scores.view(-1, scores.shape[-1])

        top_logit_indices = torch.argmax(scores, dim=-1)

        eos_highest_mask = top_logit_indices == self.eos_id
        mask_eos_highest = torch.zeros_like(scores, dtype=torch.bool)
        mask_eos_highest[eos_highest_mask, : self.eos_id] = True
        scores = scores.masked_fill(mask_eos_highest, -float("inf"))

        eos_not_highest_mask = top_logit_indices != self.eos_id
        mask_eos_unless_highest = torch.zeros_like(scores, dtype=torch.bool)
        mask_eos_unless_highest[eos_not_highest_mask, self.eos_id] = True
        scores = scores.masked_fill(mask_eos_unless_highest, -float("inf"))

        return scores


class DiaEOSDelayPatternLogitsProcessor(LogitsProcessor):

    def __init__(self, delay_pattern: list[int], eos_token_id: int, max_generation_len: int, device: str = "cpu"):
        self.num_channels = len(delay_pattern)
        self.active_batches = None
        self.delay_pattern = torch.tensor(delay_pattern, device=device, dtype=torch.int)[None, :]
        self.eos_token_id = eos_token_id
        self.max_generation_len = max_generation_len - max(delay_pattern) - 1
        self.device = device

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.reshape(-1, self.num_channels, scores.shape[-1])

        if self.active_batches is None:
            self.delay_pattern = self.delay_pattern.repeat(scores.shape[0], 1)
            self.active_batches = torch.zeros(size=(scores.shape[0],), device=self.device, dtype=torch.bool)

        channel_generated_eos = torch.argmax(scores, dim=-1)[:, 0] == self.eos_token_id
        reached_max_len = input_ids.shape[1] == self.max_generation_len

        self.active_batches |= channel_generated_eos
        self.active_batches |= reached_max_len

        forced_eos_channels = self.active_batches[:, None] & (self.delay_pattern == 0)
        idx_bsz, idx_channel = forced_eos_channels.nonzero(as_tuple=True)

        scores[idx_bsz, idx_channel, :] = -float("inf")
        scores[idx_bsz, idx_channel, self.eos_token_id] = 0.0

        scores = scores.reshape(-1, scores.shape[-1])

        self.delay_pattern -= self.active_batches[:, None].int()

        return scores
