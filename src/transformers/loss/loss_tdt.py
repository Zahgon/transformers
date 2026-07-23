
import torch

from ..utils import logging


logger = logging.get_logger(__name__)


def tdt_loss(
    token_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_token_id: int,
    durations: list[int],
    sigma: float = 0.0,
    reduction: str = "mean",
) -> torch.Tensor:
    pass


def ParakeetForTDTLoss(
    token_logits,
    duration_logits,
    labels,
    logit_lengths,
    label_lengths,
    blank_token_id,
    durations,
    sigma=0.0,
    reduction="mean",
    **kwargs,
):
    pass
