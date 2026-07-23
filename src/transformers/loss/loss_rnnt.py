
import torch

from ..utils import is_torchaudio_available, logging


logger = logging.get_logger(__name__)

if is_torchaudio_available():
    import torchaudio


def rnnt_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    logit_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_token_id: int,
    reduction: str = "mean_volume",
) -> torch.Tensor:
    pass


def ParakeetForRNNTLoss(
    logits,
    labels,
    logit_lengths,
    label_lengths,
    blank_token_id,
    reduction="mean_volume",
    **kwargs,
):
    pass
